#!/usr/bin/env python3
"""Collect real Russian App Store reviews for human CEDR-M7 annotation.

Dependencies: pyarrow and requests, both declared in pyproject.toml.
Raw source snapshots stay under the git-ignored data/raw directory.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
REVISION = "7c72eb7322a00b531365e678bad8bba77e4bc8e3"
EXPECTED_ROWS = {"train": 7528, "test": 1882}
APPS = {
    "310633997": "WhatsApp Messenger",
    "686449807": "Telegram Messenger",
    "472650686": "Яндекс Go",
    "313877526": "Яндекс Карты",
    "520797969": "Яндекс Музыка",
    "407804998": "OZON",
    "597880187": "WILDBERRIES",
    "1367959794": "Госуслуги",
    "894649641": "Додо Пицца",
    "481627348": "2ГИС",
    "477718890": "Кинопоиск",
    "1078986931": "Яндекс Еда",
}
COLLECTED_FIELDS = ["item_id", "text", "source_url", "accessed_at", "license"]
METADATA_FIELDS = [
    "item_id", "source_record_id", "app_id", "app_name", "title", "rating",
    "updated_at", "country", "snapshot_path", "snapshot_sha256", "text_sha256",
]
ANNOTATION_FIELDS = [
    "item_id", "text", "annotator_1_labels", "annotator_2_labels", "final_labels", "notes",
]
CONTACT = re.compile(
    r"https?://|www\.|[\w.+-]+@[\w.-]+\.[a-zа-я]{2,}|(?<!\w)@\w{3,}"
    r"|(?<!\w)(?:\+?7|8)[\s(-]*\d{3}[\s)-]*\d{3}[-\s]*\d{2}[-\s]*\d{2}(?!\w)",
    re.IGNORECASE,
)
CUES = {
    "disgust": r"отврат|омерз|тошн|мерзк|гадост|противн",
    "anger": r"бесит|бешен|злит|злюсь|ненавиж|возмут|раздраж|достал",
    "fear": r"страш|боюсь|боял|испуг|пуга|опаса|тревож",
    "enthusiasm": r"удив|неожидан|ошелом|изум|не ожидал",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    # Same punctuation removal as text.ipynb, plus Unicode and ё/е normalization.
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]", "", text).split())


def prepare_text(text):
    # Keep case, punctuation, emoji, spelling and negations. Only fold whitespace.
    return " ".join(text.split())


def rejection_reason(text):
    if not 40 <= len(text) <= 1500:
        return "length_outside_40_1500"
    letters = sum(ch.isalpha() for ch in text)
    russian = len(re.findall(r"[а-яё]", text, re.IGNORECASE))
    if not letters or russian / letters < 0.65 or len(re.findall(r"[а-яё]+", text, re.I)) < 6:
        return "not_enough_russian_text"
    if CONTACT.search(text):
        return "contact_or_external_link"
    return None


class NearIndex:
    """Find copies with Jaccard >= .90 over word trigrams (8+ words only)."""

    def __init__(self):
        self.postings = defaultdict(set)
        self.sizes = {}

    @staticmethod
    def shingles(text):
        words = text.split()
        if len(words) < 8:
            return set()
        return set(zip(words, words[1:], words[2:]))

    def add(self, key, text):
        parts = self.shingles(text)
        self.sizes[key] = len(parts)
        for part in parts:
            self.postings[part].add(key)

    def contains(self, text):
        parts = self.shingles(text)
        matches = Counter(key for part in parts for key in self.postings.get(part, ()))
        return any(count / (len(parts) + self.sizes[key] - count) >= 0.90
                   for key, count in matches.items())


def download(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=40, headers={"User-Agent": "CourseworkReviewCollector/1.0"})
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            if (isinstance(error, requests.HTTPError)
                    and error.response.status_code not in {408, 429, 500, 502, 503, 504}):
                raise
            if attempt == 2:
                raise
            time.sleep(1 + attempt * 2)
    raise RuntimeError("Download failed")


def baseline(raw, offline):
    import pyarrow.parquet as pq

    exact, near, records = set(), NearIndex(), []
    directory = raw / "cedr-reference" / REVISION
    directory.mkdir(parents=True, exist_ok=True)
    for split, expected in EXPECTED_ROWS.items():
        url = f"https://huggingface.co/datasets/Aniemore/cedr-m7/resolve/{REVISION}/data/{split}-00000-of-00001.parquet"
        path = directory / f"{split}.parquet"
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Missing baseline cache: {path}")
            path.write_bytes(download(url))
        table = pq.read_table(path, columns=["text"])
        if len(table) != expected:
            raise ValueError(f"CEDR-M7 {split}: expected {expected} rows, got {len(table)}")
        for i, text in enumerate(table.column("text").to_pylist()):
            clean = normalize(text)
            exact.add(clean)
            near.add(f"{split}:{i}", clean)
        records.append({"split": split, "rows": len(table), "url": url,
                        "sha256": sha256(path.read_bytes())})
    return exact, near, records


def snapshot(raw, batch, app_id, page, offline):
    path = raw / "review_collection" / batch / f"{app_id}_page_{page}.json"
    url = f"https://itunes.apple.com/ru/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Missing review cache: {path}")
        payload = download(url)
        feed = json.loads(payload)["feed"]
        entries = feed.get("entry", [])
        if isinstance(entries, dict):
            entries = [entries]
        # Save source text and review IDs, without reviewer names or profile URLs.
        data = {"source_url": url, "accessed_at": datetime.now(timezone.utc).isoformat(),
                "response_sha256": sha256(payload), "entries": [
                    {key: entry.get(key, {}) for key in
                     ("id", "title", "content", "updated", "im:rating", "im:version")}
                    for entry in entries if "im:rating" in entry
                ]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        time.sleep(0.4)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["source_url"] != url:
        raise ValueError(f"Snapshot URL mismatch: {path}")
    return data, path, sha256(path.read_bytes())


def read_csv(path, fields):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected CSV schema: {path}")
        rows = list(reader)
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate item_id in {path}")
    return rows


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def interleave_candidates(candidates, seed):
    # Alternate apps and star groups to avoid taking only one application's complaints.
    rng = random.Random(seed)
    queues = []
    for app_id in APPS:
        groups = [[], [], []]
        for row in candidates[app_id]:
            rating = int(row["metadata"]["rating"])
            groups[0 if rating <= 2 else 1 if rating == 3 else 2].append(row)
        for group in groups:
            rng.shuffle(group)
        queues.append(deque(row for batch in zip_longest(*groups) for row in batch if row))
    while any(queues):
        for queue in queues:
            if queue:
                yield queue.popleft()


def collect(args):
    output = ROOT / "data/manual/text"
    raw = ROOT / "data/raw"
    collected = read_csv(output / "collected.csv", COLLECTED_FIELDS)
    metadata = {r["item_id"]: r for r in read_csv(output / "metadata.csv", METADATA_FIELDS)}
    annotations = {r["item_id"]: r for r in read_csv(output / "annotations.csv", ANNOTATION_FIELDS)}
    known_ids = {row["item_id"] for row in collected}
    known_texts = {normalize(row["text"]) for row in collected}
    own_near = NearIndex()
    for row in collected:
        own_near.add(row["item_id"], normalize(row["text"]))
    reference, reference_near, reference_records = baseline(raw, args.offline)
    candidates, filtered, fetched = defaultdict(list), Counter(), []
    source_ids = set()
    for app_id, app_name in APPS.items():
        count = 0
        for page in range(1, args.pages + 1):
            data, path, digest = snapshot(raw, args.batch, app_id, page, args.offline)
            fetched.append({"url": data["source_url"], "accessed_at": data["accessed_at"],
                            "snapshot_path": path.relative_to(ROOT).as_posix(), "sha256": digest,
                            "entries": len(data["entries"])})
            for entry in data["entries"]:
                source_id = entry["id"]["label"]
                item_id = f"appstore-ru-{app_id}-{source_id}"
                if item_id in source_ids:
                    filtered["repeated_source_id"] += 1
                    continue
                source_ids.add(item_id)
                text = prepare_text(entry["content"]["label"])
                reason = rejection_reason(text)
                if reason:
                    filtered[reason] += 1
                    continue
                clean = normalize(text)
                if clean in reference:
                    filtered["exact_match_cedr"] += 1
                    continue
                if reference_near.contains(clean):
                    filtered["near_match_cedr"] += 1
                    continue
                row = {"item_id": item_id, "text": text, "source_url": data["source_url"],
                       "accessed_at": data["accessed_at"], "license": "not_specified"}
                detail = {"item_id": item_id, "source_record_id": source_id, "app_id": app_id,
                          "app_name": app_name, "title": entry["title"]["label"],
                          "rating": entry["im:rating"]["label"], "updated_at": entry["updated"]["label"],
                          "country": "ru", "snapshot_path": path.relative_to(ROOT).as_posix(),
                          "snapshot_sha256": digest, "text_sha256": sha256(text.encode("utf-8"))}
                if int(detail["rating"]) not in range(1, 6):
                    raise ValueError(f"Invalid rating: {item_id}")
                candidates[app_id].append({"row": row, "metadata": detail, "clean": clean})
                count += 1
        print(f"{app_name}: {count} candidates", flush=True)
    added = 0
    for candidate in interleave_candidates(candidates, args.seed):
        row, detail, clean = candidate["row"], candidate["metadata"], candidate["clean"]
        if row["item_id"] in known_ids:
            continue
        if len(collected) >= args.limit:
            break
        if clean in known_texts:
            filtered["exact_match_collected"] += 1
            continue
        if own_near.contains(clean):
            filtered["near_match_collected"] += 1
            continue
        collected.append(row)
        metadata[row["item_id"]] = detail
        known_ids.add(row["item_id"])
        known_texts.add(clean)
        own_near.add(row["item_id"], clean)
        added += 1
    if not collected:
        raise ValueError("No suitable reviews; existing output was not changed")
    # Preserve completed annotation cells on every rerun. Never infer emotions from stars.
    annotation_rows = []
    for row in collected:
        previous = annotations.get(row["item_id"])
        if previous and previous["text"] != row["text"]:
            raise ValueError(f"Annotation text changed: {row['item_id']}")
        annotation_rows.append(previous or {
            field: row[field] if field in {"item_id", "text"} else "" for field in ANNOTATION_FIELDS
        })
    overlap = sum(normalize(row["text"]) in reference for row in collected)
    near_overlap = sum(reference_near.contains(normalize(row["text"])) for row in collected)
    audit_near, audit_exact = NearIndex(), set()
    internal_exact, internal_near = 0, 0
    for row in collected:
        clean = normalize(row["text"])
        internal_exact += clean in audit_exact
        internal_near += audit_near.contains(clean)
        audit_exact.add(clean)
        audit_near.add(row["item_id"], clean)
    if overlap or near_overlap or internal_exact or internal_near:
        raise ValueError("Collection contains copies of CEDR-M7 texts or internal duplicates")
    if any(row["item_id"] not in metadata for row in collected):
        raise ValueError("Existing rows lack provenance in metadata.csv")
    details = [metadata[row["item_id"]] for row in collected]
    report = {
        "batch": args.batch, "pages_per_app": args.pages, "seed": args.seed, "limit": args.limit,
        "source": "Russian App Store customer-review RSS feeds", "license": "not_specified",
        "baseline": {"dataset": "Aniemore/cedr-m7", "revision": REVISION, "files": reference_records},
        "downloaded_entries": sum(item["entries"] for item in fetched),
        "unique_source_ids": len(source_ids), "eligible_before_collection_dedup": sum(map(len, candidates.values())),
        "filtered": dict(sorted(filtered.items())), "collected_rows": len(collected),
        "exact_normalized_overlap_with_cedr_train_and_test": overlap,
        "near_overlap_with_cedr_train_and_test": near_overlap,
        "internal_exact_duplicates": internal_exact,
        "internal_near_duplicates": internal_near,
        "per_app": dict(Counter(row["app_name"] for row in details)),
        "ratings_not_emotion_labels": dict(Counter(row["rating"] for row in details)),
        "source_updated_at_min": min(row["updated_at"] for row in details),
        "source_updated_at_max": max(row["updated_at"] for row in details),
        "lexical_cues_not_emotion_labels": {
            label: sum(bool(re.search(pattern, row["text"], re.I)) for row in collected)
            for label, pattern in CUES.items()
        },
        "near_duplicate_rule": "word-trigram Jaccard >= 0.90, for texts with at least 8 words",
        "snapshots": fetched,
    }
    write_csv(output / "collected.csv", COLLECTED_FIELDS, collected)
    write_csv(output / "metadata.csv", METADATA_FIELDS, details)
    write_csv(output / "annotations.csv", ANNOTATION_FIELDS, annotation_rows)
    (output / "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(collected)} reviews ({added} new); CEDR-M7 exact overlap: {overlap}")
    if len(collected) < args.limit:
        print("Fewer suitable reviews than the limit; increase --pages or keep the smaller batch.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--pages", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Use cached snapshots only")
    args = parser.parse_args()
    if args.limit < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.batch):
        parser.error("Positive --limit and a simple --batch name are required")
    collect(args)


if __name__ == "__main__":
    main()
