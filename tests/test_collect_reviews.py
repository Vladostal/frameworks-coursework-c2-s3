import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "collector", Path(__file__).resolve().parents[1] / "scripts/collect_reviews.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectionQualityTests(unittest.TestCase):
    def test_duplicate_normalization_ignores_layout_case_and_yo(self):
        self.assertEqual(collector.normalize("Всё  НЕ работает!!!\nОчень плохо."),
                         collector.normalize("все не работает очень плохо"))

    def test_text_for_annotation_keeps_negation_punctuation_and_emotion(self):
        text = "  Я совсем НЕ злюсь!\n\nПросто очень расстроилась 😔  "
        self.assertEqual(collector.prepare_text(text),
                         "Я совсем НЕ злюсь! Просто очень расстроилась 😔")

    def test_contact_filter_does_not_confuse_app_versions_with_phone_numbers(self):
        text = "После обновления до версии 26.44.74 приложение совсем не работает."
        self.assertIsNone(collector.rejection_reason(text))
        for contact in ["test@example.com", "+7 (999) 123-45-67", "https://example.com", "@some_user"]:
            self.assertEqual(collector.rejection_reason(f"{text} Свяжитесь со мной: {contact}"),
                             "contact_or_external_link")

    def test_near_duplicates_with_small_additions_are_removed(self):
        index = collector.NearIndex()
        original = " ".join(f"слово{i}" for i in range(40))
        index.add("reference-train-1", original)
        self.assertTrue(index.contains(original + " дополнение"))
        self.assertFalse(index.contains("другой совершенно независимый отзыв об ошибке приложения"))

    def test_short_english_and_empty_reviews_are_not_accepted(self):
        for text in ["", "Хорошо!", "This application is excellent and I absolutely love using it every day."]:
            self.assertIsNotNone(collector.rejection_reason(text))


if __name__ == "__main__":
    unittest.main()
