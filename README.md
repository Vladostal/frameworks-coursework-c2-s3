# Сравнительный анализ влияния ручного и искусственного обогащения данных на качество и устойчивость обучения моделей машинного обучения


## Данные

- **CEDR-M7**: 9410 русских текстов, 7 эмоций, MIT. [Hugging Face](https://huggingface.co/datasets/Aniemore/cedr-m7)
- **Potholes Detection**: 13 767 фото дорог с рамками ям, формат YOLOv8, CC BY 4.0. [Roboflow](https://universe.roboflow.com/project-ssayl/potholes-detection-d4rma)

Исходные датасеты скачивают ноутбуки в `data/raw/`. В Git хранятся готовые материалы по разметке отзывов.

## Ручной сбор

- **Тексты.** 350 отзывов из App Store на приложения, игры и доставку, по 50 на каждый из семи классов. Два аннотатора разметили одни и те же отзывы независимо. Совпали 240 из 350 ответов (68,6%), κ Коэна — 0,633. [Краткие правила](annotation/codebook_text.md) · [Полный протокол](annotation/codebook_text_full.md).
- **Фото ям.** 100 снимков, 15 собственных и 85 из набора Mendeley (CC BY 4.0). Источники и рамки для фото хранятся локально в `data/manual/images/`. Два аннотатора размечали фото независимо. [Правила разметки](annotation/codebook_images.md) · [Набор Mendeley](https://doi.org/10.17632/tp95cdvgm8.1).

### Материалы по отзывам

| Файл | Содержимое |
| --- | --- |
| [`dataset.csv`](data/manual/text/dataset.csv) | Отзывы и итоговые метки в формате `text,label` |
| [`sources.csv`](data/manual/text/sources.csv) | Ссылки, идентификаторы и даты источников |
| [`dual_annotations.csv`](data/manual/text/dual_annotations.csv) | Ответы Андрея (`andrey_label`) и Влада (`vlad_label`), итоговая метка (`final_label`) |
| [`dual_annotation_summary.json`](data/manual/text/dual_annotation_summary.json) | Распределения классов и показатели согласия |
| [`annotator.html`](data/manual/text/annotator.html) | Интерфейс для просмотра и разметки отзывов |

Откройте `annotator.html` в браузере. Клавиши `1`–`7` выбирают класс и переходят к следующему отзыву, стрелки листают отзывы, `0` снимает метку. Для независимой разметки нажмите «Начать новую разметку». Кнопка «Открыть CSV» загружает другой набор, а «Скачать CSV» выгружает размеченные отзывы. Промежуточные изменения сохраняются в браузере.

## Запуск

Нужен [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

В `.env` впишите `ROBOFLOW_API_KEY`, он нужен только для ноутбука с ямами.
Откройте ноутбук из `notebooks/` с ядром из `.venv` и выполните ячейки по порядку. Результаты уже сохранены, смотреть можно без запуска. Параметры скрипта сбора отзывов доступны через `uv run python scripts/collect_reviews.py --help`.

## Структура

```
notebooks/         загрузка данных и EDA
annotation/        правила разметки текста и изображений
data/manual/text/  результаты разметки отзывов
docs/              журнал курсовой работы
scripts/           сбор отзывов из App Store
tests/             проверки скрипта сбора
```
