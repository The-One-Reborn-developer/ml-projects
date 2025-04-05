# Gemma 3 Fine Tuning for Visual Recognition Task

## Setup

Python 3.11.7

1. ```python -m venv .venv```

2. ```source .venv/bin/activate```

3. ```pip install -r requirements.txt```

4. Скачать архив bank_cards.7z [отсюда](https://drive.google.com/drive/folders/0ABdDBYTpNH1KUk9PVA)

5. Создать папку romanian_bank_cards в корне проекта и распаковать архив в неё.

6. ```python create_image_urls.py```

7. ```python create_jsonl.py```

8. ```PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python fine_tune.py```

9. ```python merge.py```

## Inference

1. ```python inference.py path/to/file```
