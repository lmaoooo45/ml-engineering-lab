"""Тональность русских отзывов. Функцию analyze можно вызвать из будущего API."""
import argparse
from functools import lru_cache

from .common import write_json

MODEL = 'cointegrated/rubert-tiny-sentiment-balanced'
# Фиксируем версию модели, чтобы результат не зависел от обновлений на сайте.
REVISION = 'e0679dd4460e48648f5d4d819ae0f505384f5a9d'


# Сохраняем загруженную модель для следующих вызовов analyze.
@lru_cache(maxsize=1)
def load_model():
    from transformers import pipeline
    return pipeline('text-classification', model=MODEL, revision=REVISION, device=-1)


def analyze(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Нужен непустой текст.')
    if len(text) > 4000:
        raise ValueError('Отзыв должен содержать не больше 4000 символов.')
    classifier = load_model()
    predictions = classifier(
        text.strip(), truncation=True, function_to_apply='sigmoid'
    )
    result = predictions[0]
    return {'text': text.strip(), 'label': result['label'],
            'score': float(result['score']), 'model': MODEL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', help='Отзыв на русском языке')
    parser.add_argument('--output', help='Путь к JSON')
    args = parser.parse_args()
    try:
        write_json(analyze(args.text), args.output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
