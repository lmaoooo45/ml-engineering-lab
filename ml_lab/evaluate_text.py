"""Оценка тональности на CSV с колонками text,label."""
import argparse
import csv
import time

from .common import existing_file, write_json
from .text import analyze

LABELS = ('negative', 'neutral', 'positive')


def classification_metrics(expected, predicted):
    if not expected or len(expected) != len(predicted):
        raise ValueError('Нужны равные непустые списки меток.')
    if any(label not in LABELS for label in expected + predicted):
        raise ValueError('Ожидаются метки negative, neutral или positive.')
    # Строка матрицы: правильный класс. Столбец: ответ модели.
    matrix = {}
    for label in LABELS:
        matrix[label] = {}
        for other in LABELS:
            matrix[label][other] = 0
    for truth, guess in zip(expected, predicted):
        matrix[truth][guess] += 1
    scores = []
    for label in LABELS:
        tp = matrix[label][label]
        fp = 0
        fn = 0
        for other in LABELS:
            if other != label:
                fp += matrix[other][label]
                fn += matrix[label][other]
        # TP: верный ответ, FP: ложное срабатывание, FN: пропущенный класс.
        denominator = 2 * tp + fp + fn
        if denominator == 0:
            f1 = 0.0
        else:
            f1 = 2 * tp / denominator
        scores.append(f1)
    correct = 0
    for truth, guess in zip(expected, predicted):
        if truth == guess:
            correct += 1
    return {'count': len(expected),
            'accuracy': correct / len(expected),
            'macro_f1': sum(scores) / len(scores), 'confusion_matrix': matrix}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--output')
    args = parser.parse_args()
    with existing_file(args.csv).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = [row['label'] for row in rows]
    if not rows or any(label not in LABELS for label in expected):
        parser.error('CSV должен содержать тексты и корректные метки.')
    started = time.perf_counter()
    results = [analyze(row['text']) for row in rows]
    metrics = classification_metrics(expected, [item['label'] for item in results])
    metrics['seconds_including_model_load'] = time.perf_counter() - started
    metrics['examples'] = results
    write_json(metrics, args.output)


if __name__ == '__main__':
    main()
