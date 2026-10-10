"""Обнаружение объектов на видео с помощью SSD MobileNetV2 FPNLite."""
import argparse
import math
from pathlib import Path

from .common import existing_file, write_json

MODEL = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1'


def select_detections(outputs, threshold, limit=20):
    # В ответе модели первый индекс относится к единственному кадру.
    count = int(outputs['num_detections'][0])
    detections = []
    for index in range(count):
        score = float(outputs['detection_scores'][0][index])
        if score < threshold:
            continue
        detections.append({
            'class_id': int(outputs['detection_classes'][0][index]),
            'score': score,
            'box_yxyx': [float(v) for v in outputs['detection_boxes'][0][index]],
        })
    return sorted(detections, key=lambda item: item['score'], reverse=True)[:limit]


def detect_video(path, output_video, every=5, threshold=0.5):
    path = existing_file(path)
    if every < 1:
        raise ValueError('--every должен быть положительным.')
    if not 0 < threshold <= 1:
        raise ValueError('--threshold должен быть в интервале (0, 1].')
    target = Path(output_video)
    if path.resolve() == target.resolve():
        raise ValueError('Входное видео и результат должны иметь разные пути.')
    import cv2
    import tensorflow as tf
    import tensorflow_hub as hub
    capture = cv2.VideoCapture(str(path))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError('Не удалось открыть видео.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Не удалось определить частоту кадров.')
        ok, frame = capture.read()
        if not ok:
            raise ValueError('В видео нет доступных кадров.')
        detector = hub.load(MODEL)
        target.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (width, height))
        if not writer.isOpened():
            raise ValueError('Не удалось создать выходной MP4.')
        index = 0
        records = []
        while ok:
            # Пропущенные кадры сохраняются без рамок: старые координаты не переносим.
            if index % every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Добавляем размерность: один кадр, высота, ширина, три канала.
                batch = tf.expand_dims(tf.convert_to_tensor(rgb, dtype=tf.uint8), axis=0)
                outputs = detector(batch)
                outputs = {key: value.numpy() for key, value in outputs.items()}
                detections = select_detections(outputs, threshold)
                records.append({'frame': index, 'seconds': round(index / fps, 3),
                                'detections': detections})
                for detection in detections:
                    ymin, xmin, ymax, xmax = detection['box_yxyx']
                    # Модель выдаёт координаты от 0 до 1. Переводим их в пиксели.
                    first = (int(xmin * width), int(ymin * height))
                    last = (int(xmax * width), int(ymax * height))
                    cv2.rectangle(frame, first, last, (0, 200, 0), 2)
                    label = f"COCO {detection['class_id']}: {detection['score']:.2f}"
                    cv2.putText(frame, label, first, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 200, 0), 1)
            writer.write(frame)
            index += 1
            ok, frame = capture.read()
        return {'file': str(path), 'output_video': str(target), 'model': MODEL,
                'fps': fps, 'frames': index, 'every': every,
                'threshold': threshold, 'processed_frames': records}
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--output-video', default='outputs/detected.mp4')
    parser.add_argument('--output', default='outputs/detections.json')
    parser.add_argument('--every', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    try:
        result = detect_video(args.video, args.output_video, args.every, args.threshold)
        write_json(result, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
