"""Классификация изображения: пять наиболее вероятных классов ImageNet."""
import argparse

from .common import existing_file, write_json


def classify(path, top=5):
    if not 1 <= top <= 1000:
        raise ValueError('Число классов должно быть от 1 до 1000.')
    path = existing_file(path)
    import numpy as np
    from PIL import Image, ImageOps
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2, decode_predictions, preprocess_input,
    )
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB').resize((224, 224))
    pixels = np.asarray(image, dtype=np.float32)
    # Модель принимает набор изображений. Добавляем размерность для одного фото.
    batch = np.expand_dims(pixels, axis=0)
    batch = preprocess_input(batch)
    model = MobileNetV2(weights='imagenet')
    probabilities = model(batch, training=False).numpy()
    classes = decode_predictions(probabilities, top=top)[0]
    predictions = []
    for class_id, label, score in classes:
        predictions.append({
            'id': class_id, 'label': label, 'score': float(score),
        })
    return {'file': str(path), 'model': 'MobileNetV2/ImageNet',
            'predictions': predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--output')
    args = parser.parse_args()
    try:
        write_json(classify(args.image, args.top), args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
