"""Распознавание русской речи из короткого WAV/FLAC/OGG файла."""
import argparse
from math import gcd

from .common import existing_file, write_json

MODEL = 'openai/whisper-tiny'
# Версия модели на Hugging Face.
REVISION = '169d4a4341b33bc18d8881c4b69c2e104e1cc0af'
SAMPLE_RATE = 16000


def read_audio(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    samples, rate = sf.read(existing_file(path), dtype='float32', always_2d=True)
    if not len(samples):
        raise ValueError('Аудиофайл пуст.')
    if len(samples) / rate > 30:
        raise ValueError('Этот пример принимает записи длительностью до 30 секунд.')
    # Усредняем левый и правый каналы, чтобы получить моно.
    mono = samples.mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError('Аудио содержит некорректные значения.')
    if rate != SAMPLE_RATE:
        # Сокращаем отношение частот: например, 16000/48000 превращается в 1/3.
        divisor = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, rate // divisor)
    return mono.astype(np.float32)


def transcribe(path):
    from transformers import pipeline
    samples = read_audio(path)
    recognizer = pipeline(
        'automatic-speech-recognition',
        model=MODEL, revision=REVISION, device=-1,
    )
    result = recognizer({'raw': samples, 'sampling_rate': SAMPLE_RATE},
                        generate_kwargs={'language': 'russian', 'task': 'transcribe'})
    return {'text': result['text'].strip(), 'model': MODEL,
            'duration_seconds': round(len(samples) / SAMPLE_RATE, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio')
    parser.add_argument('--output')
    args = parser.parse_args()
    try:
        write_json(transcribe(args.audio), args.output)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
