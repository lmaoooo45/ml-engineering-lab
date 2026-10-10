import json

import numpy as np
import pytest
import soundfile as sf

from ml_lab.audio import read_audio
from ml_lab.common import write_json
from ml_lab.evaluate_text import classification_metrics
from ml_lab.text import analyze
from ml_lab.video import select_detections


def test_audio_stereo_resampling(tmp_path):
    path = tmp_path / 'stereo.wav'
    t = np.arange(8000) / 8000
    wave = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, np.column_stack([wave, wave]), 8000, subtype='FLOAT')
    samples = read_audio(path)
    assert samples.shape == (16000,)
    assert samples.dtype == np.float32
    assert np.isfinite(samples).all()
    # После ресэмплинга тон должен остаться около 440 Гц.
    frequency = np.argmax(abs(np.fft.rfft(samples)))
    assert frequency == 440


def test_audio_long_recording_rejected(tmp_path):
    path = tmp_path / 'long.wav'
    sf.write(path, np.zeros(31 * 8000), 8000)
    with pytest.raises(ValueError, match='30 секунд'):
        read_audio(path)


@pytest.mark.parametrize('text', ['', '  ', 'а' * 4001, None])
def test_invalid_text_rejected_without_loading_model(text):
    with pytest.raises(ValueError):
        analyze(text)


def test_detection_threshold_and_count():
    output = {'num_detections': np.array([2]),
              'detection_scores': np.array([[0.8, 0.2, 0.99]]),
              'detection_classes': np.array([[1, 3, 7]]),
              'detection_boxes': np.array([[[0.1, 0.2, 0.5, 0.6]] * 3])}
    result = select_detections(output, 0.5)
    assert len(result) == 1
    assert result[0]['class_id'] == 1
    assert result[0]['box_yxyx'] == [0.1, 0.2, 0.5, 0.6]


def test_metrics_for_wrong_prediction():
    result = classification_metrics(['negative', 'neutral', 'positive'],
                                    ['positive', 'neutral', 'positive'])
    assert result['accuracy'] == pytest.approx(2 / 3)
    assert result['macro_f1'] == pytest.approx((0 + 1 + 2 / 3) / 3)
    assert result['confusion_matrix']['negative']['positive'] == 1


def test_json_output_creates_directory(tmp_path):
    path = tmp_path / 'results' / 'result.json'
    write_json({'text': 'пример'}, path)
    assert json.loads(path.read_text(encoding='utf-8')) == {'text': 'пример'}
