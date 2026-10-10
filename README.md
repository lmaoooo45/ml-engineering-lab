# ML Engineering Lab

Четыре небольших приложения с готовыми моделями: тональность отзывов, распознавание речи, классификация изображений и обнаружение объектов на видео. Обучение моделей с нуля не выполняется.

| Задача | Модель | Библиотека | Фреймворк |
|---|---|---|---|
| Текст | rubert-tiny-sentiment-balanced | Transformers | PyTorch |
| Аудио | Whisper Tiny | Transformers | PyTorch |
| Изображения | MobileNetV2 / ImageNet | Keras Applications | TensorFlow |
| Видео | SSD MobileNetV2 FPNLite 320 / COCO | TensorFlow Hub, OpenCV | TensorFlow |

## Установка

Проверенная среда: Linux, Python 3.12, CPU. GPU не требуется. Для первого запуска нужен Интернет: библиотеки и веса моделей скачиваются отдельно. Зависимости занимают несколько гигабайт.

```bash
python -m venv .venv
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export KERAS_HOME="$PWD/.cache/keras"
export TFHUB_CACHE_DIR="$PWD/.cache/tfhub"
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

Если проект находится в `/workspace/ml-engineering-lab` можно выполнить `bash scripts/install.sh`. `requirements.txt` содержит основные зависимости, `requirements-lock.txt` - точные версии всей проверенной установки, кроме отдельно устанавливаемого PyTorch.

При ограниченной сети нужны `huggingface.co` и домены выдачи весов Hugging Face (в этой проверке - `us.aws.cdn.hf.co`), `tfhub.dev`, `kaggle.com`, `www.kaggle.com`, `storage.googleapis.com`. Реестры пакетов: `pypi.org`, `files.pythonhosted.org`, `download.pytorch.org`, `download-r2.pytorch.org`. Для публичных моделей ключи доступа не нужны.

## Запуск

Все команды выполняются из корня репозитория с активированной `.venv`.

```bash
python -m ml_lab.text 'Хороший товар, покупкой доволен!' --output outputs/review.json
python -m ml_lab.evaluate_text examples/reviews.csv --output outputs/text_metrics.json
python -m ml_lab.audio examples/speech.wav --output outputs/transcript.json
python -m ml_lab.image examples/photo.jpg --output outputs/classes.json
python -m ml_lab.video examples/clip.mp4 --every 5 --threshold 0.5
```

`speech.wav`, `photo.jpg`, `clip.mp4` нужно добавить самостоятельно; эти файлы в репозиторий не включены. Для аудио запишите до 30 секунд русской речи. Для изображений подходят обычные фотографии предметов; для видео - короткая запись с людьми или транспортом.

Текстовое решение возвращает `label` и `score`. Метки: `negative`, `neutral`, `positive`. `score` - оценка уверенности, а не точность модели. Изображения возвращают top-5 классов на английском. Видео сохраняет `outputs/detected.mp4` и JSON с координатами рамок. Координаты нормированы и идут в порядке `[ymin, xmin, ymax, xmax]`, классы - ID COCO (например, 1 - человек, 3 - автомобиль). Результат видео не содержит звука.

По умолчанию анализируется каждый пятый кадр. На остальных кадрах рамки не рисуются, чтобы не показывать устаревшие координаты. Для рамок на каждом кадре задайте `--every 1`; это медленнее. Это детекция, а не отслеживание объектов: постоянных ID объектов нет.

## Проверки и документация

- [Технические задания](docs/specifications.md)
- [Отчёт: модели, датасеты, метрики и ограничения](docs/report.md)
- [Фактические результаты проверки](docs/validation.md)
- [План проверки качества](docs/evaluation.md)

`pytest` проверяет обработку данных и расчёт метрик без скачивания весов. Он не заменяет запуск четырёх моделей на реальных данных. CSV отзывов содержит 12 вручную составленных примеров для быстрой проверки. Они не заменяют проверку на большом наборе реальных отзывов.

## Будущий API

Функция `ml_lab.text.analyze(text)` работает отдельно от командного интерфейса. Её можно вызвать из обработчика `POST /sentiment`, передав строку и вернув JSON. После первой загрузки модель остаётся в памяти процесса. API в этой работе пока не реализован; для него отдельно понадобятся проверка размера запроса, обработка ошибок и управление параллельными запросами.

## Несколько пояснений к коду

Тяжёлые библиотеки импортируются внутри функций, чтобы справка команд и проверка неправильных входов не требовали загрузки моделей. `device=-1` в Transformers означает запуск на CPU. `REVISION` фиксирует версию модели на Hugging Face.

Модели принимают наборы входных данных. Поэтому перед одним фото или кадром добавляется размерность: число изображений, высота, ширина, каналы. `numpy()` переводит ответ TensorFlow в массив, с которым удобнее работать при сохранении JSON.
