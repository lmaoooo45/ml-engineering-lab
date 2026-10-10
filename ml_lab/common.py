import json
from pathlib import Path


def write_json(data, output=None):
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


def existing_file(value):
    path = Path(value)
    if not path.is_file():
        raise ValueError(f'Файл не найден: {path}')
    return path
