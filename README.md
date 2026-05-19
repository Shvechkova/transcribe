# Транскрибация и диаризация созвонов

Локальная транскрибация записей встреч (Whisper) с разделением по спикерам (pyannote.audio). Полностью оффлайн после однократной загрузки моделей.

## Что внутри

- `transcribe.py` — транскрибация webm/mp4/mp3/wav в `.txt` и `.srt` через faster-whisper (large-v3, русский).
- `diarize_and_merge.py` — диаризация (определение спикеров) через pyannote и слияние с готовым `.srt` в файлы `*.speakers.txt` и `*.speakers.srt`.

## Требования

- Windows / macOS / Linux
- Python 3.10–3.14
- ~6 ГБ свободного места под модели (3 ГБ Whisper + ~500 МБ pyannote)
- Минимум 8 ГБ RAM
- CPU: для 30-минутной записи нужно ~38 мин транскрибации + ~15 мин диаризации на обычном ноутбуке

## Установка

```powershell
# 1. Клонируем
git clone https://github.com/Shvechkova/transcribe.git
cd transcribe

# 2. (опц., но рекомендуется) виртуальное окружение
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux

# 3. Зависимости
pip install faster-whisper pyannote.audio numpy av
```

## Настройка HuggingFace (только для диаризации)

Диаризация использует gated-модели pyannote — нужен бесплатный токен и принятие лицензии.

1. Зарегистрироваться на https://huggingface.co
2. Создать read-токен: https://huggingface.co/settings/tokens → **New token** → тип **Read**
3. Нажать **Agree and access repository** на трёх страницах:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1

Если нужна только транскрибация без определения спикеров — токен не нужен.

## Запуск

### Только транскрибация

```powershell
python transcribe.py "path\to\запись.webm"
```

На выходе рядом с исходником появятся:
- `запись.txt` — голый текст
- `запись.srt` — субтитры с таймкодами
- `запись.progress.log` — прогресс по сегментам

### Транскрибация + диаризация

```powershell
$env:HF_TOKEN="hf_ваш_токен"
python transcribe.py "path\to\запись.webm"
python diarize_and_merge.py "path\to\запись.webm"
```

Дополнительно появятся:
- `запись.speakers.txt` — текст с метками `[чч:мм:сс] Спикер N`
- `запись.speakers.srt` — субтитры с префиксом `Спикер N:` в каждой реплике
- `запись.16k.wav` — промежуточное аудио (можно удалить)
- `запись.16k.turns.pkl` — кеш результата диаризации (используется при повторных запусках)

### macOS/Linux

```bash
export HF_TOKEN=hf_ваш_токен
python transcribe.py "path/to/запись.webm"
python diarize_and_merge.py "path/to/запись.webm"
```

## Параметры по умолчанию

- Модель Whisper: `large-v3`, `float32`, beam_size=5, VAD-фильтр включён
- Язык: русский (жёстко прописан в `transcribe.py`, поменять в вызове `model.transcribe(language="ru", ...)`)
- Пайплайн диаризации: `pyannote/speaker-diarization-3.1`, число спикеров определяется автоматически

## Поддерживаемые форматы аудио/видео

Любые, которые умеет PyAV / FFmpeg: webm, mp4, mkv, mov, mp3, wav, m4a, flac и др.

## Известные нюансы

- На первом запуске Whisper качает ~3 ГБ, pyannote ~500 МБ. Дальше — из локального кеша (`~/.cache/huggingface`).
- На Windows pyannote 4.x выводит шумный traceback про `torchcodec`/`libtorchcodec_core*.dll` — это **не ошибка**, скрипт обходит его, загружая wav в память напрямую.
- Диаризация может склеивать короткие реплики-подтверждения («да», «угу») с соседним длинным куском того же спикера — нормальное поведение pyannote, не баг.
