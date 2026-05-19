import sys
import time
from pathlib import Path
from faster_whisper import WhisperModel


def format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def main(audio_path: str, model_size: str = "large-v3", compute_type: str = "float32") -> None:
    audio = Path(audio_path)
    if not audio.exists():
        sys.exit(f"File not found: {audio}")

    out_dir = audio.parent
    base = audio.stem
    txt_path = out_dir / f"{base}.txt"
    srt_path = out_dir / f"{base}.srt"
    log_path = out_dir / f"{base}.progress.log"

    log = open(log_path, "w", encoding="utf-8", buffering=1)

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    say(f"Loading model {model_size} ({compute_type}) on CPU...")
    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=0)
    say(f"Model loaded in {time.time() - t0:.1f}s")

    say(f"Transcribing: {audio.name}")
    t1 = time.time()
    segments, info = model.transcribe(
        str(audio),
        language="ru",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    say(f"Duration: {info.duration:.1f}s ({info.duration / 60:.1f} min), detected lang: {info.language} ({info.language_probability:.2f})")

    with open(txt_path, "w", encoding="utf-8") as ftxt, open(srt_path, "w", encoding="utf-8") as fsrt:
        for i, seg in enumerate(segments, 1):
            text = seg.text.strip()
            ftxt.write(text + "\n")
            ftxt.flush()
            fsrt.write(f"{i}\n{format_ts(seg.start)} --> {format_ts(seg.end)}\n{text}\n\n")
            fsrt.flush()
            pct = (seg.end / info.duration) * 100 if info.duration else 0
            elapsed = time.time() - t1
            say(f"[{pct:5.1f}%] {format_ts(seg.start)} - {format_ts(seg.end)} | elapsed {elapsed / 60:.1f}m | {text[:80]}")

    total = time.time() - t1
    say(f"DONE in {total / 60:.1f}m. Output: {txt_path.name}, {srt_path.name}")
    log.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
