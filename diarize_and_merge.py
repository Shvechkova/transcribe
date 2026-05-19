import os
import re
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import av
import numpy as np


SRT_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def parse_ts(ts: str) -> float:
    m = SRT_TS_RE.match(ts)
    h, mn, s, ms = m.groups()
    return int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def fmt_short(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds - h * 3600 - m * 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def read_srt(path: Path):
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    segs = []
    for b in blocks:
        lines = b.splitlines()
        if len(lines) < 3:
            continue
        m = re.match(r"(\S+)\s+-->\s+(\S+)", lines[1])
        if not m:
            continue
        start = parse_ts(m.group(1))
        end = parse_ts(m.group(2))
        text = " ".join(lines[2:]).strip()
        segs.append((start, end, text))
    return segs


def extract_wav(src: Path, dst: Path, sr: int = 16000) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] Extracting audio to {dst.name} ({sr} Hz mono)...", flush=True)
    container = av.open(str(src))
    stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
    samples = []
    for frame in container.decode(stream):
        for rframe in resampler.resample(frame):
            samples.append(rframe.to_ndarray().reshape(-1))
    for rframe in resampler.resample(None) or []:
        samples.append(rframe.to_ndarray().reshape(-1))
    audio = np.concatenate(samples).astype(np.int16)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    print(f"[{time.strftime('%H:%M:%S')}] Wav size: {dst.stat().st_size / 1e6:.1f} MB, duration: {len(audio) / sr / 60:.1f} min", flush=True)


def diarize(wav: Path, token: str):
    import pickle
    import torch
    from pyannote.audio import Pipeline

    cache = wav.with_suffix(".turns.pkl")
    if cache.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Reusing cached turns from {cache.name}", flush=True)
        return pickle.loads(cache.read_bytes())

    print(f"[{time.strftime('%H:%M:%S')}] Loading diarization pipeline...", flush=True)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    print(f"[{time.strftime('%H:%M:%S')}] Loading wav into memory...", flush=True)
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio).unsqueeze(0)
    print(f"[{time.strftime('%H:%M:%S')}] Running diarization (this is the slow step)...", flush=True)
    t0 = time.time()
    result = pipeline({"waveform": waveform, "sample_rate": sr})
    print(f"[{time.strftime('%H:%M:%S')}] Diarization finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    annot = getattr(result, "speaker_diarization", None) or getattr(result, "diarization", None) or result
    print(f"[{time.strftime('%H:%M:%S')}] Result type: {type(result).__name__}, annotation type: {type(annot).__name__}", flush=True)
    turns = [(turn.start, turn.end, speaker) for turn, _, speaker in annot.itertracks(yield_label=True)]
    cache.write_bytes(pickle.dumps(turns))
    print(f"[{time.strftime('%H:%M:%S')}] Cached {len(turns)} turns to {cache.name}", flush=True)
    return turns


def speaker_for(start: float, end: float, turns) -> str:
    overlaps = defaultdict(float)
    for ts, te, sp in turns:
        ov = max(0.0, min(end, te) - max(start, ts))
        if ov > 0:
            overlaps[sp] += ov
    if not overlaps:
        return "?"
    return max(overlaps.items(), key=lambda x: x[1])[0]


def renumber_speakers(turns):
    order = {}
    n = 0
    for _, _, sp in turns:
        if sp not in order:
            n += 1
            order[sp] = f"Спикер {n}"
    return order


def main(audio_path: str) -> None:
    audio = Path(audio_path)
    if not audio.exists():
        sys.exit(f"File not found: {audio}")
    srt = audio.with_suffix(".srt")
    if not srt.exists():
        sys.exit(f"SRT not found: {srt}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var not set")

    wav = audio.with_suffix(".16k.wav")
    if not wav.exists():
        extract_wav(audio, wav)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Reusing existing {wav.name}", flush=True)

    segs = read_srt(srt)
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(segs)} segments from SRT", flush=True)

    turns = diarize(wav, token)
    print(f"[{time.strftime('%H:%M:%S')}] Got {len(turns)} speaker turns", flush=True)

    label_map = renumber_speakers(turns)
    print(f"[{time.strftime('%H:%M:%S')}] Speakers detected: {len(label_map)}", flush=True)

    out_path = audio.parent / (audio.stem + ".speakers.txt")
    out_srt = audio.parent / (audio.stem + ".speakers.srt")
    prev_sp = None
    with open(out_path, "w", encoding="utf-8") as ftxt, open(out_srt, "w", encoding="utf-8") as fsrt:
        for i, (start, end, text) in enumerate(segs, 1):
            sp_raw = speaker_for(start, end, turns)
            sp = label_map.get(sp_raw, sp_raw)
            if sp != prev_sp:
                ftxt.write(f"\n[{fmt_short(start)}] {sp}:\n")
                prev_sp = sp
            ftxt.write(f"{text}\n")
            fsrt.write(f"{i}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{sp}: {text}\n\n")

    print(f"[{time.strftime('%H:%M:%S')}] DONE. Output: {out_path.name}, {out_srt.name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
