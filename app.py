from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import queue
import tempfile
import threading
import time
import urllib.parse
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, TypeVar

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from moss_tts_nano_runtime import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    NanoTTSService,
)
from text_normalization_pipeline import (
    TextNormalizationSnapshot as SharedTextNormalizationSnapshot,
    WeTextProcessingManager as SharedWeTextProcessingManager,
    prepare_tts_request_texts as shared_prepare_tts_request_texts,
)


APP_DIR = Path(__file__).resolve().parent
DEMO_METADATA_PATH = APP_DIR / "assets" / "demo.jsonl"
CUSTOM_VOICE_METADATA_PATH = Path(
    os.environ.get("CUSTOM_VOICE_METADATA_PATH", str(APP_DIR / "assets" / "custom_voices.json"))
).expanduser()
CUSTOM_VOICE_AUDIO_DIR = Path(
    os.environ.get("CUSTOM_VOICE_AUDIO_DIR", str(APP_DIR / "assets" / "audio" / "custom"))
).expanduser()
PROMPT_UPLOAD_DIR = APP_DIR / ".app_prompt_uploads"
CUSTOM_VOICE_UPLOAD_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
CUSTOM_VOICE_LANGUAGES = {"zh", "en", "ja", "ko", "other"}
CUSTOM_VOICE_LOCK = threading.Lock()


def _env_default(name: str, fallback: str | Path) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return str(fallback)
    return value


@dataclass(frozen=True)
class DemoEntry:
    demo_id: str
    name: str
    prompt_audio_path: Path
    prompt_audio_relative_path: str
    text: str
    language: str = "zh"
    is_custom: bool = False
    created_at: str = ""
    updated_at: str = ""


def _normalize_custom_voice_language(language: str | None) -> str:
    normalized = str(language or "zh").strip().lower()
    if normalized == "jp":
        normalized = "ja"
    return normalized if normalized in CUSTOM_VOICE_LANGUAGES else "zh"


def _sanitize_custom_voice_title(title: str | None) -> str:
    normalized = " ".join(str(title or "").strip().split())
    if not normalized:
        raise ValueError("title is required.")
    if len(normalized) > 80:
        normalized = normalized[:80].strip()
    return normalized


def _format_custom_voice_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _relative_to_app(path: Path) -> str:
    return str(path.resolve().relative_to(APP_DIR)).replace("\\", "/")


def _custom_voice_audio_path_to_payload(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(APP_DIR)).replace("\\", "/")
    except ValueError:
        return str(resolved_path)


def _resolve_custom_voice_payload_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(str(path_value))
    if not candidate.is_absolute():
        candidate = APP_DIR / candidate
    try:
        path = candidate.resolve()
        path.relative_to(CUSTOM_VOICE_AUDIO_DIR.resolve())
    except Exception:
        return None
    return path


def _custom_voice_record_to_entry(payload: dict[str, object]) -> DemoEntry | None:
    voice_id = str(payload.get("id", "")).strip()
    title = str(payload.get("title") or payload.get("name") or "").strip()
    relative_path = str(payload.get("audio_path") or payload.get("role") or "").strip()
    if not voice_id or not title or not relative_path:
        return None

    prompt_audio_path = _resolve_custom_voice_payload_path(relative_path)
    if prompt_audio_path is None:
        logging.warning("skip custom voice outside custom audio dir: id=%s path=%s", voice_id, relative_path)
        return None
    if not prompt_audio_path.is_file():
        logging.warning("skip custom voice with missing audio: id=%s path=%s", voice_id, prompt_audio_path)
        return None

    return DemoEntry(
        demo_id=voice_id,
        name=title,
        prompt_audio_path=prompt_audio_path,
        prompt_audio_relative_path=_custom_voice_audio_path_to_payload(prompt_audio_path),
        text=str(payload.get("text") or ""),
        language=_normalize_custom_voice_language(str(payload.get("language") or "")),
        is_custom=True,
        created_at=str(payload.get("created_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
    )


def _read_custom_voice_records_unlocked() -> list[dict[str, object]]:
    if not CUSTOM_VOICE_METADATA_PATH.is_file():
        return []
    try:
        payload = json.loads(CUSTOM_VOICE_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("failed to parse custom voice metadata: %s", CUSTOM_VOICE_METADATA_PATH, exc_info=True)
        return []
    if not isinstance(payload, list):
        logging.warning("custom voice metadata must be a JSON list: %s", CUSTOM_VOICE_METADATA_PATH)
        return []
    return [item for item in payload if isinstance(item, dict)]


def _write_custom_voice_records_unlocked(records: list[dict[str, object]]) -> None:
    CUSTOM_VOICE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CUSTOM_VOICE_METADATA_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(CUSTOM_VOICE_METADATA_PATH)


def _load_custom_voice_entries() -> list[DemoEntry]:
    with CUSTOM_VOICE_LOCK:
        records = _read_custom_voice_records_unlocked()
    entries: list[DemoEntry] = []
    for payload in records:
        entry = _custom_voice_record_to_entry(payload)
        if entry is not None:
            entries.append(entry)
    return entries


def _demo_entry_to_payload(demo_entry: DemoEntry) -> dict[str, object]:
    return {
        "id": demo_entry.demo_id,
        "name": demo_entry.name,
        "title": demo_entry.name,
        "prompt_speech": demo_entry.prompt_audio_relative_path,
        "audio_path": demo_entry.prompt_audio_relative_path,
        "text": demo_entry.text,
        "language": demo_entry.language,
        "is_custom": demo_entry.is_custom,
        "builtin": not demo_entry.is_custom,
        "created_at": demo_entry.created_at,
        "updated_at": demo_entry.updated_at,
    }


def _infer_demo_language(prompt_audio_relative_path: str, payload: dict[str, object]) -> str:
    language = str(payload.get("language") or "").strip()
    if language:
        return _normalize_custom_voice_language(language)
    file_stem = Path(prompt_audio_relative_path).stem.lower()
    if file_stem.startswith("en_"):
        return "en"
    if file_stem.startswith("jp_") or file_stem.startswith("ja_"):
        return "ja"
    if file_stem.startswith("ko_"):
        return "ko"
    return "zh"


def _load_demo_entries() -> list[DemoEntry]:
    if not DEMO_METADATA_PATH.is_file():
        logging.warning("demo metadata file not found: %s", DEMO_METADATA_PATH)
        return _load_custom_voice_entries()

    demo_entries: list[DemoEntry] = []
    for line_index, raw_line in enumerate(DEMO_METADATA_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            logging.warning("failed to parse demo metadata line=%s path=%s", line_index, DEMO_METADATA_PATH, exc_info=True)
            continue

        prompt_audio_relative_path = str(payload.get("role", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not prompt_audio_relative_path or not text:
            logging.warning("skip invalid demo metadata line=%s role/text missing", line_index)
            continue

        prompt_audio_path = (APP_DIR / prompt_audio_relative_path).resolve()
        if not prompt_audio_path.is_file():
            logging.warning(
                "skip demo metadata line=%s prompt speech missing: %s",
                line_index,
                prompt_audio_path,
            )
            continue

        try:
            prompt_audio_relative_path = str(prompt_audio_path.relative_to(APP_DIR))
        except ValueError:
            logging.warning(
                "skip demo metadata line=%s prompt speech escaped app dir: %s",
                line_index,
                prompt_audio_path,
            )
            continue

        demo_index = len(demo_entries) + 1
        name = str(payload.get("name", "")).strip() or f"Demo {demo_index}: {prompt_audio_path.stem}"
        demo_entries.append(
            DemoEntry(
                demo_id=f"demo-{demo_index}",
                name=name,
                prompt_audio_path=prompt_audio_path,
                prompt_audio_relative_path=prompt_audio_relative_path,
                text=text,
                language=_infer_demo_language(prompt_audio_relative_path, payload),
                is_custom=False,
            )
        )
    demo_entries.extend(_load_custom_voice_entries())
    return demo_entries


def _resolve_vscode_root_path(vscode_proxy_uri: Optional[str], server_port: int) -> Optional[str]:
    if not vscode_proxy_uri:
        return None
    raw = vscode_proxy_uri.strip()
    if not raw or raw == "/":
        return None

    port_str = str(server_port)
    replacements = (
        "{{port}}",
        "{port}",
        "%7B%7Bport%7D%7D",
        "%7b%7bport%7d%7d",
        "%7Bport%7D",
        "%7bport%7d",
    )
    resolved = raw
    for token in replacements:
        resolved = resolved.replace(token, port_str)

    parsed = urllib.parse.urlsplit(resolved)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
    else:
        path = resolved

    if not path.startswith("/"):
        path = "/" + path
    normalized = path.rstrip("/")
    return normalized or None


@dataclass(frozen=True)
class WarmupSnapshot:
    state: str
    progress: float
    message: str
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def failed(self) -> bool:
        return self.state == "failed"


class WarmupManager:
    def __init__(self, runtime: NanoTTSService, text_normalizer_manager: "WeTextProcessingManager | None" = None) -> None:
        self.runtime = runtime
        self.text_normalizer_manager = text_normalizer_manager
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._state = "pending"
        self._progress = 0.0
        self._message = "Waiting for startup warmup."
        self._error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target=self._run, name="nano-tts-warmup", daemon=True)
            self._thread.start()

    def snapshot(self) -> WarmupSnapshot:
        with self._lock:
            return WarmupSnapshot(
                state=self._state,
                progress=self._progress,
                message=self._message,
                error=self._error,
            )

    def ensure_ready(self) -> WarmupSnapshot:
        with self._lock:
            if not self._started:
                self._started = True
                self._thread = threading.Thread(target=self._run, name="nano-tts-warmup", daemon=True)
                self._thread.start()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join()
        return self.snapshot()

    def _set_state(
        self,
        *,
        state: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            if progress is not None:
                self._progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                self._message = message
            self._error = error

    def _run(self) -> None:
        try:
            self._set_state(state="running", progress=0.1, message="Loading Nano-TTS model.", error=None)
            self.runtime.get_model()
            self._set_state(state="running", progress=0.6, message="Running startup warmup synthesis.", error=None)
            result = self.runtime.warmup()
            _maybe_delete_file(result["audio_path"])
            if self.text_normalizer_manager is not None:
                self._set_state(
                    state="running",
                    progress=0.85,
                    message="Loading WeTextProcessing text normalization.",
                    error=None,
                )
                normalization_snapshot = self.text_normalizer_manager.ensure_ready()
                if normalization_snapshot.failed:
                    raise RuntimeError(normalization_snapshot.error or normalization_snapshot.message)
            self._set_state(
                state="ready",
                progress=1.0,
                message=(
                    f"Warmup complete. device={self.runtime.device} "
                    f"elapsed={result['elapsed_seconds']:.2f}s"
                    + (" | WeTextProcessing ready." if self.text_normalizer_manager is not None else "")
                ),
                error=None,
            )
        except Exception as exc:
            logging.exception("Nano-TTS warmup failed")
            self._set_state(state="failed", progress=1.0, message="Warmup failed.", error=str(exc))


T = TypeVar("T")


class RequestRuntimeManager:
    def __init__(self, default_runtime: NanoTTSService) -> None:
        self.default_runtime = default_runtime
        self.default_cpu_threads = max(1, int(os.cpu_count() or 1))
        self._lock = threading.Lock()
        self._cpu_execution_lock = threading.Lock()
        self._cpu_runtime: NanoTTSService | None = None

    @staticmethod
    def normalize_requested_execution_device(requested: str | None) -> str:
        normalized = str(requested or "default").strip().lower()
        if normalized not in {"default", "cpu"}:
            return "default"
        return normalized

    def is_dedicated_cpu_request(self, requested: str | None) -> bool:
        normalized = self.normalize_requested_execution_device(requested)
        return normalized == "cpu" and self.default_runtime.device.type != "cpu"

    def is_cpu_runtime_loaded(self) -> bool:
        with self._lock:
            return self._cpu_runtime is not None

    def _build_cpu_runtime_locked(self) -> NanoTTSService:
        if self._cpu_runtime is not None:
            return self._cpu_runtime
        self._cpu_runtime = NanoTTSService(
            checkpoint_path=self.default_runtime.checkpoint_path,
            audio_tokenizer_path=self.default_runtime.audio_tokenizer_path,
            device="cpu",
            dtype="float32",
            attn_implementation=self.default_runtime.attn_implementation or "auto",
            output_dir=self.default_runtime.output_dir,
            voice_presets=self.default_runtime.voice_presets,
        )
        return self._cpu_runtime

    def resolve_runtime(self, requested: str | None) -> tuple[NanoTTSService, str]:
        normalized = self.normalize_requested_execution_device(requested)
        if normalized != "cpu":
            return self.default_runtime, str(self.default_runtime.device.type)
        if self.default_runtime.device.type == "cpu":
            return self.default_runtime, "cpu"
        with self._lock:
            return self._build_cpu_runtime_locked(), "cpu"

    def _resolve_cpu_threads(self, cpu_threads: int | None) -> int:
        if cpu_threads is None:
            return self.default_cpu_threads
        try:
            normalized_threads = int(cpu_threads)
        except Exception:
            return self.default_cpu_threads
        if normalized_threads <= 0:
            return self.default_cpu_threads
        return max(1, normalized_threads)

    def call_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        callback: Callable[[NanoTTSService], T],
    ) -> tuple[T, str, int | None]:
        runtime, execution_device = self.resolve_runtime(requested_execution_device)
        if runtime.device.type != "cpu":
            return callback(runtime), execution_device, None

        resolved_cpu_threads = self._resolve_cpu_threads(cpu_threads)
        with self._cpu_execution_lock:
            previous_threads = torch.get_num_threads()
            threads_changed = previous_threads != resolved_cpu_threads
            if threads_changed:
                torch.set_num_threads(resolved_cpu_threads)
            try:
                return callback(runtime), execution_device, resolved_cpu_threads
            finally:
                if threads_changed:
                    torch.set_num_threads(previous_threads)

    def iter_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        factory: Callable[[NanoTTSService], Iterator[T]],
    ) -> Iterator[tuple[T, str, int | None]]:
        runtime, execution_device = self.resolve_runtime(requested_execution_device)
        if runtime.device.type != "cpu":
            for item in factory(runtime):
                yield item, execution_device, None
            return

        resolved_cpu_threads = self._resolve_cpu_threads(cpu_threads)
        with self._cpu_execution_lock:
            previous_threads = torch.get_num_threads()
            threads_changed = previous_threads != resolved_cpu_threads
            if threads_changed:
                torch.set_num_threads(resolved_cpu_threads)
            try:
                for item in factory(runtime):
                    yield item, execution_device, resolved_cpu_threads
            finally:
                if threads_changed:
                    torch.set_num_threads(previous_threads)


@dataclass
class StreamingJob:
    stream_id: str
    audio_queue: "queue.Queue[bytes | None]" = field(default_factory=lambda: queue.Queue(maxsize=64))
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    first_audio_at: float | None = None
    completed_at: float | None = None
    state: str = "starting"
    run_status: str = "Starting realtime synthesis..."
    error: str | None = None
    prompt_audio_path: str | None = None
    sample_rate: int = 48000
    channels: int = 2
    emitted_audio_seconds: float = 0.0
    lead_seconds: float = 0.0
    current_chunk_index: int | None = None
    text_chunks: list[str] = field(default_factory=list)
    chunk_index_base: int | None = None
    audio_chunk_ranges: list[tuple[float, float, int]] = field(default_factory=list)
    is_closed: bool = False
    final_result: dict[str, object] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _resolve_playback_chunk_index_locked(self) -> int | None:
        if not self.audio_chunk_ranges:
            return self.current_chunk_index

        playback_audio_seconds = max(0.0, float(self.emitted_audio_seconds) - float(self.lead_seconds))
        for start_seconds, end_seconds, chunk_index in self.audio_chunk_ranges:
            if playback_audio_seconds <= end_seconds + 1e-6:
                return chunk_index
        return self.audio_chunk_ranges[-1][2]

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "stream_id": self.stream_id,
                "state": self.state,
                "run_status": self.run_status,
                "error": self.error,
                "prompt_audio_path": self.prompt_audio_path,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "emitted_audio_seconds": self.emitted_audio_seconds,
                "lead_seconds": self.lead_seconds,
                "current_chunk_index": self.current_chunk_index,
                "playback_chunk_index": self._resolve_playback_chunk_index_locked(),
                "text_chunks": list(self.text_chunks),
                "first_audio_latency_seconds": (
                    None
                    if self.started_at is None or self.first_audio_at is None
                    else max(0.0, self.first_audio_at - self.started_at)
                ),
                "completed_at": self.completed_at,
                "ready": self.state == "done",
                "failed": self.state == "failed",
                "closed": self.is_closed,
            }


class StreamingJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, StreamingJob] = {}

    def create(self) -> StreamingJob:
        stream_id = f"stream-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        job = StreamingJob(stream_id=stream_id)
        with self._lock:
            self._jobs[stream_id] = job
        return job

    def get(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            return self._jobs.get(stream_id)

    def close(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            job = self._jobs.get(stream_id)
        if job is None:
            return None
        with job.lock:
            job.is_closed = True
            job.state = "closed" if job.state not in {"done", "failed"} else job.state
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass
        return job

    def delete(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            return self._jobs.pop(stream_id, None)


def _warmup_status_text(snapshot: WarmupSnapshot) -> str:
    progress_pct = int(round(snapshot.progress * 100.0))
    if snapshot.failed:
        return f"Warmup failed: {snapshot.error or snapshot.message}"
    if snapshot.ready:
        return snapshot.message
    return f"Warmup in progress ({progress_pct}%): {snapshot.message}"


def _format_run_status(result: dict[str, object]) -> str:
    waveform_numpy = np.asarray(result["waveform_numpy"])
    sample_count = int(waveform_numpy.shape[0]) if waveform_numpy.ndim >= 1 else 0
    sample_rate = int(result["sample_rate"])
    audio_seconds = sample_count / sample_rate if sample_rate > 0 else 0.0
    global_attn = str(result.get("effective_global_attn_implementation", "unknown"))
    local_attn = str(result.get("effective_local_attn_implementation", global_attn))
    attn_summary = global_attn if global_attn == local_attn else f"{global_attn}/{local_attn}"
    tts_batch_size = result.get("voice_clone_chunk_batch_size")
    codec_batch_size = result.get("voice_clone_codec_batch_size")
    batch_summary = ""
    if tts_batch_size is not None or codec_batch_size is not None:
        batch_summary = f" | tts_batch={int(tts_batch_size or 1)} | codec_batch={int(codec_batch_size or 1)}"
    execution_summary = ""
    execution_device = result.get("execution_device")
    cpu_threads = result.get("cpu_threads")
    if execution_device:
        execution_summary = f" | exec={execution_device}"
        if cpu_threads is not None:
            execution_summary += f" | cpu_threads={int(cpu_threads)}"
    prompt_audio_display_path = str(result.get("prompt_audio_display_path") or "").strip()
    prompt_audio_path = str(result.get("prompt_audio_path") or "").strip()
    speaker_summary = f"voice={result['voice']}"
    if prompt_audio_display_path:
        if prompt_audio_display_path.lower().startswith("uploaded:"):
            speaker_summary = f"prompt={prompt_audio_display_path.split(':', 1)[1].strip()}"
        else:
            speaker_summary = f"prompt={Path(prompt_audio_display_path).stem}"
    elif prompt_audio_path:
        speaker_summary = f"prompt={Path(prompt_audio_path).stem}"
    return (
        f"Done | mode={result['mode']} | {speaker_summary} | "
        f"attn={attn_summary}{batch_summary}{execution_summary} | audio={audio_seconds:.2f}s | elapsed={float(result['elapsed_seconds']):.2f}s"
    )


def _format_stream_status(snapshot: dict[str, object]) -> str:
    if bool(snapshot.get("failed")):
        return f"Stream failed: {snapshot.get('error') or snapshot.get('run_status') or 'Unknown error'}"
    if bool(snapshot.get("ready")):
        return str(snapshot.get("run_status") or "Stream complete.")
    if bool(snapshot.get("closed")):
        return "Stream closed."
    return str(snapshot.get("run_status") or "Streaming...")


def _normalize_stream_chunk_index(
    raw_chunk_index: object,
    *,
    chunk_count: int,
    current_base: int | None,
) -> tuple[int | None, int | None]:
    try:
        numeric_chunk_index = int(raw_chunk_index)
    except Exception:
        return None, current_base

    if chunk_count <= 0:
        return max(0, numeric_chunk_index), current_base

    normalized_base = current_base
    if normalized_base is None:
        if numeric_chunk_index == 0:
            normalized_base = 0
        elif numeric_chunk_index == chunk_count:
            normalized_base = 1
        elif numeric_chunk_index == 1:
            normalized_base = 1
        else:
            normalized_base = 0

    normalized_chunk_index = numeric_chunk_index - normalized_base
    if 0 <= normalized_chunk_index < chunk_count:
        return normalized_chunk_index, normalized_base
    if 0 <= numeric_chunk_index < chunk_count:
        return numeric_chunk_index, 0
    if 1 <= numeric_chunk_index <= chunk_count:
        return numeric_chunk_index - 1, 1
    return None, normalized_base


def _audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    elif audio_np.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(int(audio_int16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


def _audio_to_pcm16le_bytes(audio_array) -> bytes:
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    elif audio_np.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    return audio_int16.tobytes()


def _read_audio_file_base64(path_value: str | None) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_file():
        return ""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        logging.warning("failed to read audio file for base64 response: %s", path, exc_info=True)
        return ""


def _maybe_delete_file(path_value: str | None) -> None:
    if not path_value:
        return
    try:
        Path(path_value).unlink(missing_ok=True)
    except Exception:
        logging.warning("failed to remove temporary file: %s", path_value, exc_info=True)


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _sanitize_uploaded_prompt_filename(filename: str | None) -> str:
    base_name = Path(str(filename or "")).name.strip()
    if not base_name:
        return "prompt_speech.wav"
    return base_name


def _format_uploaded_prompt_display_name(filename: str | None) -> str:
    return f"Uploaded: {_sanitize_uploaded_prompt_filename(filename)}"


async def _persist_uploaded_prompt_audio(upload: UploadFile | None) -> tuple[str | None, str | None]:
    if upload is None:
        return None, None

    original_filename = _sanitize_uploaded_prompt_filename(upload.filename)
    suffix = Path(original_filename).suffix
    if not suffix or len(suffix) > 16:
        suffix = ".wav"

    PROMPT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    bytes_written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix="prompt-speech-",
            suffix=suffix,
            dir=str(PROMPT_UPLOAD_DIR),
        ) as handle:
            temp_path = handle.name
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    finally:
        await upload.close()

    if not temp_path or bytes_written <= 0:
        _maybe_delete_file(temp_path)
        raise ValueError("Uploaded prompt speech is empty.")

    return temp_path, _format_uploaded_prompt_display_name(original_filename)


def _resolve_custom_voice_audio_path(relative_path: str | None) -> Path | None:
    return _resolve_custom_voice_payload_path(relative_path)


async def _persist_custom_voice_audio(upload: UploadFile, voice_id: str) -> str:
    original_filename = _sanitize_uploaded_prompt_filename(upload.filename)
    suffix = Path(original_filename).suffix.lower()
    if suffix not in CUSTOM_VOICE_UPLOAD_SUFFIXES:
        suffix = ".wav"

    CUSTOM_VOICE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    target_path = (CUSTOM_VOICE_AUDIO_DIR / f"{voice_id}{suffix}").resolve()
    temp_path: str | None = None
    bytes_written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix=f"{voice_id}-",
            suffix=suffix,
            dir=str(CUSTOM_VOICE_AUDIO_DIR),
        ) as handle:
            temp_path = handle.name
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    finally:
        await upload.close()

    if not temp_path or bytes_written <= 0:
        _maybe_delete_file(temp_path)
        raise ValueError("custom voice audio is empty.")

    Path(temp_path).replace(target_path)
    return _custom_voice_audio_path_to_payload(target_path)


def _render_index_html(
    *,
    request: Request,
    runtime: NanoTTSService,
    demo_entries: list[DemoEntry],
    warmup_status: str,
    text_normalization_status: str,
) -> str:
    base_path = request.scope.get("root_path", "").rstrip("/")
    template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MOSS-TTS-Nano Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef1f7;
      --bg-soft: #f7f8fc;
      --panel: #ffffff;
      --text: #1f2534;
      --muted: #647089;
      --line: #dbe2f0;
      --line-strong: #cfd8ea;
      --chip: #dfe5ff;
      --chip-text: #4f63d8;
      --accent: #6a6ef6;
      --accent-strong: #565cea;
      --accent-soft: rgba(106, 110, 246, 0.12);
      --danger: #ba1f46;
      --shadow: 0 14px 32px rgba(34, 47, 78, 0.06);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(1100px 380px at 12% -10%, #ffffff 0%, transparent 70%),
        linear-gradient(180deg, var(--bg-soft) 0%, var(--bg) 54%);
      color: var(--text);
      font-family: "Plus Jakarta Sans", "Noto Sans SC", "Segoe UI", sans-serif;
    }
    .page {
      max-width: 1460px;
      margin: 0 auto;
      padding: 24px 26px 30px;
    }
    .hero {
      margin-bottom: 12px;
    }
    .hero-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
    }
    .hero h1 {
      margin: 0 0 10px;
      font-size: 38px;
      letter-spacing: 0;
    }
    .language-switch {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.86);
      box-shadow: 0 6px 14px rgba(34, 47, 78, 0.05);
      flex: 0 0 auto;
    }
    .language-option {
      border: 0;
      border-radius: 6px;
      padding: 7px 10px;
      min-width: 44px;
      background: transparent;
      color: #52617f;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
    }
    .language-option.active {
      background: var(--accent);
      color: #ffffff;
    }
    .hero .lead {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 19px;
      line-height: 1.55;
      max-width: 980px;
    }
    .hero-points {
      margin: 0 0 8px 20px;
      padding: 0;
      color: #33415f;
      line-height: 1.6;
    }
    .hero-points strong {
      color: #121826;
    }
    .build-note {
      margin: 0;
      color: #4f5d7c;
    }
    .top-tabs {
      display: flex;
      align-items: center;
      gap: 22px;
      margin-top: 14px;
      border-bottom: 1px solid var(--line);
    }
    .top-tab {
      border: 0;
      background: transparent;
      color: #4e5f89;
      font-size: 15px;
      font-weight: 500;
      padding: 10px 0;
      position: relative;
      cursor: default;
    }
    .top-tab.active {
      color: var(--accent-strong);
      font-weight: 700;
    }
    .top-tab.active::after {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: -1px;
      height: 2px;
      background: var(--accent-strong);
    }
    .top-tab:disabled {
      opacity: 1;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(320px, 0.92fr);
      gap: 14px;
      margin-top: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      box-shadow: var(--shadow);
    }
    .field {
      margin-bottom: 11px;
    }
    .field > label[for],
    .field > .field-tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 7px;
      font-size: 14px;
      line-height: 1;
      font-weight: 700;
      color: var(--chip-text);
      background: var(--chip);
      border-radius: 6px;
      padding: 5px 8px;
    }
    .field > label[for]::before,
    .field > .field-tag::before {
      content: "♫";
      font-size: 11px;
      opacity: 0.75;
    }
    .field > label:not([for]):not(.field-tag) {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 2px;
      color: #273554;
      font-size: 14px;
      font-weight: 500;
    }
    .field input,
    .field textarea,
    .field select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 14px;
      color: var(--text);
      background: #fff;
      transition: border-color 140ms ease, box-shadow 140ms ease;
    }
    .field input:focus,
    .field textarea:focus,
    .field select:focus {
      outline: 0;
      border-color: #aeb9f7;
      box-shadow: 0 0 0 3px rgba(106, 110, 246, 0.15);
    }
    .field textarea {
      min-height: 108px;
      resize: vertical;
    }
    #normalized-text-output {
      min-height: 114px;
      background: #fbfcff;
    }
    input[type="file"] {
      border-style: dashed;
      border-color: var(--line-strong);
      padding: 54px 12px;
      background: linear-gradient(180deg, #ffffff 0%, #f5f8ff 100%);
      color: #4d5d83;
    }
    input[type="file"]::file-selector-button {
      border: 0;
      border-radius: 999px;
      padding: 8px 12px;
      margin-right: 10px;
      background: #edf1ff;
      color: var(--chip-text);
      font-weight: 700;
      cursor: pointer;
    }
    .prompt-audio-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, #ffffff 0%, #f7f9ff 100%);
      padding: 10px;
    }
    #prompt-audio-preview {
      margin-top: 0;
    }
    #prompt-audio-upload[hidden],
    #prompt-audio-preview[hidden] {
      display: none;
    }
    .prompt-audio-actions {
      display: flex;
      gap: 8px;
      margin-top: 8px;
      flex-wrap: wrap;
    }
    .prompt-audio-actions button {
      min-height: 34px;
      font-size: 13px;
      padding: 8px 12px;
    }
    .voice-manager {
      margin-top: 12px;
      margin-bottom: 12px;
    }
    .voice-manager-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .voice-manager-actions button {
      min-height: 34px;
      font-size: 13px;
      padding: 8px 12px;
    }
    .voice-manager-preview {
      margin-top: 8px;
    }
    .voice-manager-status {
      min-height: 20px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      margin: 12px 0;
      background: #f9fafe;
    }
    summary {
      cursor: pointer;
      font-weight: 600;
      color: #2f3f65;
    }
    .buttons {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 10px;
      margin-top: 12px;
    }
    button:not(.top-tab):not(.language-option) {
      border: 0;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      background: linear-gradient(90deg, #6469f6 0%, #636ef8 50%, #5f61f0 100%);
      color: #fff;
      transition: transform 140ms ease, box-shadow 140ms ease, opacity 140ms ease;
    }
    button:not(.top-tab):not(.language-option):hover {
      transform: translateY(-1px);
      box-shadow: 0 8px 16px rgba(97, 101, 242, 0.24);
    }
    #generate-btn {
      width: 100%;
      min-height: 42px;
    }
    button.secondary {
      background: #edf1fb;
      color: #44527a;
      border: 1px solid var(--line-strong);
    }
    button:not(.top-tab):not(.language-option):disabled {
      opacity: 0.6;
      cursor: wait;
      transform: none;
      box-shadow: none;
    }
    .status {
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 14px;
      color: var(--muted);
      min-height: 52px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fcfdff;
      padding: 10px 12px;
    }
    .status.error {
      color: var(--danger);
      border-color: rgba(186, 31, 70, 0.28);
      background: rgba(186, 31, 70, 0.06);
    }
    .meta {
      font-size: 13px;
      color: var(--muted);
      margin-top: 7px;
      line-height: 1.5;
    }
    .playback-script {
      min-height: 108px;
      display: flex;
      flex-wrap: wrap;
      align-content: flex-start;
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, #ffffff 0%, #f6f8fe 100%);
      overflow: auto;
    }
    .playback-script.empty {
      display: block;
      color: var(--muted);
    }
    .playback-segment {
      display: inline-flex;
      align-items: center;
      padding: 8px 11px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: #4a5c82;
      line-height: 1.6;
      transition: background-color 160ms ease, color 160ms ease, border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }
    .playback-segment.played {
      border-color: rgba(106, 110, 246, 0.24);
      background: var(--accent-soft);
      color: #424db6;
    }
    .playback-segment.active {
      border-color: var(--accent-strong);
      background: var(--accent);
      color: #ffffff;
      box-shadow: 0 8px 18px rgba(106, 110, 246, 0.32);
      transform: translateY(-1px);
    }
    audio {
      width: 100%;
      margin-top: 10px;
      border-radius: 8px;
      background: #f8f9fe;
    }
    code {
      padding: 2px 6px;
      border-radius: 4px;
      background: #eef1fb;
      color: #33426a;
    }
    a {
      color: #4a5be0;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    @media (max-width: 1120px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .buttons {
        grid-template-columns: 1fr;
      }
      .hero h1 {
        font-size: 32px;
      }
    }
    @media (max-width: 860px) {
      .row {
        grid-template-columns: 1fr;
      }
      .page {
        padding: 16px 14px 20px;
      }
      .hero .lead {
        font-size: 16px;
      }
      .hero h1 {
        font-size: 28px;
      }
      .hero-header {
        align-items: flex-start;
      }
      .language-switch {
        margin-top: 2px;
      }
      .top-tabs {
        gap: 14px;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div class="hero-header">
        <h1>MOSS-TTS-Nano Demo</h1>
        <div class="language-switch" role="group" aria-label="Language">
          <button id="language-zh" class="language-option" type="button" data-lang="zh">中文</button>
          <button id="language-en" class="language-option" type="button" data-lang="en">EN</button>
        </div>
      </div>
      <p class="lead">State-of-the-art text-to-speech demo for multilingual voice cloning.</p>
      <ul class="hero-points">
        <li><strong>Voice Clone</strong> - Clone any voice from a reference audio.</li>
        <li><strong>Voice Presets</strong> - Choose built-in demos from <code>assets/demo.jsonl</code>.</li>
      </ul>
      <p class="build-note">Built with <a href="https://github.com/OpenMOSS/MOSS-TTS-Nano" target="_blank" rel="noopener noreferrer">MOSS-TTS-Nano</a>.</p>
      <div class="top-tabs" role="tablist" aria-label="Demo mode">
        <button class="top-tab active" type="button" aria-selected="true">Voice Clone</button>
      </div>
    </div>

    <div class="grid">
      <div class="panel input-panel">
        <div class="field">
          <label for="demo">Demo</label>
          <select id="demo"></select>
        </div>

        <div class="field">
          <label for="prompt-audio-upload">Prompt Speech</label>
          <div class="prompt-audio-box">
            <input id="prompt-audio-upload" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.opus,.aac">
            <audio id="prompt-audio-preview" controls hidden></audio>
            <div id="prompt-audio-source" class="meta">Using the selected demo prompt speech.</div>
            <div class="prompt-audio-actions">
              <button id="choose-prompt-audio-btn" class="secondary" type="button" hidden>选择文件</button>
              <button id="clear-prompt-audio-btn" class="secondary" type="button" hidden>使用 Demo 音频</button>
            </div>
          </div>
        </div>

        <details id="voice-manager" class="voice-manager">
          <summary>Voice Library</summary>
          <div class="field" style="margin-top: 12px;">
            <label for="voice-manager-select">Voice Entry</label>
            <select id="voice-manager-select"></select>
            <div id="voice-manager-selected-info" class="meta"></div>
          </div>
          <audio id="voice-manager-preview" class="voice-manager-preview" controls hidden></audio>
          <div class="row">
            <div class="field">
              <label for="custom-voice-title">Display Title</label>
              <input id="custom-voice-title" type="text" maxlength="80" placeholder="Name shown in the voice selector">
            </div>
            <div class="field">
              <label for="custom-voice-language">Language</label>
              <select id="custom-voice-language">
                <option value="zh">中文</option>
                <option value="en">English</option>
                <option value="ja">日本語</option>
                <option value="ko">한국어</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label for="custom-voice-audio">Voice Recording</label>
            <input id="custom-voice-audio" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.opus,.aac">
          </div>
          <div class="field">
            <label for="custom-voice-text">Default Text (optional)</label>
            <textarea id="custom-voice-text" placeholder="Optional text to fill when this voice is selected."></textarea>
          </div>
          <div class="voice-manager-actions">
            <button id="create-voice-btn" type="button">Add Voice</button>
            <button id="update-voice-btn" class="secondary" type="button">Update Voice</button>
            <button id="delete-voice-btn" class="secondary" type="button">Delete Voice</button>
            <button id="refresh-voices-btn" class="secondary" type="button">Refresh Voices</button>
          </div>
          <div id="voice-manager-status" class="meta voice-manager-status"></div>
        </details>

        <div class="field">
          <label for="text">Text</label>
          <textarea id="text" placeholder="Enter the text you want to synthesize..."></textarea>
        </div>

        <details id="generation-options">
          <summary>Generation Options</summary>
          <div class="row" style="margin-top: 12px;">
            <div class="field">
              <label for="max-new-frames">Max New Frames</label>
              <input id="max-new-frames" type="number" min="64" max="1024" step="1" value="375">
            </div>
            <div class="field">
              <label for="voice-clone-max-text-tokens">Voice Clone Max Text Tokens</label>
              <input id="voice-clone-max-text-tokens" type="number" min="25" max="200" step="1" value="75">
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label for="tts-max-batch-size">Max TTS Batch Size (0=auto)</label>
              <input id="tts-max-batch-size" type="number" min="0" step="1" value="1">
            </div>
            <div class="field">
              <label for="codec-max-batch-size">Max Codec Batch Size (0=auto)</label>
              <input id="codec-max-batch-size" type="number" min="0" step="1" value="0">
            </div>
          </div>
          <div id="batch-note" class="meta">
            0 keeps the current default behavior. Set Max TTS Batch Size to 1 to force split chunks to run one by one.
            Buffered generation keeps chunk order and decodes codec sub-batches no larger than the current TTS batch.
            Realtime Streaming Decode keeps output order and uses the smallest active chunk-group width among auto batching, Max TTS Batch Size, and Max Codec Batch Size.
          </div>
          <div class="field">
            <label for="cpu-thread-count">CPU Threads</label>
            <input id="cpu-thread-count" type="number" min="1" step="1" value="4">
          </div>
          <div id="cpu-note" class="meta">
            This app is CPU-only. CPU Threads maps to torch.set_num_threads for that request.
          </div>
          <div class="row">
            <div class="field">
              <label for="attn-implementation">Attention Backend</label>
              <select id="attn-implementation">
                <option value="model_default">model_default</option>
                <option value="sdpa">sdpa</option>
                <option value="eager">eager</option>
              </select>
            </div>
            <div class="field">
              <label for="seed">Seed</label>
              <input id="seed" type="number" step="1" value="0">
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label for="text-temperature">Text Temperature</label>
              <input id="text-temperature" type="number" min="0.1" max="2.0" step="0.05" value="1.0">
            </div>
            <div class="field">
              <label for="text-top-p">Text Top P</label>
              <input id="text-top-p" type="number" min="0.1" max="1.0" step="0.05" value="1.0">
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label for="text-top-k">Text Top K</label>
              <input id="text-top-k" type="number" min="1" max="100" step="1" value="50">
            </div>
            <div class="field">
              <label for="audio-temperature">Audio Temperature</label>
              <input id="audio-temperature" type="number" min="0.1" max="2.0" step="0.05" value="0.8">
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label for="audio-top-p">Audio Top P</label>
              <input id="audio-top-p" type="number" min="0.1" max="1.0" step="0.05" value="0.95">
            </div>
            <div class="field">
              <label for="audio-top-k">Audio Top K</label>
              <input id="audio-top-k" type="number" min="1" max="100" step="1" value="25">
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label for="audio-repetition-penalty">Audio Repetition Penalty</label>
              <input id="audio-repetition-penalty" type="number" min="1.0" max="2.0" step="0.05" value="1.2">
            </div>
            <div class="field"></div>
          </div>
          <div class="field">
            <label><input id="do-sample" type="checkbox" checked> Do Sample</label>
          </div>
          <div class="field">
            <label><input id="enable-text-normalization" type="checkbox" checked> Enable WeTextProcessing</label>
          </div>
          <div class="field">
            <label><input id="enable-robust-text-normalization" type="checkbox" checked> Enable normalize_tts_text</label>
          </div>
          <div id="normalization-note" class="meta">
            WeTextProcessing and normalize_tts_text can now be toggled independently for each request.
            WeTextProcessing is preloaded during startup so enabling it does not add first-request graph-build latency.
          </div>
          <div class="row">
            <div class="field">
              <label><input id="realtime-stream" type="checkbox" checked> Realtime Streaming Decode</label>
            </div>
            <div class="field">
              <label for="initial-playback-delay-seconds">Initial Playback Delay (s)</label>
              <input id="initial-playback-delay-seconds" type="number" min="0.00" step="0.01" value="0.08">
            </div>
          </div>
        </details>

        <div class="buttons">
          <button id="generate-btn" type="button">Generate</button>
          <button id="pause-btn" class="secondary" type="button" disabled>Pause Playback</button>
          <button id="refresh-btn" class="secondary" type="button">Refresh Warmup Status</button>
        </div>
      </div>

      <div class="panel output-panel">
        <div class="field">
          <label class="field-tag">Warmup Status</label>
          <div id="warmup-status" class="status">__WARMUP_STATUS__</div>
        </div>
        <div class="field">
          <label class="field-tag">Text Normalization Status</label>
          <div id="text-normalization-status" class="status">__TEXT_NORMALIZATION_STATUS__</div>
        </div>
        <div class="field">
          <label class="field-tag">Run Status</label>
          <div id="run-status" class="status">Idle.</div>
        </div>
        <div id="stream-metrics" class="meta"></div>
        <div class="field">
          <label class="field-tag">Normalized Text</label>
          <textarea id="normalized-text-output" readonly style="min-height: 120px;"></textarea>
        </div>
        <div class="field">
          <label class="field-tag">Playback Script</label>
          <div id="playback-script" class="playback-script empty">The current sentence will be highlighted here during playback.</div>
        </div>
        <div class="field">
          <label class="field-tag">Generated Speech</label>
          <div id="resolved-prompt" class="meta"></div>
        </div>
        <audio id="audio-output" controls></audio>
        <div class="meta"><span id="checkpoint-label">Checkpoint</span>: __CHECKPOINT__</div>
        <div class="meta"><span id="audio-tokenizer-label">Audio Tokenizer</span>: __AUDIO_TOKENIZER__</div>
      </div>
    </div>
  </div>

  <script>
    const APP_BASE = __APP_BASE__;
    let DEMOS = __DEMOS__;
    const DEFAULT_DEMO_ID = __DEFAULT_DEMO_ID__;
    const DEFAULT_ATTN_IMPLEMENTATION = __DEFAULT_ATTN_IMPLEMENTATION__;
    const DEFAULT_CPU_THREADS = __DEFAULT_CPU_THREADS__;

    const demoSelect = document.getElementById("demo");
    const promptAudioUploadInput = document.getElementById("prompt-audio-upload");
    const promptAudioPreview = document.getElementById("prompt-audio-preview");
    const promptAudioSource = document.getElementById("prompt-audio-source");
    const choosePromptAudioBtn = document.getElementById("choose-prompt-audio-btn");
    const clearPromptAudioBtn = document.getElementById("clear-prompt-audio-btn");
    const voiceManagerSelect = document.getElementById("voice-manager-select");
    const voiceManagerSelectedInfo = document.getElementById("voice-manager-selected-info");
    const voiceManagerPreview = document.getElementById("voice-manager-preview");
    const customVoiceTitleInput = document.getElementById("custom-voice-title");
    const customVoiceLanguageSelect = document.getElementById("custom-voice-language");
    const customVoiceAudioInput = document.getElementById("custom-voice-audio");
    const customVoiceTextInput = document.getElementById("custom-voice-text");
    const createVoiceBtn = document.getElementById("create-voice-btn");
    const updateVoiceBtn = document.getElementById("update-voice-btn");
    const deleteVoiceBtn = document.getElementById("delete-voice-btn");
    const refreshVoicesBtn = document.getElementById("refresh-voices-btn");
    const voiceManagerStatus = document.getElementById("voice-manager-status");
    const warmupStatus = document.getElementById("warmup-status");
    const textNormalizationStatus = document.getElementById("text-normalization-status");
    const runStatus = document.getElementById("run-status");
    const streamMetrics = document.getElementById("stream-metrics");
    const textInput = document.getElementById("text");
    const normalizedTextOutput = document.getElementById("normalized-text-output");
    const playbackScript = document.getElementById("playback-script");
    const resolvedPrompt = document.getElementById("resolved-prompt");
    const audioOutput = document.getElementById("audio-output");
    const generateBtn = document.getElementById("generate-btn");
    const pauseBtn = document.getElementById("pause-btn");
    const refreshBtn = document.getElementById("refresh-btn");
    const realtimeStreamToggle = document.getElementById("realtime-stream");
    const initialPlaybackDelayInput = document.getElementById("initial-playback-delay-seconds");

    let currentAudioObjectUrl = null;
    let currentStreamId = null;
    let currentStreamAbortController = null;
    let currentStreamStatusTimer = null;
    let currentAudioContext = null;
    let nextPlaybackTime = 0;
    let currentInitialPlaybackDelaySeconds = 0.08;
    let currentRealtimePlaybackPaused = false;
    let currentRealtimePlaybackCompletionTimer = null;
    let playbackChunks = [];
    let bufferedPlaybackBoundaries = [];
    let currentPlaybackChunkIndex = null;
    let currentPlaybackMarkedComplete = false;
    let currentPromptAudioPreviewUrl = null;
    let currentRealtimePlaybackStartAt = null;
    let currentRealtimePlaybackScheduledAudioSeconds = 0;
    let currentRealtimePlaybackChunkRanges = [];

    const demosById = new Map();
    document.getElementById("attn-implementation").value = DEFAULT_ATTN_IMPLEMENTATION;

    const languageButtons = Array.from(document.querySelectorAll(".language-option"));
    const IS_ONNX_DEMO = document.title.includes("ONNX");
    const UI_TEXT = {
      en: {
        htmlLang: "en",
        documentTitle: "MOSS-TTS-Nano Demo",
        languageAria: "Language",
        demoModeAria: "Demo mode",
        title: "MOSS-TTS-Nano Demo",
        lead: "State-of-the-art text-to-speech demo for multilingual voice cloning.",
        pointVoiceClone: "<strong>Voice Clone</strong> - Clone any voice from a reference audio.",
        pointPresets: "<strong>Voice Presets</strong> - Choose built-in demos from <code>assets/demo.jsonl</code>.",
        buildNote: 'Built with <a href="https://github.com/OpenMOSS/MOSS-TTS-Nano" target="_blank" rel="noopener noreferrer">MOSS-TTS-Nano</a>.',
        topTab: "Voice Clone",
        labelDemo: "Voice / Demo",
        labelPromptSpeech: "Prompt Speech",
        labelText: "Text",
        placeholderText: "Enter the text you want to synthesize...",
        generationOptions: "Generation Options",
        maxNewFrames: "Max New Frames",
        voiceCloneMaxTextTokens: "Voice Clone Max Text Tokens",
        ttsMaxBatchSize: "Max TTS Batch Size (0=auto)",
        codecMaxBatchSize: "Max Codec Batch Size (0=auto)",
        batchNote: "0 keeps the current default behavior. Set Max TTS Batch Size to 1 to force split chunks to run one by one. Buffered generation keeps chunk order and decodes codec sub-batches no larger than the current TTS batch. Realtime Streaming Decode keeps output order and uses the smallest active chunk-group width among auto batching, Max TTS Batch Size, and Max Codec Batch Size.",
        cpuThreads: "CPU Threads",
        cpuNote: "This app is CPU-only. CPU Threads maps to torch.set_num_threads for that request.",
        attnImplementation: "Attention Backend",
        samplingMode: "Sampling Mode",
        seed: "Seed",
        textTemperature: "Text Temperature",
        textTopP: "Text Top P",
        textTopK: "Text Top K",
        audioTemperature: "Audio Temperature",
        audioTopP: "Audio Top P",
        audioTopK: "Audio Top K",
        audioRepetitionPenalty: "Audio Repetition Penalty",
        doSample: "Do Sample",
        doSampleDerived: "Do Sample (derived from Sampling Mode)",
        enableWetext: "Enable WeTextProcessing",
        enableRobust: "Enable normalize_tts_text",
        normalizationNote: "WeTextProcessing and normalize_tts_text can now be toggled independently for each request. WeTextProcessing is preloaded during startup so enabling it does not add first-request graph-build latency.",
        realtimeDecode: "Realtime Streaming Decode",
        initialPlaybackDelay: "Initial Playback Delay (s)",
        generate: "Generate",
        pausePlayback: "Pause Playback",
        resumePlayback: "Resume Playback",
        refreshWarmup: "Refresh Warmup Status",
        warmupStatus: "Warmup Status",
        textNormalizationStatus: "Text Normalization Status",
        runStatus: "Run Status",
        normalizedText: "Normalized Text",
        playbackScript: "Playback Script",
        generatedSpeech: "Generated Speech",
        checkpoint: "Checkpoint",
        audioTokenizer: "Audio Tokenizer",
        promptSelectedDemo: "Using the selected demo prompt speech.",
        chooseFile: "Choose File",
        resetDemoAudio: "Use Demo Audio",
        voiceLibrary: "Voice Library",
        voiceEntry: "Voice Entry",
        displayTitle: "Display Title",
        displayTitlePlaceholder: "Name shown in the voice selector",
        language: "Language",
        voiceRecording: "Voice Recording",
        defaultTextOptional: "Default Text (optional)",
        defaultTextPlaceholder: "Optional text to fill when this voice is selected.",
        addVoice: "Add Voice",
        updateVoice: "Update Voice",
        deleteVoice: "Delete Voice",
        refreshVoices: "Refresh Voices",
        voiceKindBuiltin: "Built-in",
        voiceKindCustom: "Custom",
        noVoices: "No voices available.",
        builtinVoiceInfo: "Built-in voice. It can be selected for generation, but cannot be modified or deleted.",
        customVoiceInfo: "Custom voice. You can update its title, language, default text, or replace its recording.",
        voiceTitleRequired: "Please enter a display title.",
        voiceAudioRequired: "Please choose a voice recording.",
        voiceSelectRequired: "Please select a voice entry.",
        voiceCreated: "Custom voice added.",
        voiceUpdated: "Custom voice updated.",
        voiceDeleted: "Custom voice deleted.",
        voiceRefreshed: "Voice list refreshed.",
        voiceDeleteConfirm: "Delete this custom voice? This cannot be undone.",
        builtinModifyBlocked: "Built-in voices cannot be modified or deleted.",
        filePicker: "Choose file | No file selected",
        usingUploadedPrompt: "Using uploaded prompt speech: {name}",
        usingCustomPrompt: "Using custom voice: {name}",
        usingDemoPromptPath: "Using demo prompt speech: {path}",
        usingDemoPrompt: "Using demo prompt speech.",
        playbackPlaceholder: "The current sentence will be highlighted here during playback.",
        statusIdle: "Idle.",
        statusUnknown: "Unknown status.",
        statusGeneratedReady: "Generated speech is ready.",
        statusDone: "Done.",
        statusGeneratingRealtime: "Generating realtime speech...",
        statusStreamingRealtime: "Streaming realtime audio...",
        statusStreamComplete: "Stream complete.",
        statusStartingRealtime: "Starting realtime stream...",
        statusRunningSynthesis: "Running synthesis...",
        browserNoStreaming: "This browser does not support Web Audio streaming playback.",
        readableStreamMissing: "ReadableStream is not available on this response.",
        finalResultNotReady: "Streaming finished but the final result is not ready yet.",
        metricState: "state",
        metricEmitted: "emitted",
        metricLead: "lead",
        metricPlaybackDelay: "playback_delay",
        metricFirstAudio: "first_audio",
      },
      zh: {
        htmlLang: "zh-CN",
        documentTitle: "MOSS-TTS-Nano 演示",
        languageAria: "语言",
        demoModeAria: "演示模式",
        title: "MOSS-TTS-Nano 演示",
        lead: "面向多语言音色克隆的文字转语音演示。",
        pointVoiceClone: "<strong>音色克隆</strong> - 使用参考音频克隆任意声音。",
        pointPresets: "<strong>内置示例</strong> - 从 <code>assets/demo.jsonl</code> 选择预设文本和参考音频。",
        buildNote: '基于 <a href="https://github.com/OpenMOSS/MOSS-TTS-Nano" target="_blank" rel="noopener noreferrer">MOSS-TTS-Nano</a> 构建。',
        topTab: "音色克隆",
        labelDemo: "音色 / 示例",
        labelPromptSpeech: "参考音频",
        labelText: "合成文本",
        placeholderText: "请输入要合成的文本...",
        generationOptions: "生成参数",
        maxNewFrames: "最大生成帧数",
        voiceCloneMaxTextTokens: "音色克隆文本 Token 上限",
        ttsMaxBatchSize: "最大 TTS 批大小 (0=自动)",
        codecMaxBatchSize: "最大 Codec 批大小 (0=自动)",
        batchNote: "0 表示保持当前默认行为。将最大 TTS 批大小设为 1，可强制拆分后的文本逐段生成。普通生成会保持文本块顺序，并让 codec 子批大小不超过当前 TTS 批大小。实时流式解码会保持输出顺序，并在自动批处理、最大 TTS 批大小、最大 Codec 批大小之间选择最小的有效块组宽度。",
        cpuThreads: "CPU 线程数",
        cpuNote: "当前 Web 服务为 CPU 模式。CPU 线程数会在本次请求中映射到 torch.set_num_threads。",
        attnImplementation: "注意力后端",
        samplingMode: "采样模式",
        seed: "随机种子",
        textTemperature: "文本温度",
        textTopP: "文本 Top P",
        textTopK: "文本 Top K",
        audioTemperature: "音频温度",
        audioTopP: "音频 Top P",
        audioTopK: "音频 Top K",
        audioRepetitionPenalty: "音频重复惩罚",
        doSample: "启用采样",
        doSampleDerived: "启用采样 (由采样模式决定)",
        enableWetext: "启用 WeTextProcessing",
        enableRobust: "启用 normalize_tts_text",
        normalizationNote: "WeTextProcessing 和 normalize_tts_text 可以按请求分别开关。WeTextProcessing 会在启动时预加载，启用后不会额外增加首次请求的建图等待。",
        realtimeDecode: "实时流式解码",
        initialPlaybackDelay: "初始播放延迟 (秒)",
        generate: "生成",
        pausePlayback: "暂停播放",
        resumePlayback: "继续播放",
        refreshWarmup: "刷新预热状态",
        warmupStatus: "预热状态",
        textNormalizationStatus: "文本归一化状态",
        runStatus: "运行状态",
        normalizedText: "归一化文本",
        playbackScript: "播放文本",
        generatedSpeech: "生成音频",
        checkpoint: "模型",
        audioTokenizer: "音频分词器",
        promptSelectedDemo: "正在使用所选示例的参考音频。",
        chooseFile: "选择文件",
        resetDemoAudio: "使用 Demo 音频",
        voiceLibrary: "音色库管理",
        voiceEntry: "音色条目",
        displayTitle: "显示标题",
        displayTitlePlaceholder: "显示在音色下拉框中的名称",
        language: "语言",
        voiceRecording: "音色录音",
        defaultTextOptional: "默认文本 (可选)",
        defaultTextPlaceholder: "选择该音色时自动填入的文本，可不填。",
        addVoice: "新增音色",
        updateVoice: "更新音色",
        deleteVoice: "删除音色",
        refreshVoices: "刷新音色",
        voiceKindBuiltin: "系统内置",
        voiceKindCustom: "自定义",
        noVoices: "暂无可用音色。",
        builtinVoiceInfo: "系统内置音色，可用于生成，但不能修改或删除。",
        customVoiceInfo: "自定义音色，可修改标题、语言、默认文本，也可以替换录音。",
        voiceTitleRequired: "请输入显示标题。",
        voiceAudioRequired: "请选择一段音色录音。",
        voiceSelectRequired: "请选择一个音色条目。",
        voiceCreated: "自定义音色已新增。",
        voiceUpdated: "自定义音色已更新。",
        voiceDeleted: "自定义音色已删除。",
        voiceRefreshed: "音色列表已刷新。",
        voiceDeleteConfirm: "确定删除这个自定义音色吗？此操作不可撤销。",
        builtinModifyBlocked: "系统内置音色不能修改或删除。",
        filePicker: "选择文件 | 未选择任何文件",
        usingUploadedPrompt: "正在使用上传的参考音频：{name}",
        usingCustomPrompt: "正在使用自定义音色：{name}",
        usingDemoPromptPath: "正在使用 Demo 参考音频：{path}",
        usingDemoPrompt: "正在使用 Demo 参考音频。",
        playbackPlaceholder: "播放时会在这里高亮当前句子。",
        statusIdle: "空闲。",
        statusUnknown: "未知状态。",
        statusGeneratedReady: "生成音频已就绪。",
        statusDone: "完成。",
        statusGeneratingRealtime: "正在生成实时语音...",
        statusStreamingRealtime: "正在流式播放实时音频...",
        statusStreamComplete: "流式生成完成。",
        statusStartingRealtime: "正在启动实时流...",
        statusRunningSynthesis: "正在合成...",
        browserNoStreaming: "当前浏览器不支持 Web Audio 流式播放。",
        readableStreamMissing: "当前响应不支持 ReadableStream。",
        finalResultNotReady: "流式生成已结束，但最终结果尚未准备好。",
        metricState: "状态",
        metricEmitted: "已输出",
        metricLead: "缓冲领先",
        metricPlaybackDelay: "播放延迟",
        metricFirstAudio: "首音频",
      },
    };

    function readPreferredLanguage() {
      try {
        const savedLanguage = window.localStorage.getItem("moss_tts_language");
        if (savedLanguage === "zh" || savedLanguage === "en") {
          return savedLanguage;
        }
      } catch (error) {
      }
      return navigator.language && navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
    }

    let currentLanguage = readPreferredLanguage();

    function t(key, params = {}) {
      const dictionary = UI_TEXT[currentLanguage] || UI_TEXT.en;
      const fallback = UI_TEXT.en[key] || key;
      const template = dictionary[key] || fallback;
      return String(template).replace(/\{(\w+)\}/g, (_match, name) => {
        return params[name] === undefined || params[name] === null ? "" : String(params[name]);
      });
    }

    function setText(selector, value) {
      const node = document.querySelector(selector);
      if (node) {
        node.textContent = value;
      }
    }

    function setHtml(selector, value) {
      const node = document.querySelector(selector);
      if (node) {
        node.innerHTML = value;
      }
    }

    function setCheckboxLabel(inputId, value) {
      const input = document.getElementById(inputId);
      if (!input) {
        return;
      }
      const label = input.closest("label");
      if (!label) {
        return;
      }
      label.textContent = "";
      label.appendChild(input);
      label.append(` ${value}`);
    }

    function translateKnownText(node, keys) {
      if (!node) {
        return;
      }
      const currentText = node.textContent.trim();
      for (const key of keys) {
        if (currentText === UI_TEXT.en[key] || currentText === UI_TEXT.zh[key]) {
          node.textContent = t(key);
          return;
        }
      }
    }

    function applyLanguage(language) {
      currentLanguage = language === "en" ? "en" : "zh";
      try {
        window.localStorage.setItem("moss_tts_language", currentLanguage);
      } catch (error) {
      }

      document.documentElement.lang = t("htmlLang");
      document.title = IS_ONNX_DEMO ? `${t("title")} ONNX` : t("documentTitle");
      document.querySelector(".language-switch")?.setAttribute("aria-label", t("languageAria"));
      document.querySelector(".top-tabs")?.setAttribute("aria-label", t("demoModeAria"));

      for (const button of languageButtons) {
        const isActive = button.dataset.lang === currentLanguage;
        button.classList.toggle("active", isActive);
        button.setAttribute("aria-pressed", String(isActive));
      }

      setText(".hero h1", IS_ONNX_DEMO ? `${t("title")} ONNX` : t("title"));
      setText(".hero .lead", t("lead"));
      setHtml(".hero-points li:nth-child(1)", t("pointVoiceClone"));
      setHtml(".hero-points li:nth-child(2)", t("pointPresets"));
      setHtml(".build-note", t("buildNote"));
      setText(".top-tab.active", t("topTab"));
      setText('label[for="demo"]', t("labelDemo"));
      setText('label[for="prompt-audio-upload"]', t("labelPromptSpeech"));
      setText('label[for="text"]', t("labelText"));
      textInput.placeholder = t("placeholderText");
      setText("#voice-manager > summary", t("voiceLibrary"));
      setText("#generation-options > summary", t("generationOptions"));
      setText('label[for="voice-manager-select"]', t("voiceEntry"));
      setText('label[for="custom-voice-title"]', t("displayTitle"));
      customVoiceTitleInput.placeholder = t("displayTitlePlaceholder");
      setText('label[for="custom-voice-language"]', t("language"));
      setText('label[for="custom-voice-audio"]', t("voiceRecording"));
      setText('label[for="custom-voice-text"]', t("defaultTextOptional"));
      customVoiceTextInput.placeholder = t("defaultTextPlaceholder");
      setText("#create-voice-btn", t("addVoice"));
      setText("#update-voice-btn", t("updateVoice"));
      setText("#delete-voice-btn", t("deleteVoice"));
      setText("#refresh-voices-btn", t("refreshVoices"));
      setText('label[for="max-new-frames"]', t("maxNewFrames"));
      setText('label[for="voice-clone-max-text-tokens"]', t("voiceCloneMaxTextTokens"));
      setText('label[for="tts-max-batch-size"]', t("ttsMaxBatchSize"));
      setText('label[for="codec-max-batch-size"]', t("codecMaxBatchSize"));
      setText("#batch-note", t("batchNote"));
      setText('label[for="cpu-thread-count"]', t("cpuThreads"));
      setText("#cpu-note", t("cpuNote"));
      setText('label[for="attn-implementation"]', t(IS_ONNX_DEMO ? "samplingMode" : "attnImplementation"));
      setText('label[for="seed"]', t("seed"));
      setText('label[for="text-temperature"]', t("textTemperature"));
      setText('label[for="text-top-p"]', t("textTopP"));
      setText('label[for="text-top-k"]', t("textTopK"));
      setText('label[for="audio-temperature"]', t("audioTemperature"));
      setText('label[for="audio-top-p"]', t("audioTopP"));
      setText('label[for="audio-top-k"]', t("audioTopK"));
      setText('label[for="audio-repetition-penalty"]', t("audioRepetitionPenalty"));
      setCheckboxLabel("do-sample", t(IS_ONNX_DEMO ? "doSampleDerived" : "doSample"));
      setCheckboxLabel("enable-text-normalization", t("enableWetext"));
      setCheckboxLabel("enable-robust-text-normalization", t("enableRobust"));
      setText("#normalization-note", t("normalizationNote"));
      setCheckboxLabel("realtime-stream", t("realtimeDecode"));
      setText('label[for="initial-playback-delay-seconds"]', t("initialPlaybackDelay"));
      setText("#choose-prompt-audio-btn", t("chooseFile"));
      setText("#clear-prompt-audio-btn", t("resetDemoAudio"));
      setText("#generate-btn", t("generate"));
      setText("#refresh-btn", t("refreshWarmup"));

      const fieldTags = document.querySelectorAll(".output-panel .field-tag");
      const fieldTagKeys = [
        "warmupStatus",
        "textNormalizationStatus",
        "runStatus",
        "normalizedText",
        "playbackScript",
        "generatedSpeech",
      ];
      fieldTagKeys.forEach((key, index) => {
        if (fieldTags[index]) {
          fieldTags[index].textContent = t(key);
        }
      });
      setText("#checkpoint-label", t("checkpoint"));
      setText("#audio-tokenizer-label", t("audioTokenizer"));

      translateKnownText(runStatus, [
        "statusIdle",
        "statusDone",
        "statusGeneratedReady",
        "statusGeneratingRealtime",
        "statusStreamingRealtime",
        "statusStreamComplete",
        "statusStartingRealtime",
        "statusRunningSynthesis",
      ]);
      translateKnownText(resolvedPrompt, ["statusGeneratedReady", "statusGeneratingRealtime"]);
      if (playbackScript.classList.contains("empty")) {
        playbackScript.textContent = t("playbackPlaceholder");
      }
      renderDemoOptions(demoSelect.value || DEFAULT_DEMO_ID);
      renderVoiceManagerOptions(voiceManagerSelect.value);
      syncVoiceManagerForm();
      updatePromptAudioPanel(getSelectedDemo());
      updatePauseButtonState();
    }

    for (const button of languageButtons) {
      button.addEventListener("click", () => {
        applyLanguage(button.dataset.lang);
      });
    }

    function voiceKindLabel(demo) {
      return demo && demo.is_custom ? t("voiceKindCustom") : t("voiceKindBuiltin");
    }

    function languageLabel(language) {
      const labels = {
        zh: "中文",
        en: "English",
        ja: "日本語",
        ko: "한국어",
        other: currentLanguage === "zh" ? "其他" : "Other",
      };
      return labels[language] || labels.zh;
    }

    function formatDemoOptionLabel(demo) {
      if (!demo) {
        return "";
      }
      const language = demo.language ? languageLabel(demo.language) : "";
      const suffixParts = [voiceKindLabel(demo)];
      if (language) {
        suffixParts.push(language);
      }
      return `${demo.name} (${suffixParts.join(" / ")})`;
    }

    function renderDemoOptions(selectedId = "") {
      const previousId = selectedId || demoSelect.value || DEFAULT_DEMO_ID;
      demosById.clear();
      demoSelect.innerHTML = "";
      for (const demo of DEMOS) {
        demosById.set(demo.id, demo);
        const option = document.createElement("option");
        option.value = demo.id;
        option.textContent = formatDemoOptionLabel(demo);
        demoSelect.appendChild(option);
      }
      if (previousId && demosById.has(previousId)) {
        demoSelect.value = previousId;
      } else if (DEMOS.length > 0) {
        demoSelect.value = DEMOS[0].id;
      }
      generateBtn.disabled = DEMOS.length === 0 && !getUploadedPromptAudioFile();
    }

    function renderVoiceManagerOptions(selectedId = "") {
      const previousId = selectedId || voiceManagerSelect.value || demoSelect.value || DEFAULT_DEMO_ID;
      voiceManagerSelect.innerHTML = "";
      for (const demo of DEMOS) {
        const option = document.createElement("option");
        option.value = demo.id;
        option.textContent = formatDemoOptionLabel(demo);
        voiceManagerSelect.appendChild(option);
      }
      if (previousId && demosById.has(previousId)) {
        voiceManagerSelect.value = previousId;
      } else if (DEMOS.length > 0) {
        voiceManagerSelect.value = DEMOS[0].id;
      }
    }

    function getManagedVoice() {
      return demosById.get(voiceManagerSelect.value) || null;
    }

    function setVoiceManagerStatus(message, isError = false) {
      voiceManagerStatus.textContent = message || "";
      voiceManagerStatus.style.color = isError ? "var(--danger)" : "var(--muted)";
    }

    function syncVoiceManagerForm() {
      const voice = getManagedVoice();
      if (!voice) {
        customVoiceTitleInput.value = "";
        customVoiceLanguageSelect.value = "zh";
        customVoiceTextInput.value = "";
        voiceManagerSelectedInfo.textContent = t("noVoices");
        voiceManagerPreview.hidden = true;
        voiceManagerPreview.removeAttribute("src");
        updateVoiceBtn.disabled = true;
        deleteVoiceBtn.disabled = true;
        return;
      }

      customVoiceTitleInput.value = voice.name || "";
      customVoiceLanguageSelect.value = voice.language || "zh";
      customVoiceTextInput.value = voice.text || "";
      const isCustom = Boolean(voice.is_custom);
      voiceManagerSelectedInfo.textContent = isCustom ? t("customVoiceInfo") : t("builtinVoiceInfo");
      updateVoiceBtn.disabled = !isCustom;
      deleteVoiceBtn.disabled = !isCustom;
      if (voice.id) {
        voiceManagerPreview.src = getDemoPromptAudioUrl(voice.id);
        voiceManagerPreview.hidden = false;
      } else {
        voiceManagerPreview.hidden = true;
        voiceManagerPreview.removeAttribute("src");
      }
    }

    async function refreshVoiceLibrary(selectedId = "") {
      const payload = await fetchJson(`${APP_BASE}/api/voices`);
      DEMOS = Array.isArray(payload.voices) ? payload.voices : [];
      renderDemoOptions(selectedId || demoSelect.value);
      renderVoiceManagerOptions(selectedId || voiceManagerSelect.value);
      syncVoiceManagerForm();
      updatePromptAudioPanel(getSelectedDemo());
      return payload;
    }

    function getSelectedDemo() {
      return demosById.get(demoSelect.value) || DEMOS[0] || null;
    }

    function getUploadedPromptAudioFile() {
      const files = promptAudioUploadInput.files;
      return files && files.length > 0 ? files[0] : null;
    }

    function clearPromptAudioPreviewUrl() {
      if (!currentPromptAudioPreviewUrl) {
        return;
      }
      URL.revokeObjectURL(currentPromptAudioPreviewUrl);
      currentPromptAudioPreviewUrl = null;
    }

    function getDemoPromptAudioUrl(demoId) {
      return `${APP_BASE}/api/demo-prompt-audio/${encodeURIComponent(demoId)}`;
    }

    function showPromptAudioFilePicker(message = t("filePicker")) {
      promptAudioPreview.pause();
      promptAudioPreview.removeAttribute("src");
      promptAudioPreview.load();
      promptAudioPreview.hidden = true;
      promptAudioUploadInput.hidden = false;
      choosePromptAudioBtn.hidden = true;
      clearPromptAudioBtn.hidden = true;
      promptAudioSource.textContent = message;
    }

    function showPromptAudioPreview({
      sourceUrl,
      message,
      showResetToDemo,
    }) {
      promptAudioPreview.src = sourceUrl;
      promptAudioPreview.hidden = false;
      promptAudioUploadInput.hidden = true;
      choosePromptAudioBtn.hidden = false;
      clearPromptAudioBtn.hidden = !showResetToDemo;
      promptAudioSource.textContent = message;
    }

    function updatePromptAudioPanel(demo = getSelectedDemo()) {
      const uploadedPromptAudio = getUploadedPromptAudioFile();
      if (uploadedPromptAudio) {
        clearPromptAudioPreviewUrl();
        currentPromptAudioPreviewUrl = URL.createObjectURL(uploadedPromptAudio);
        showPromptAudioPreview({
          sourceUrl: currentPromptAudioPreviewUrl,
          message: t("usingUploadedPrompt", { name: uploadedPromptAudio.name }),
          showResetToDemo: true,
        });
        return;
      }

      clearPromptAudioPreviewUrl();

      if (demo && demo.id) {
        showPromptAudioPreview({
          sourceUrl: getDemoPromptAudioUrl(demo.id),
          message: demo.is_custom
            ? t("usingCustomPrompt", { name: demo.name || demo.prompt_speech || demo.id })
            : demo.prompt_speech
            ? t("usingDemoPromptPath", { path: demo.prompt_speech })
            : t("usingDemoPrompt"),
          showResetToDemo: false,
        });
        return;
      }

      showPromptAudioFilePicker();
    }

    function applySelectedDemo(replaceText = true) {
      const demo = getSelectedDemo();
      const uploadedPromptAudio = getUploadedPromptAudioFile();

      if (!demo && !uploadedPromptAudio) {
        resolvedPrompt.textContent = "";
        if (replaceText) {
          textInput.value = "";
        }
        updatePromptAudioPanel(null);
        renderPlaybackScript([], textInput.value);
        generateBtn.disabled = true;
        return;
      }

      if (demo && replaceText && demo.text) {
        textInput.value = demo.text;
      }
      updatePromptAudioPanel(demo);
      previewPlaybackScriptFromInputs();
      generateBtn.disabled = false;
    }

    function setStatus(node, text, isError = false) {
      node.textContent = text;
      node.classList.toggle("error", Boolean(isError));
    }

    function clearAudioOutput() {
      if (currentAudioObjectUrl) {
        URL.revokeObjectURL(currentAudioObjectUrl);
        currentAudioObjectUrl = null;
      }
      audioOutput.pause();
      audioOutput.removeAttribute("src");
      audioOutput.load();
      bufferedPlaybackBoundaries = [];
      currentPlaybackMarkedComplete = false;
      setPlaybackHighlight(null);
      updatePauseButtonState();
    }

    function clearNormalizedOutputs() {
      normalizedTextOutput.value = "";
    }

    function updateNormalizedOutputs(payload) {
      normalizedTextOutput.value = payload.normalized_text || "";
    }

    function splitTextForDisplay(text) {
      const normalizedText = String(text || "").trim();
      if (!normalizedText) {
        return [];
      }

      const segments = [];
      for (const rawLine of normalizedText.split(/\\n+/)) {
        const line = rawLine.trim();
        if (!line) {
          continue;
        }
        const parts = line.match(/[^。！？!?；;.\\n]+(?:[。！？!?；;]+|\\.(?=\\s|$)|$)/g);
        if (parts && parts.length > 1) {
          for (const part of parts) {
            const cleanedPart = part.trim();
            if (cleanedPart) {
              segments.push(cleanedPart);
            }
          }
          continue;
        }
        segments.push(line);
      }
      return segments.length ? segments : [normalizedText];
    }

    function normalizePlaybackChunks(chunks, fallbackText = "") {
      if (Array.isArray(chunks)) {
        const normalizedChunks = chunks
          .map((chunk) => String(chunk || "").trim())
          .filter(Boolean);
        if (normalizedChunks.length > 0) {
          return normalizedChunks;
        }
      }
      return splitTextForDisplay(fallbackText);
    }

    function rebuildBufferedPlaybackBoundaries() {
      bufferedPlaybackBoundaries = [];
      if (!playbackChunks.length) {
        return;
      }
      const duration = Number(audioOutput.duration);
      if (!Number.isFinite(duration) || duration <= 0) {
        return;
      }

      const weights = playbackChunks.map((chunk) => {
        const compactChunk = String(chunk).replace(/\\s+/g, "");
        return Math.max(1, Array.from(compactChunk).length);
      });
      const totalWeight = weights.reduce((sum, value) => sum + value, 0);
      let accumulatedSeconds = 0;
      bufferedPlaybackBoundaries = weights.map((weight) => {
        accumulatedSeconds += (duration * weight) / totalWeight;
        return accumulatedSeconds;
      });
    }

    function setPlaybackHighlight(activeIndex, options = {}) {
      const markAllPlayed = Boolean(options.markAllPlayed);
      const segmentNodes = playbackScript.querySelectorAll(".playback-segment");
      let normalizedActiveIndex = null;
      if (segmentNodes.length > 0 && activeIndex !== null && activeIndex !== undefined) {
        const numericIndex = Number(activeIndex);
        if (Number.isFinite(numericIndex)) {
          normalizedActiveIndex = Math.max(0, Math.min(segmentNodes.length - 1, Math.trunc(numericIndex)));
        }
      }

      const shouldScroll =
        normalizedActiveIndex !== null &&
        normalizedActiveIndex !== currentPlaybackChunkIndex &&
        !markAllPlayed;

      segmentNodes.forEach((node, index) => {
        const isActive = normalizedActiveIndex !== null && index === normalizedActiveIndex && !markAllPlayed;
        const isPlayed = markAllPlayed || (normalizedActiveIndex !== null && index < normalizedActiveIndex);
        node.classList.toggle("active", isActive);
        node.classList.toggle("played", isPlayed);
      });

      if (shouldScroll) {
        segmentNodes[normalizedActiveIndex].scrollIntoView({
          block: "nearest",
          inline: "nearest",
          behavior: "smooth"
        });
      }

      currentPlaybackChunkIndex = normalizedActiveIndex;
      currentPlaybackMarkedComplete = markAllPlayed;
    }

    function renderPlaybackScript(chunks, fallbackText = "") {
      playbackChunks = normalizePlaybackChunks(chunks, fallbackText);
      bufferedPlaybackBoundaries = [];
      currentPlaybackChunkIndex = null;
      currentPlaybackMarkedComplete = false;
      playbackScript.innerHTML = "";

      if (!playbackChunks.length) {
        playbackScript.classList.add("empty");
        playbackScript.textContent = t("playbackPlaceholder");
        return;
      }

      playbackScript.classList.remove("empty");
      for (const chunk of playbackChunks) {
        const segmentNode = document.createElement("span");
        segmentNode.className = "playback-segment";
        segmentNode.textContent = chunk;
        playbackScript.appendChild(segmentNode);
      }
      rebuildBufferedPlaybackBoundaries();
      setPlaybackHighlight(null);
    }

    function updateBufferedPlaybackHighlight() {
      if (hasRealtimePlayback()) {
        return;
      }
      if (!hasBufferedPlayback() || !playbackChunks.length || currentPlaybackMarkedComplete) {
        return;
      }
      if (audioOutput.ended) {
        setPlaybackHighlight(null, { markAllPlayed: true });
        return;
      }
      if (!Number.isFinite(audioOutput.duration) || audioOutput.duration <= 0) {
        if (!audioOutput.paused) {
          setPlaybackHighlight(0);
        }
        return;
      }
      if (!bufferedPlaybackBoundaries.length) {
        rebuildBufferedPlaybackBoundaries();
      }
      if (!bufferedPlaybackBoundaries.length) {
        return;
      }

      const clampedCurrentTime = Math.max(0, Math.min(Number(audioOutput.currentTime || 0), Number(audioOutput.duration)));
      let activeIndex = bufferedPlaybackBoundaries.findIndex((boundary) => clampedCurrentTime <= boundary);
      if (activeIndex < 0) {
        activeIndex = playbackChunks.length - 1;
      }
      setPlaybackHighlight(activeIndex);
    }

    function previewPlaybackScriptFromInputs() {
      if (hasBufferedPlayback() || hasRealtimePlayback()) {
        return;
      }
      renderPlaybackScript([], textInput.value);
    }

    function hasBufferedPlayback() {
      return Boolean(audioOutput.getAttribute("src"));
    }

    function hasRealtimePlayback() {
      return Boolean(currentAudioContext);
    }

    function updatePauseButtonState() {
      if (hasRealtimePlayback()) {
        pauseBtn.disabled = false;
        pauseBtn.textContent = currentRealtimePlaybackPaused ? t("resumePlayback") : t("pausePlayback");
        return;
      }
      if (hasBufferedPlayback()) {
        pauseBtn.disabled = false;
        pauseBtn.textContent = audioOutput.paused ? t("resumePlayback") : t("pausePlayback");
        return;
      }
      pauseBtn.disabled = true;
      pauseBtn.textContent = t("pausePlayback");
    }

    function resetRealtimePlaybackTracking() {
      currentRealtimePlaybackStartAt = null;
      currentRealtimePlaybackScheduledAudioSeconds = 0;
      currentRealtimePlaybackChunkRanges = [];
    }

    function normalizeRealtimeChunkRanges(rawRanges) {
      if (!Array.isArray(rawRanges)) {
        return [];
      }
      const normalizedRanges = [];
      for (const rawRange of rawRanges) {
        if (!Array.isArray(rawRange) || rawRange.length < 3) {
          continue;
        }
        const startSeconds = Number(rawRange[0]);
        const endSeconds = Number(rawRange[1]);
        const chunkIndex = Number(rawRange[2]);
        if (!Number.isFinite(startSeconds) || !Number.isFinite(endSeconds) || !Number.isFinite(chunkIndex)) {
          continue;
        }
        normalizedRanges.push([
          Math.max(0, startSeconds),
          Math.max(0, endSeconds),
          Math.max(0, Math.trunc(chunkIndex))
        ]);
      }
      normalizedRanges.sort((a, b) => a[1] - b[1]);
      return normalizedRanges;
    }

    function resolveRealtimeChunkIndexByPlaybackSeconds(playbackSeconds) {
      if (!currentRealtimePlaybackChunkRanges.length) {
        return null;
      }
      const clampedSeconds = Math.max(0, Number(playbackSeconds) || 0);
      for (const [, endSeconds, chunkIndex] of currentRealtimePlaybackChunkRanges) {
        if (clampedSeconds <= endSeconds + 1e-6) {
          return chunkIndex;
        }
      }
      return currentRealtimePlaybackChunkRanges[currentRealtimePlaybackChunkRanges.length - 1][2];
    }

    function updateRealtimePlaybackHighlightFromLocalClock() {
      if (!currentAudioContext || currentRealtimePlaybackPaused) {
        return;
      }
      if (currentRealtimePlaybackStartAt === null || !currentRealtimePlaybackChunkRanges.length) {
        return;
      }
      const elapsedPlaybackSeconds = Math.max(0, currentAudioContext.currentTime - currentRealtimePlaybackStartAt);
      const clampedPlaybackSeconds = Math.min(
        elapsedPlaybackSeconds,
        currentRealtimePlaybackScheduledAudioSeconds > 0
          ? currentRealtimePlaybackScheduledAudioSeconds
          : elapsedPlaybackSeconds
      );
      const chunkIndex = resolveRealtimeChunkIndexByPlaybackSeconds(clampedPlaybackSeconds);
      if (chunkIndex !== null && chunkIndex !== undefined) {
        setPlaybackHighlight(chunkIndex);
      }
    }

    function base64ToBlob(base64Value, mimeType) {
      const binary = atob(base64Value);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      return new Blob([bytes], { type: mimeType });
    }

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      const text = await response.text();
      if (!response.ok) {
        throw new Error(text || `HTTP ${response.status}`);
      }
      return text ? JSON.parse(text) : {};
    }

    function buildVoiceFormData({ requireAudio }) {
      const title = customVoiceTitleInput.value.trim();
      if (!title) {
        throw new Error(t("voiceTitleRequired"));
      }
      const audioFile = customVoiceAudioInput.files && customVoiceAudioInput.files.length > 0
        ? customVoiceAudioInput.files[0]
        : null;
      if (requireAudio && !audioFile) {
        throw new Error(t("voiceAudioRequired"));
      }
      const formData = new FormData();
      formData.append("title", title);
      formData.append("language", customVoiceLanguageSelect.value || "zh");
      formData.append("text", customVoiceTextInput.value || "");
      if (audioFile) {
        formData.append("prompt_audio", audioFile);
      }
      return formData;
    }

    async function createCustomVoice() {
      try {
        setVoiceManagerStatus("");
        const payload = await fetchJson(`${APP_BASE}/api/voices`, {
          method: "POST",
          body: buildVoiceFormData({ requireAudio: true }),
        });
        const voiceId = payload.voice && payload.voice.id ? payload.voice.id : "";
        customVoiceAudioInput.value = "";
        await refreshVoiceLibrary(voiceId);
        if (voiceId) {
          demoSelect.value = voiceId;
          voiceManagerSelect.value = voiceId;
          syncVoiceManagerForm();
          applySelectedDemo(false);
        }
        setVoiceManagerStatus(t("voiceCreated"));
      } catch (error) {
        setVoiceManagerStatus(String(error), true);
      }
    }

    async function updateCustomVoice() {
      const voice = getManagedVoice();
      if (!voice) {
        setVoiceManagerStatus(t("voiceSelectRequired"), true);
        return;
      }
      if (!voice.is_custom) {
        setVoiceManagerStatus(t("builtinModifyBlocked"), true);
        return;
      }
      try {
        setVoiceManagerStatus("");
        const payload = await fetchJson(`${APP_BASE}/api/voices/${encodeURIComponent(voice.id)}`, {
          method: "PUT",
          body: buildVoiceFormData({ requireAudio: false }),
        });
        const voiceId = payload.voice && payload.voice.id ? payload.voice.id : voice.id;
        customVoiceAudioInput.value = "";
        await refreshVoiceLibrary(voiceId);
        demoSelect.value = voiceId;
        voiceManagerSelect.value = voiceId;
        syncVoiceManagerForm();
        updatePromptAudioPanel(getSelectedDemo());
        setVoiceManagerStatus(t("voiceUpdated"));
      } catch (error) {
        setVoiceManagerStatus(String(error), true);
      }
    }

    async function deleteCustomVoice() {
      const voice = getManagedVoice();
      if (!voice) {
        setVoiceManagerStatus(t("voiceSelectRequired"), true);
        return;
      }
      if (!voice.is_custom) {
        setVoiceManagerStatus(t("builtinModifyBlocked"), true);
        return;
      }
      if (!window.confirm(t("voiceDeleteConfirm"))) {
        return;
      }
      try {
        setVoiceManagerStatus("");
        await fetchJson(`${APP_BASE}/api/voices/${encodeURIComponent(voice.id)}`, {
          method: "DELETE",
        });
        customVoiceAudioInput.value = "";
        await refreshVoiceLibrary(DEFAULT_DEMO_ID);
        applySelectedDemo(false);
        setVoiceManagerStatus(t("voiceDeleted"));
      } catch (error) {
        setVoiceManagerStatus(String(error), true);
      }
    }

    function buildFormData() {
      const demo = getSelectedDemo();
      const uploadedPromptAudio = getUploadedPromptAudioFile();
      const formData = new FormData();
      formData.append("text", textInput.value);
      if (demo) {
        formData.append("demo_id", demo.id);
      }
      if (uploadedPromptAudio) {
        formData.append("prompt_audio", uploadedPromptAudio);
      }
      formData.append("max_new_frames", document.getElementById("max-new-frames").value);
      formData.append("voice_clone_max_text_tokens", document.getElementById("voice-clone-max-text-tokens").value);
      formData.append("attn_implementation", document.getElementById("attn-implementation").value);
      formData.append("do_sample", document.getElementById("do-sample").checked ? "1" : "0");
      formData.append("text_temperature", document.getElementById("text-temperature").value);
      formData.append("text_top_p", document.getElementById("text-top-p").value);
      formData.append("text_top_k", document.getElementById("text-top-k").value);
      formData.append("audio_temperature", document.getElementById("audio-temperature").value);
      formData.append("audio_top_p", document.getElementById("audio-top-p").value);
      formData.append("audio_top_k", document.getElementById("audio-top-k").value);
      formData.append("audio_repetition_penalty", document.getElementById("audio-repetition-penalty").value);
      formData.append("seed", document.getElementById("seed").value);
      formData.append("tts_max_batch_size", document.getElementById("tts-max-batch-size").value);
      formData.append("codec_max_batch_size", document.getElementById("codec-max-batch-size").value);
      formData.append("enable_text_normalization", document.getElementById("enable-text-normalization").checked ? "1" : "0");
      formData.append("enable_normalize_tts_text", document.getElementById("enable-robust-text-normalization").checked ? "1" : "0");
      formData.append("cpu_threads", document.getElementById("cpu-thread-count").value || String(DEFAULT_CPU_THREADS));
      return formData;
    }

    function mergeUint8Arrays(a, b) {
      const merged = new Uint8Array(a.length + b.length);
      merged.set(a, 0);
      merged.set(b, a.length);
      return merged;
    }

    function resolveInitialPlaybackDelaySeconds() {
      const rawValue = Number(initialPlaybackDelayInput.value);
      if (!Number.isFinite(rawValue)) {
        return 0.08;
      }
      return Math.max(0.0, rawValue);
    }

    function schedulePcmChunk(pcmChunk, sampleRate, channels) {
      if (!currentAudioContext || pcmChunk.byteLength <= 0) {
        return;
      }
      const bytesPerFrame = channels * 2;
      const totalFrames = Math.floor(pcmChunk.byteLength / bytesPerFrame);
      if (totalFrames <= 0) {
        return;
      }

      const audioBuffer = currentAudioContext.createBuffer(channels, totalFrames, sampleRate);
      const view = new DataView(pcmChunk.buffer, pcmChunk.byteOffset, totalFrames * bytesPerFrame);
      for (let channelIndex = 0; channelIndex < channels; channelIndex += 1) {
        const channelData = audioBuffer.getChannelData(channelIndex);
        for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
          const byteOffset = (frameIndex * channels + channelIndex) * 2;
          channelData[frameIndex] = view.getInt16(byteOffset, true) / 32768.0;
        }
      }

      const source = currentAudioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(currentAudioContext.destination);
      const now = currentAudioContext.currentTime;
      const startAt = Math.max(nextPlaybackTime || (now + currentInitialPlaybackDelaySeconds), now + 0.02);
      source.start(startAt);
      const endAt = startAt + audioBuffer.duration;
      nextPlaybackTime = endAt;
      if (currentRealtimePlaybackStartAt === null) {
        currentRealtimePlaybackStartAt = startAt;
      }
      currentRealtimePlaybackScheduledAudioSeconds = Math.max(
        currentRealtimePlaybackScheduledAudioSeconds,
        endAt - currentRealtimePlaybackStartAt
      );
    }

    function clearRealtimePlaybackCompletionTimer() {
      if (currentRealtimePlaybackCompletionTimer) {
        window.clearTimeout(currentRealtimePlaybackCompletionTimer);
        currentRealtimePlaybackCompletionTimer = null;
      }
    }

    function monitorRealtimePlaybackCompletion() {
      clearRealtimePlaybackCompletionTimer();
      if (!currentAudioContext) {
        return;
      }

      async function pollRealtimePlaybackCompletion() {
        if (!currentAudioContext) {
          return;
        }
        if (currentRealtimePlaybackPaused) {
          currentRealtimePlaybackCompletionTimer = window.setTimeout(() => {
            pollRealtimePlaybackCompletion().catch(() => {});
          }, 120);
          return;
        }

        updateRealtimePlaybackHighlightFromLocalClock();

        const remainingSeconds = nextPlaybackTime - currentAudioContext.currentTime;
        if (remainingSeconds > 0.05) {
          currentRealtimePlaybackCompletionTimer = window.setTimeout(() => {
            pollRealtimePlaybackCompletion().catch(() => {});
          }, 120);
          return;
        }

        setPlaybackHighlight(null, { markAllPlayed: true });
        try {
          await currentAudioContext.close();
        } catch (error) {
        }
        currentAudioContext = null;
        currentRealtimePlaybackPaused = false;
        nextPlaybackTime = 0;
        resetRealtimePlaybackTracking();
        updatePauseButtonState();
      }

      currentRealtimePlaybackCompletionTimer = window.setTimeout(() => {
        pollRealtimePlaybackCompletion().catch(() => {});
      }, 120);
    }

    async function closeRealtimeStream() {
      clearRealtimePlaybackCompletionTimer();
      if (currentStreamStatusTimer) {
        window.clearInterval(currentStreamStatusTimer);
        currentStreamStatusTimer = null;
      }
      if (currentStreamAbortController) {
        currentStreamAbortController.abort();
        currentStreamAbortController = null;
      }
      if (currentStreamId) {
        fetch(`${APP_BASE}/api/generate-stream/${encodeURIComponent(currentStreamId)}/close`, {
          method: "POST"
        }).catch(() => {});
        currentStreamId = null;
      }
      if (currentAudioContext) {
        try {
          await currentAudioContext.close();
        } catch (error) {
        }
        currentAudioContext = null;
      }
      nextPlaybackTime = 0;
      currentRealtimePlaybackPaused = false;
      resetRealtimePlaybackTracking();
      currentPlaybackMarkedComplete = false;
      setPlaybackHighlight(null);
      updatePauseButtonState();
    }

    async function refreshWarmupStatus() {
      try {
        const [warmupData, normalizationData] = await Promise.all([
          fetchJson(`${APP_BASE}/api/warmup-status`),
          fetchJson(`${APP_BASE}/api/text-normalization-status`),
        ]);
        setStatus(warmupStatus, warmupData.status_text || t("statusUnknown"));
        setStatus(
          textNormalizationStatus,
          normalizationData.status_text || t("statusUnknown"),
          Boolean(normalizationData.failed)
        );
      } catch (error) {
        setStatus(warmupStatus, String(error), true);
        setStatus(textNormalizationStatus, String(error), true);
      }
    }

    async function generateBuffered(formData) {
      clearAudioOutput();
      streamMetrics.textContent = "";
      const data = await fetchJson(`${APP_BASE}/api/generate`, {
        method: "POST",
        body: formData
      });

      updateNormalizedOutputs(data);
      renderPlaybackScript(data.text_chunks, data.normalized_text || textInput.value);
      const audioBlob = base64ToBlob(data.audio_base64, "audio/wav");
      currentAudioObjectUrl = URL.createObjectURL(audioBlob);
      audioOutput.src = currentAudioObjectUrl;
      audioOutput.play().catch(() => {});
      rebuildBufferedPlaybackBoundaries();
      updateBufferedPlaybackHighlight();
      updatePauseButtonState();
      resolvedPrompt.textContent = t("statusGeneratedReady");
      setStatus(runStatus, data.run_status || t("statusDone"));
      if (data.warmup_status_text) {
        setStatus(warmupStatus, data.warmup_status_text);
      }
      if (data.text_normalization_status_text) {
        setStatus(textNormalizationStatus, data.text_normalization_status_text);
      }
    }

    async function generateRealtime(formData) {
      await closeRealtimeStream();
      clearAudioOutput();
      resolvedPrompt.textContent = t("statusGeneratingRealtime");
      streamMetrics.textContent = "";

      const startData = await fetchJson(`${APP_BASE}/api/generate-stream/start`, {
        method: "POST",
        body: formData
      });

      currentStreamId = startData.stream_id;
      setStatus(runStatus, startData.run_status || t("statusStreamingRealtime"));
      if (startData.warmup_status_text) {
        setStatus(warmupStatus, startData.warmup_status_text);
      }
      if (startData.text_normalization_status_text) {
        setStatus(textNormalizationStatus, startData.text_normalization_status_text);
      }
      updateNormalizedOutputs(startData);
      renderPlaybackScript(startData.text_chunks, startData.normalized_text || textInput.value);

      const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextCtor) {
        throw new Error(t("browserNoStreaming"));
      }
      currentInitialPlaybackDelaySeconds = resolveInitialPlaybackDelaySeconds();
      currentAudioContext = new AudioContextCtor({ sampleRate: startData.sample_rate });
      await currentAudioContext.resume();
      nextPlaybackTime = currentAudioContext.currentTime + currentInitialPlaybackDelaySeconds;
      currentRealtimePlaybackPaused = false;
      resetRealtimePlaybackTracking();
      clearRealtimePlaybackCompletionTimer();
      updatePauseButtonState();
      currentStreamAbortController = new AbortController();

      async function updateStreamStatus() {
        if (!currentStreamId) {
          return null;
        }
        const snapshot = await fetchJson(startData.status_url);
        if (snapshot.run_status) {
          setStatus(runStatus, snapshot.run_status, Boolean(snapshot.failed));
        }
        const metrics = [
          `${t("metricState")}=${snapshot.state}`,
          `${t("metricEmitted")}=${Number(snapshot.emitted_audio_seconds || 0).toFixed(2)}s`,
          `${t("metricLead")}=${Number(snapshot.lead_seconds || 0).toFixed(2)}s`,
          `${t("metricPlaybackDelay")}=${currentInitialPlaybackDelaySeconds.toFixed(2)}s`
        ];
        if (snapshot.first_audio_latency_seconds !== null && snapshot.first_audio_latency_seconds !== undefined) {
          metrics.push(`${t("metricFirstAudio")}=${Number(snapshot.first_audio_latency_seconds).toFixed(2)}s`);
        }
        streamMetrics.textContent = metrics.join(" | ");
        if (!currentRealtimePlaybackPaused && snapshot.playback_chunk_index !== null && snapshot.playback_chunk_index !== undefined) {
          setPlaybackHighlight(snapshot.playback_chunk_index);
        }
        return snapshot;
      }

      currentStreamStatusTimer = window.setInterval(() => {
        updateStreamStatus().catch(() => {});
      }, 500);
      await updateStreamStatus();

      const response = await fetch(startData.audio_url, {
        signal: currentStreamAbortController.signal
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }
      if (!response.body) {
        throw new Error(t("readableStreamMissing"));
      }

      const reader = response.body.getReader();
      const channels = Number(startData.channels || 2);
      const sampleRate = Number(startData.sample_rate || 48000);
      const bytesPerFrame = channels * 2;
      let remainder = new Uint8Array(0);

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (!value || value.length === 0) {
          continue;
        }
        const merged = mergeUint8Arrays(remainder, value);
        const alignedLength = Math.floor(merged.length / bytesPerFrame) * bytesPerFrame;
        if (alignedLength <= 0) {
          remainder = merged;
          continue;
        }
        schedulePcmChunk(merged.subarray(0, alignedLength), sampleRate, channels);
        remainder = merged.subarray(alignedLength);
      }

      await updateStreamStatus();
      if (currentStreamStatusTimer) {
        window.clearInterval(currentStreamStatusTimer);
        currentStreamStatusTimer = null;
      }
      let result = null;
      for (let attempt = 0; attempt < 40; attempt += 1) {
        result = await fetchJson(startData.result_url);
        if (result.ready) {
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 100));
      }
      if (!result || !result.ready) {
        throw new Error(t("finalResultNotReady"));
      }
      if (result.audio_base64) {
        if (currentAudioObjectUrl) {
          URL.revokeObjectURL(currentAudioObjectUrl);
          currentAudioObjectUrl = null;
        }
        const audioBlob = base64ToBlob(result.audio_base64, "audio/wav");
        currentAudioObjectUrl = URL.createObjectURL(audioBlob);
        audioOutput.src = currentAudioObjectUrl;
        audioOutput.load();
        rebuildBufferedPlaybackBoundaries();
      }
      resolvedPrompt.textContent = t("statusGeneratedReady");
      streamMetrics.textContent = result.stream_metrics || streamMetrics.textContent;
      if (Array.isArray(result.text_chunks) && result.text_chunks.length > 0 && playbackChunks.length === 0) {
        renderPlaybackScript(result.text_chunks, normalizedTextOutput.value || textInput.value);
      }
      currentRealtimePlaybackChunkRanges = normalizeRealtimeChunkRanges(result.audio_chunk_ranges);
      updateRealtimePlaybackHighlightFromLocalClock();
      setStatus(runStatus, result.run_status || t("statusStreamComplete"));
      if (currentStreamId) {
        fetch(`${APP_BASE}/api/generate-stream/${encodeURIComponent(currentStreamId)}/close`, {
          method: "POST"
        }).catch(() => {});
        currentStreamId = null;
      }
      monitorRealtimePlaybackCompletion();
      updatePauseButtonState();
    }

    async function togglePausePlayback() {
      if (hasRealtimePlayback()) {
        if (currentRealtimePlaybackPaused) {
          await currentAudioContext.resume();
          currentRealtimePlaybackPaused = false;
          updateRealtimePlaybackHighlightFromLocalClock();
        } else {
          await currentAudioContext.suspend();
          currentRealtimePlaybackPaused = true;
        }
        updatePauseButtonState();
        return;
      }

      if (hasBufferedPlayback()) {
        if (audioOutput.paused) {
          await audioOutput.play().catch(() => {});
        } else {
          audioOutput.pause();
        }
        updatePauseButtonState();
      }
    }

    async function generate() {
      generateBtn.disabled = true;
      refreshBtn.disabled = true;
      setStatus(runStatus, realtimeStreamToggle.checked ? t("statusStartingRealtime") : t("statusRunningSynthesis"));

      try {
        const formData = buildFormData();
        clearNormalizedOutputs();
        renderPlaybackScript([], textInput.value);
        if (realtimeStreamToggle.checked) {
          await generateRealtime(formData);
        } else {
          await closeRealtimeStream();
          await generateBuffered(formData);
        }
      } catch (error) {
        if (realtimeStreamToggle.checked) {
          await closeRealtimeStream();
        }
        setStatus(runStatus, String(error), true);
      } finally {
        generateBtn.disabled = false;
        refreshBtn.disabled = false;
      }
    }

    generateBtn.addEventListener("click", generate);
    demoSelect.addEventListener("change", async () => {
      await closeRealtimeStream();
      clearAudioOutput();
      applySelectedDemo(true);
      clearNormalizedOutputs();
      resolvedPrompt.textContent = "";
      streamMetrics.textContent = "";
      setStatus(runStatus, t("statusIdle"));
    });
    promptAudioUploadInput.addEventListener("change", () => {
      applySelectedDemo(false);
      clearNormalizedOutputs();
      resolvedPrompt.textContent = "";
      setStatus(runStatus, t("statusIdle"));
    });
    choosePromptAudioBtn.addEventListener("click", () => {
      promptAudioUploadInput.click();
    });
    clearPromptAudioBtn.addEventListener("click", () => {
      promptAudioUploadInput.value = "";
      applySelectedDemo(false);
      clearNormalizedOutputs();
      resolvedPrompt.textContent = "";
      setStatus(runStatus, t("statusIdle"));
    });
    voiceManagerSelect.addEventListener("change", syncVoiceManagerForm);
    createVoiceBtn.addEventListener("click", () => {
      createCustomVoice().catch((error) => setVoiceManagerStatus(String(error), true));
    });
    updateVoiceBtn.addEventListener("click", () => {
      updateCustomVoice().catch((error) => setVoiceManagerStatus(String(error), true));
    });
    deleteVoiceBtn.addEventListener("click", () => {
      deleteCustomVoice().catch((error) => setVoiceManagerStatus(String(error), true));
    });
    refreshVoicesBtn.addEventListener("click", () => {
      refreshVoiceLibrary(voiceManagerSelect.value)
        .then(() => setVoiceManagerStatus(t("voiceRefreshed")))
        .catch((error) => setVoiceManagerStatus(String(error), true));
    });
    pauseBtn.addEventListener("click", () => {
      togglePausePlayback().catch((error) => {
        setStatus(runStatus, String(error), true);
      });
    });
    refreshBtn.addEventListener("click", refreshWarmupStatus);
    textInput.addEventListener("input", previewPlaybackScriptFromInputs);
    audioOutput.addEventListener("loadedmetadata", () => {
      rebuildBufferedPlaybackBoundaries();
      updateBufferedPlaybackHighlight();
    });
    audioOutput.addEventListener("timeupdate", updateBufferedPlaybackHighlight);
    audioOutput.addEventListener("play", () => {
      currentPlaybackMarkedComplete = false;
      updatePauseButtonState();
      updateBufferedPlaybackHighlight();
    });
    audioOutput.addEventListener("pause", updatePauseButtonState);
    audioOutput.addEventListener("ended", () => {
      updatePauseButtonState();
      setPlaybackHighlight(null, { markAllPlayed: true });
    });
    window.addEventListener("beforeunload", () => {
      clearPromptAudioPreviewUrl();
    });
    renderDemoOptions(DEFAULT_DEMO_ID);
    renderVoiceManagerOptions(DEFAULT_DEMO_ID);
    applyLanguage(currentLanguage);
    setStatus(runStatus, t("statusIdle"));
    applySelectedDemo(true);
    refreshWarmupStatus();
    window.setInterval(refreshWarmupStatus, 5000);
  </script>
</body>
</html>
"""
    demos_payload = [_demo_entry_to_payload(demo_entry) for demo_entry in demo_entries]
    replacements = {
        "__APP_BASE__": json.dumps(base_path),
        "__DEMOS__": json.dumps(demos_payload, ensure_ascii=False),
        "__DEFAULT_DEMO_ID__": json.dumps(demo_entries[0].demo_id if demo_entries else ""),
        "__DEFAULT_ATTN_IMPLEMENTATION__": json.dumps(runtime.attn_implementation or "model_default"),
        "__DEFAULT_CPU_THREADS__": json.dumps(max(1, int(os.cpu_count() or 1))),
        "__WARMUP_STATUS__": warmup_status,
        "__TEXT_NORMALIZATION_STATUS__": text_normalization_status,
        "__CHECKPOINT__": str(runtime.checkpoint_path),
        "__AUDIO_TOKENIZER__": str(runtime.audio_tokenizer_path),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _build_app(
    runtime: NanoTTSService,
    warmup_manager: WarmupManager,
    text_normalizer_manager: WeTextProcessingManager | None,
    root_path: str | None,
) -> FastAPI:
    app = FastAPI(title="MOSS-TTS-Nano Demo", root_path=root_path or "")
    stream_jobs = StreamingJobManager()
    runtime_manager = RequestRuntimeManager(runtime)

    def _current_demo_entries() -> list[DemoEntry]:
        return _load_demo_entries()

    def _current_demo_entries_by_id() -> dict[str, DemoEntry]:
        return {demo_entry.demo_id: demo_entry for demo_entry in _current_demo_entries()}

    def _resolve_voice_clone_text_chunks(
        *,
        text: str,
        voice_clone_max_text_tokens: int,
        cpu_threads: int,
    ) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []

        try:
            chunks, _, _ = runtime_manager.call_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                callback=lambda selected_runtime: selected_runtime.split_voice_clone_text(
                    text=normalized_text,
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                ),
            )
        except Exception:
            logging.warning("failed to resolve playback text chunks", exc_info=True)
            return [normalized_text]

        normalized_chunks = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
        return normalized_chunks or [normalized_text]

    def _resolve_demo_entry(demo_id: str) -> DemoEntry:
        normalized_demo_id = str(demo_id or "").strip()
        if not normalized_demo_id:
            raise ValueError("demo_id is required.")
        demo_entry = _current_demo_entries_by_id().get(normalized_demo_id)
        if demo_entry is None:
            raise ValueError(f"Unknown demo_id: {normalized_demo_id}")
        return demo_entry

    async def _resolve_prompt_audio_request(
        *,
        demo_id: str,
        prompt_audio: UploadFile | None,
    ) -> tuple[DemoEntry | None, str, str, str | None]:
        normalized_demo_id = str(demo_id or "").strip()
        demo_entry = _resolve_demo_entry(normalized_demo_id) if normalized_demo_id else None

        uploaded_prompt_audio_path, uploaded_prompt_audio_display_path = await _persist_uploaded_prompt_audio(prompt_audio)
        if uploaded_prompt_audio_path is not None and uploaded_prompt_audio_display_path is not None:
            return (
                demo_entry,
                uploaded_prompt_audio_path,
                uploaded_prompt_audio_display_path,
                uploaded_prompt_audio_path,
            )

        if demo_entry is None:
            raise ValueError("demo_id is required unless prompt speech is uploaded.")

        return (
            demo_entry,
            str(demo_entry.prompt_audio_path),
            demo_entry.prompt_audio_relative_path,
            None,
        )

    def _stream_metrics_text(snapshot: dict[str, object]) -> str:
        metrics = [
            f"state={snapshot['state']}",
            f"emitted={float(snapshot['emitted_audio_seconds']):.2f}s",
            f"lead={float(snapshot['lead_seconds']):.2f}s",
        ]
        first_audio_latency = snapshot.get("first_audio_latency_seconds")
        if first_audio_latency is not None:
            metrics.append(f"first_audio={float(first_audio_latency):.2f}s")
        return " | ".join(metrics)

    def _text_normalization_status_text(snapshot: SharedTextNormalizationSnapshot | None) -> str:
        if snapshot is None:
            return "WeTextProcessing disabled."
        if snapshot.failed:
            return f"{snapshot.message} error={snapshot.error}"
        return snapshot.message

    def _resolve_attn_for_runtime(selected_runtime: NanoTTSService, requested_attn: str) -> str:
        normalized = str(requested_attn or "model_default").strip().lower()
        if selected_runtime.device.type != "cpu":
            return requested_attn
        if normalized in {"", "auto", "default", "model_default", "flash_attention_2"}:
            return "eager"
        return requested_attn

    def _put_stream_audio(job: StreamingJob, pcm_bytes: bytes) -> None:
        while True:
            with job.lock:
                if job.is_closed:
                    return
            try:
                job.audio_queue.put(pcm_bytes, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run_streaming_job(
        job: StreamingJob,
        *,
        text: str,
        prompt_audio_path: str,
        prompt_audio_display_path: str,
        prompt_audio_cleanup_path: str | None,
        max_new_frames: int,
        voice_clone_max_text_tokens: int,
        tts_max_batch_size: int,
        codec_max_batch_size: int,
        cpu_threads: int,
        attn_implementation: str,
        do_sample: bool,
        text_temperature: float,
        text_top_p: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        seed: int | None,
    ) -> None:
        try:
            initial_execution_label = "cpu"
            with job.lock:
                job.started_at = time.monotonic()
                job.state = "running"
                job.run_status = f"Streaming realtime audio... exec={initial_execution_label}"

            def _stream_factory(selected_runtime: NanoTTSService):
                return selected_runtime.synthesize_stream(
                    text=text,
                    mode="voice_clone",
                    voice=None,
                    prompt_audio_path=prompt_audio_path,
                    max_new_frames=int(max_new_frames),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    tts_max_batch_size=int(tts_max_batch_size),
                    codec_max_batch_size=int(codec_max_batch_size),
                    attn_implementation=_resolve_attn_for_runtime(selected_runtime, attn_implementation),
                    do_sample=bool(do_sample),
                    text_temperature=float(text_temperature),
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                    audio_repetition_penalty=float(audio_repetition_penalty),
                    seed=seed,
                )

            for event, resolved_execution_device, resolved_cpu_threads in runtime_manager.iter_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                factory=_stream_factory,
            ):
                event_type = str(event.get("type", ""))
                with job.lock:
                    if job.is_closed:
                        break

                if event_type == "audio":
                    waveform_numpy = np.asarray(event["waveform_numpy"], dtype=np.float32)
                    pcm_bytes = _audio_to_pcm16le_bytes(waveform_numpy)
                    if not pcm_bytes:
                        continue
                    sample_rate = int(event["sample_rate"])
                    channels = 1 if waveform_numpy.ndim == 1 else int(waveform_numpy.shape[1])
                    is_pause = bool(event.get("is_pause", False))
                    event_duration_seconds = (
                        float(waveform_numpy.shape[0]) / float(sample_rate)
                        if sample_rate > 0 and waveform_numpy.ndim >= 1
                        else 0.0
                    )
                    with job.lock:
                        job.sample_rate = sample_rate
                        job.channels = channels
                        job.emitted_audio_seconds = float(event.get("emitted_audio_seconds", 0.0))
                        job.lead_seconds = float(event.get("lead_seconds", 0.0))
                        normalized_chunk_index, job.chunk_index_base = _normalize_stream_chunk_index(
                            event.get("chunk_index"),
                            chunk_count=len(job.text_chunks),
                            current_base=job.chunk_index_base,
                        )
                        if normalized_chunk_index is not None:
                            job.current_chunk_index = normalized_chunk_index
                            if not is_pause and event_duration_seconds > 0.0:
                                chunk_end_seconds = job.emitted_audio_seconds
                                chunk_start_seconds = max(0.0, chunk_end_seconds - event_duration_seconds)
                                job.audio_chunk_ranges.append(
                                    (chunk_start_seconds, chunk_end_seconds, normalized_chunk_index)
                                )
                        if job.first_audio_at is None and not is_pause:
                            job.first_audio_at = time.monotonic()
                        job.run_status = (
                            f"Streaming | emitted={job.emitted_audio_seconds:.2f}s | lead={job.lead_seconds:.2f}s"
                        )
                    _put_stream_audio(job, pcm_bytes)
                    continue

                if event_type == "result":
                    formatted_result = dict(event)
                    formatted_result["execution_device"] = resolved_execution_device
                    formatted_result["prompt_audio_display_path"] = prompt_audio_display_path
                    if resolved_cpu_threads is not None:
                        formatted_result["cpu_threads"] = resolved_cpu_threads
                    formatted_run_status = _format_run_status(formatted_result)
                    with job.lock:
                        job.final_result = {
                            "audio_path": event.get("audio_path"),
                            "prompt_audio_path": prompt_audio_display_path,
                            "run_status": formatted_run_status,
                            "text_chunks": list(job.text_chunks),
                        }
                        job.prompt_audio_path = prompt_audio_display_path
                        job.state = "done"
                        job.completed_at = time.monotonic()
                        job.run_status = formatted_run_status
        except Exception as exc:
            logging.exception("Nano-TTS realtime streaming job failed")
            with job.lock:
                job.state = "failed"
                job.error = str(exc)
                job.completed_at = time.monotonic()
                job.run_status = f"Stream failed: {exc}"
        finally:
            _maybe_delete_file(prompt_audio_cleanup_path)
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return HTMLResponse(
            _render_index_html(
                request=request,
                runtime=runtime,
                demo_entries=_current_demo_entries(),
                warmup_status=_warmup_status_text(warmup_manager.snapshot()),
                text_normalization_status=_text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
            )
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "cpu_runtime_loaded": runtime_manager.is_cpu_runtime_loaded(),
            "default_cpu_threads": runtime_manager.default_cpu_threads,
            "attn_implementation": runtime.attn_implementation or "model_default",
            "checkpoint_default_attn_implementation": runtime._checkpoint_global_attn_implementation or "unknown",
            "checkpoint_default_local_attn_implementation": runtime._checkpoint_local_attn_implementation or "unknown",
            "configured_attn_implementation": runtime._configured_global_attn_implementation or "unknown",
            "configured_local_attn_implementation": runtime._configured_local_attn_implementation or "unknown",
            "checkpoint_path": str(runtime.checkpoint_path),
            "audio_tokenizer_path": str(runtime.audio_tokenizer_path),
            "text_normalization_status": _text_normalization_status_text(
                text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
            ),
        }

    @app.get("/api/warmup-status")
    async def warmup_status():
        snapshot = warmup_manager.snapshot()
        return {
            "state": snapshot.state,
            "progress": snapshot.progress,
            "message": snapshot.message,
            "error": snapshot.error,
            "ready": snapshot.ready,
            "failed": snapshot.failed,
            "status_text": _warmup_status_text(snapshot),
        }

    @app.get("/api/text-normalization-status")
    async def text_normalization_status():
        snapshot = text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
        if snapshot is None:
            return {
                "state": "disabled",
                "message": "WeTextProcessing disabled.",
                "error": None,
                "ready": False,
                "failed": False,
                "available": False,
                "status_text": "WeTextProcessing disabled.",
            }
        return {
            "state": snapshot.state,
            "message": snapshot.message,
            "error": snapshot.error,
            "ready": snapshot.ready,
            "failed": snapshot.failed,
            "available": snapshot.available,
            "status_text": _text_normalization_status_text(snapshot),
        }

    @app.get("/api/demo-prompt-audio/{demo_id}")
    async def demo_prompt_audio(demo_id: str):
        try:
            demo_entry = _resolve_demo_entry(demo_id)
        except ValueError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})

        media_type = "audio/wav" if demo_entry.prompt_audio_path.suffix.lower() == ".wav" else "application/octet-stream"
        return FileResponse(
            path=str(demo_entry.prompt_audio_path),
            media_type=media_type,
            filename=demo_entry.prompt_audio_path.name,
        )

    @app.get("/api/voices")
    async def list_voices():
        voices = [_demo_entry_to_payload(demo_entry) for demo_entry in _current_demo_entries()]
        return {
            "voices": voices,
            "total": len(voices),
            "custom_count": sum(1 for voice in voices if bool(voice.get("is_custom"))),
        }

    @app.post("/api/voices")
    async def create_voice(
        title: str = Form(...),
        language: str = Form("zh"),
        prompt_audio: UploadFile = File(...),
        text: str = Form(""),
    ):
        try:
            normalized_title = _sanitize_custom_voice_title(title)
            normalized_language = _normalize_custom_voice_language(language)
            voice_id = f"custom-{uuid.uuid4().hex[:12]}"
            now = _format_custom_voice_timestamp()
            audio_relative_path = await _persist_custom_voice_audio(prompt_audio, voice_id)
            record: dict[str, object] = {
                "id": voice_id,
                "title": normalized_title,
                "name": normalized_title,
                "language": normalized_language,
                "audio_path": audio_relative_path,
                "text": str(text or ""),
                "created_at": now,
                "updated_at": now,
            }
            with CUSTOM_VOICE_LOCK:
                records = _read_custom_voice_records_unlocked()
                records.append(record)
                _write_custom_voice_records_unlocked(records)
            entry = _custom_voice_record_to_entry(record)
            if entry is None:
                raise RuntimeError("created custom voice is invalid.")
            return {"voice": _demo_entry_to_payload(entry)}
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        except Exception as exc:
            logging.exception("failed to create custom voice")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.put("/api/voices/{voice_id}")
    async def update_voice(
        voice_id: str,
        title: str = Form(...),
        language: str = Form("zh"),
        prompt_audio: UploadFile | None = File(None),
        text: str = Form(""),
    ):
        normalized_voice_id = str(voice_id or "").strip()
        try:
            existing_entry = _resolve_demo_entry(normalized_voice_id)
            if not existing_entry.is_custom:
                return JSONResponse(status_code=403, content={"error": "builtin voices cannot be modified."})
        except ValueError:
            return JSONResponse(status_code=404, content={"error": "custom voice not found."})

        try:
            normalized_title = _sanitize_custom_voice_title(title)
            normalized_language = _normalize_custom_voice_language(language)
            old_audio_path_to_delete: Path | None = None
            with CUSTOM_VOICE_LOCK:
                records = _read_custom_voice_records_unlocked()
                record_index = next(
                    (index for index, record in enumerate(records) if str(record.get("id")) == normalized_voice_id),
                    None,
                )
                if record_index is None:
                    return JSONResponse(status_code=404, content={"error": "custom voice not found."})
                record = dict(records[record_index])

            audio_relative_path = str(record.get("audio_path") or "")
            if prompt_audio is not None and str(prompt_audio.filename or "").strip():
                old_audio_path_to_delete = _resolve_custom_voice_audio_path(audio_relative_path)
                audio_relative_path = await _persist_custom_voice_audio(prompt_audio, normalized_voice_id)

            now = _format_custom_voice_timestamp()
            record.update(
                {
                    "title": normalized_title,
                    "name": normalized_title,
                    "language": normalized_language,
                    "audio_path": audio_relative_path,
                    "text": str(text or ""),
                    "updated_at": now,
                    "created_at": str(record.get("created_at") or now),
                }
            )
            with CUSTOM_VOICE_LOCK:
                records = _read_custom_voice_records_unlocked()
                record_index = next(
                    (index for index, item in enumerate(records) if str(item.get("id")) == normalized_voice_id),
                    None,
                )
                if record_index is None:
                    return JSONResponse(status_code=404, content={"error": "custom voice not found."})
                records[record_index] = record
                _write_custom_voice_records_unlocked(records)

            if old_audio_path_to_delete is not None and old_audio_path_to_delete != _resolve_custom_voice_audio_path(audio_relative_path):
                _maybe_delete_file(str(old_audio_path_to_delete))

            entry = _custom_voice_record_to_entry(record)
            if entry is None:
                raise RuntimeError("updated custom voice is invalid.")
            return {"voice": _demo_entry_to_payload(entry)}
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        except Exception as exc:
            logging.exception("failed to update custom voice")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.delete("/api/voices/{voice_id}")
    async def delete_voice(voice_id: str):
        normalized_voice_id = str(voice_id or "").strip()
        try:
            existing_entry = _resolve_demo_entry(normalized_voice_id)
            if not existing_entry.is_custom:
                return JSONResponse(status_code=403, content={"error": "builtin voices cannot be deleted."})
        except ValueError:
            return JSONResponse(status_code=404, content={"error": "custom voice not found."})

        with CUSTOM_VOICE_LOCK:
            records = _read_custom_voice_records_unlocked()
            record_index = next(
                (index for index, record in enumerate(records) if str(record.get("id")) == normalized_voice_id),
                None,
            )
            if record_index is None:
                return JSONResponse(status_code=404, content={"error": "custom voice not found."})
            record = records.pop(record_index)
            _write_custom_voice_records_unlocked(records)

        audio_path = _resolve_custom_voice_audio_path(str(record.get("audio_path") or ""))
        _maybe_delete_file(str(audio_path) if audio_path is not None else None)
        return {
            "deleted": True,
            "voice_id": normalized_voice_id,
            "voices": [_demo_entry_to_payload(demo_entry) for demo_entry in _current_demo_entries()],
        }

    @app.post("/api/generate-stream/start")
    async def generate_stream_start(
        text: str = Form(...),
        demo_id: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
        max_new_frames: int = Form(375),
        voice_clone_max_text_tokens: int = Form(75),
        tts_max_batch_size: int = Form(0),
        codec_max_batch_size: int = Form(0),
        enable_text_normalization: str = Form("1"),
        enable_normalize_tts_text: str = Form("1"),
        cpu_threads: int = Form(0),
        attn_implementation: str = Form("model_default"),
        do_sample: str = Form("1"),
        text_temperature: float = Form(1.0),
        text_top_p: float = Form(1.0),
        text_top_k: int = Form(50),
        audio_temperature: float = Form(0.8),
        audio_top_p: float = Form(0.95),
        audio_top_k: int = Form(25),
        audio_repetition_penalty: float = Form(1.2),
        seed: str = Form("0"),
    ):
        try:
            demo_entry, prompt_audio_path, prompt_audio_display_path, prompt_audio_cleanup_path = (
                await _resolve_prompt_audio_request(demo_id=demo_id, prompt_audio=prompt_audio)
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        resolved_text = str(text or "").strip() or (demo_entry.text if demo_entry is not None else "")
        if not resolved_text:
            _maybe_delete_file(prompt_audio_cleanup_path)
            return JSONResponse(status_code=400, content={"error": "text is required."})

        try:
            prepared_texts = shared_prepare_tts_request_texts(
                text=resolved_text,
                enable_wetext=_coerce_bool(enable_text_normalization, False),
                enable_normalize_tts_text=_coerce_bool(enable_normalize_tts_text, True),
                text_normalizer_manager=text_normalizer_manager,
            )
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise
        warmup_snapshot = warmup_manager.snapshot()
        if not warmup_snapshot.ready:
            warmup_snapshot = warmup_manager.ensure_ready()
            if not warmup_snapshot.ready:
                _maybe_delete_file(prompt_audio_cleanup_path)
                return JSONResponse(
                    status_code=500,
                    content={"error": _warmup_status_text(warmup_snapshot)},
                )

        try:
            normalized_seed = None if seed in {"", "0"} else int(seed)
            text_chunks = _resolve_voice_clone_text_chunks(
                text=str(prepared_texts["text"]),
                voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                cpu_threads=int(cpu_threads),
            )
            job = stream_jobs.create()
            with job.lock:
                job.prompt_audio_path = prompt_audio_display_path
                job.text_chunks = list(text_chunks)
            thread = threading.Thread(
                target=_run_streaming_job,
                kwargs={
                    "job": job,
                    "text": str(prepared_texts["text"]),
                    "prompt_audio_path": prompt_audio_path,
                    "prompt_audio_display_path": prompt_audio_display_path,
                    "prompt_audio_cleanup_path": prompt_audio_cleanup_path,
                    "max_new_frames": int(max_new_frames),
                    "voice_clone_max_text_tokens": int(voice_clone_max_text_tokens),
                    "tts_max_batch_size": int(tts_max_batch_size),
                    "codec_max_batch_size": int(codec_max_batch_size),
                    "cpu_threads": int(cpu_threads),
                    "attn_implementation": attn_implementation,
                    "do_sample": _coerce_bool(do_sample, True),
                    "text_temperature": float(text_temperature),
                    "text_top_p": float(text_top_p),
                    "text_top_k": int(text_top_k),
                    "audio_temperature": float(audio_temperature),
                    "audio_top_p": float(audio_top_p),
                    "audio_top_k": int(audio_top_k),
                    "audio_repetition_penalty": float(audio_repetition_penalty),
                    "seed": normalized_seed,
                },
                name=f"nano-tts-stream-{job.stream_id}",
                daemon=True,
            )
            thread.start()
            prompt_audio_cleanup_path = None

            initial_execution_label = "cpu"

            return {
                "stream_id": job.stream_id,
                "audio_url": f"{app.root_path}/api/generate-stream/{job.stream_id}/audio",
                "status_url": f"{app.root_path}/api/generate-stream/{job.stream_id}/status",
                "result_url": f"{app.root_path}/api/generate-stream/{job.stream_id}/result",
                "sample_rate": job.sample_rate,
                "channels": job.channels,
                "run_status": f"Streaming realtime audio... exec={initial_execution_label}",
                "prompt_audio_path": prompt_audio_display_path,
                "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
                "text_normalization_status_text": _text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
                "text_chunks": text_chunks,
                "normalized_text": str(prepared_texts["normalized_text"]),
                "normalization_method": str(prepared_texts["normalization_method"]),
                "text_normalization_language": str(prepared_texts["text_normalization_language"]),
            }
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise

    @app.get("/api/generate-stream/{stream_id}/status")
    async def generate_stream_status(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        snapshot = job.snapshot()
        snapshot["status_text"] = _format_stream_status(snapshot)
        snapshot["stream_metrics"] = _stream_metrics_text(snapshot)
        return snapshot

    @app.get("/api/generate-stream/{stream_id}/audio")
    async def generate_stream_audio(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})

        def _iter_audio():
            while True:
                item = job.audio_queue.get()
                if item is None:
                    break
                yield item

        return StreamingResponse(
            _iter_audio(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Codec": "pcm_s16le",
                "X-Audio-Sample-Rate": str(job.sample_rate),
                "X-Audio-Channels": str(job.channels),
                "X-Stream-Id": stream_id,
            },
        )

    @app.get("/api/generate-stream/{stream_id}/result")
    async def generate_stream_result(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        snapshot = job.snapshot()
        if snapshot["failed"]:
            return JSONResponse(status_code=500, content={"error": snapshot["error"], **snapshot})
        if not snapshot["ready"] or job.final_result is None:
            return JSONResponse(status_code=202, content=snapshot)

        result = dict(job.final_result)
        audio_chunk_ranges: list[list[float | int]] = []
        with job.lock:
            audio_chunk_ranges = [
                [float(start_seconds), float(end_seconds), int(chunk_index)]
                for start_seconds, end_seconds, chunk_index in job.audio_chunk_ranges
            ]
        audio_base64_payload = str(result.get("audio_base64") or "")
        audio_path_for_response = str(result.get("audio_path") or "").strip()
        if not audio_base64_payload and audio_path_for_response:
            audio_base64_payload = _read_audio_file_base64(audio_path_for_response)
            if audio_base64_payload:
                with job.lock:
                    if job.final_result is not None:
                        job.final_result["audio_base64"] = audio_base64_payload
                        job.final_result["audio_path"] = ""
                _maybe_delete_file(audio_path_for_response)

        return {
            "stream_id": stream_id,
            "ready": True,
            "state": snapshot["state"],
            "prompt_audio_path": result.get("prompt_audio_path") or snapshot.get("prompt_audio_path") or "",
            "run_status": result.get("run_status") or snapshot["run_status"],
            "stream_metrics": _stream_metrics_text(snapshot),
            "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
            "text_chunks": result.get("text_chunks") or snapshot.get("text_chunks") or [],
            "audio_chunk_ranges": audio_chunk_ranges,
            "audio_base64": audio_base64_payload,
        }

    @app.post("/api/generate-stream/{stream_id}/close")
    async def generate_stream_close(stream_id: str):
        job = stream_jobs.close(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        audio_cleanup_path = ""
        with job.lock:
            if job.final_result is not None:
                audio_cleanup_path = str(job.final_result.get("audio_path") or "").strip()
        snapshot = job.snapshot()
        snapshot["status_text"] = _format_stream_status(snapshot)
        stream_jobs.delete(stream_id)
        _maybe_delete_file(audio_cleanup_path)
        return snapshot

    @app.post("/api/generate")
    async def generate(
        text: str = Form(...),
        demo_id: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
        max_new_frames: int = Form(375),
        voice_clone_max_text_tokens: int = Form(75),
        tts_max_batch_size: int = Form(0),
        codec_max_batch_size: int = Form(0),
        enable_text_normalization: str = Form("1"),
        enable_normalize_tts_text: str = Form("1"),
        cpu_threads: int = Form(0),
        attn_implementation: str = Form("model_default"),
        do_sample: str = Form("1"),
        text_temperature: float = Form(1.0),
        text_top_p: float = Form(1.0),
        text_top_k: int = Form(50),
        audio_temperature: float = Form(0.8),
        audio_top_p: float = Form(0.95),
        audio_top_k: int = Form(25),
        audio_repetition_penalty: float = Form(1.2),
        seed: str = Form("0"),
    ):
        try:
            demo_entry, prompt_audio_path, prompt_audio_display_path, prompt_audio_cleanup_path = (
                await _resolve_prompt_audio_request(demo_id=demo_id, prompt_audio=prompt_audio)
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        resolved_text = str(text or "").strip() or (demo_entry.text if demo_entry is not None else "")
        if not resolved_text:
            _maybe_delete_file(prompt_audio_cleanup_path)
            return JSONResponse(status_code=400, content={"error": "text is required."})

        try:
            prepared_texts = shared_prepare_tts_request_texts(
                text=resolved_text,
                enable_wetext=_coerce_bool(enable_text_normalization, False),
                enable_normalize_tts_text=_coerce_bool(enable_normalize_tts_text, True),
                text_normalizer_manager=text_normalizer_manager,
            )
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise
        warmup_snapshot = warmup_manager.snapshot()
        if not warmup_snapshot.ready:
            warmup_snapshot = warmup_manager.ensure_ready()
            if not warmup_snapshot.ready:
                _maybe_delete_file(prompt_audio_cleanup_path)
                return JSONResponse(
                    status_code=500,
                    content={"error": _warmup_status_text(warmup_snapshot)},
                )

        generated_audio_path: str | None = None
        try:
            normalized_seed = None if seed in {"", "0"} else int(seed)

            def _synthesize(selected_runtime: NanoTTSService):
                return selected_runtime.synthesize(
                    text=str(prepared_texts["text"]),
                    mode="voice_clone",
                    voice=None,
                    prompt_audio_path=prompt_audio_path,
                    max_new_frames=int(max_new_frames),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    tts_max_batch_size=int(tts_max_batch_size),
                    codec_max_batch_size=int(codec_max_batch_size),
                    attn_implementation=_resolve_attn_for_runtime(selected_runtime, attn_implementation),
                    do_sample=_coerce_bool(do_sample, True),
                    text_temperature=float(text_temperature),
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                    audio_repetition_penalty=float(audio_repetition_penalty),
                    seed=normalized_seed,
                )

            result, resolved_execution_device, resolved_cpu_threads = runtime_manager.call_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                callback=_synthesize,
            )
            result["execution_device"] = resolved_execution_device
            result["prompt_audio_display_path"] = prompt_audio_display_path
            if resolved_cpu_threads is not None:
                result["cpu_threads"] = resolved_cpu_threads
            text_chunks = [
                str(chunk).strip()
                for chunk in (result.get("voice_clone_text_chunks") or [])
                if str(chunk).strip()
            ]
            if not text_chunks:
                text_chunks = _resolve_voice_clone_text_chunks(
                    text=str(prepared_texts["text"]),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    cpu_threads=int(cpu_threads),
                )
            generated_audio_path = str(result["audio_path"])
            wav_bytes = _audio_to_wav_bytes(result["waveform_numpy"], int(result["sample_rate"]))
            return {
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "sample_rate": int(result["sample_rate"]),
                "run_status": _format_run_status(result),
                "prompt_audio_path": prompt_audio_display_path,
                "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
                "text_normalization_status_text": _text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
                "text_chunks": text_chunks,
                "normalized_text": str(prepared_texts["normalized_text"]),
                "normalization_method": str(prepared_texts["normalization_method"]),
                "text_normalization_language": str(prepared_texts["text_normalization_language"]),
            }
        except Exception as exc:
            logging.exception("Nano-TTS generation failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})
        finally:
            _maybe_delete_file(generated_audio_path)
            _maybe_delete_file(prompt_audio_cleanup_path)

    return app


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano web demo")
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        dest="checkpoint_path",
        type=str,
        default=_env_default("MOSS_TTS_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH),
    )
    parser.add_argument(
        "--audio-tokenizer-path",
        "--audio_tokenizer_path",
        dest="audio_tokenizer_path",
        type=str,
        default=_env_default("MOSS_TTS_AUDIO_TOKENIZER_PATH", DEFAULT_AUDIO_TOKENIZER_PATH),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=_env_default("MOSS_TTS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "auto"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--attn-implementation",
        "--attn_implementation",
        dest="attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager"],
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    resolved_runtime_device = "cpu"
    if args.device != "cpu":
        logging.info("CPU-only app mode: ignoring --device=%s and forcing cpu.", args.device)

    runtime = NanoTTSService(
        checkpoint_path=args.checkpoint_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        device=resolved_runtime_device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        output_dir=args.output_dir,
    )
    text_normalizer_manager = SharedWeTextProcessingManager()
    text_normalizer_manager.start()
    warmup_manager = WarmupManager(runtime, text_normalizer_manager=text_normalizer_manager)
    warmup_manager.start()

    vscode_proxy_uri = os.getenv("VSCODE_PROXY_URI", "")
    root_path = _resolve_vscode_root_path(vscode_proxy_uri, args.port)
    logging.info("root_path=%s", root_path)
    if args.share:
        logging.warning("--share is ignored by the FastAPI-based Nano-TTS app.")

    app = _build_app(runtime, warmup_manager, text_normalizer_manager, root_path)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        root_path=root_path or "",
    )


if __name__ == "__main__":
    main()
