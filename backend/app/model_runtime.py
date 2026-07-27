from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import MODEL_DIR, MODEL_ID, ensure_directories

_model: Any | None = None
_model_lock = asyncio.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltx-video")
_state_lock = threading.Lock()
_download_lock = threading.Lock()
_download_future = None
_marker_path = MODEL_DIR / ".ltx-video-model.json"


def _model_is_installed() -> bool:
    if not _marker_path.is_file():
        return False
    try:
        marker = json.loads(_marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return marker.get("model_id") == MODEL_ID and MODEL_DIR.exists()


_installed_at_import = _model_is_installed()
_state = {
    "status": "installed" if _installed_at_import else "not_installed",
    "detail": (
        "Mô hình đã được tải và sẵn sàng sử dụng."
        if _installed_at_import
        else "Cần tải mô hình LTX-Video trước khi sử dụng."
    ),
    "device": None,
    "installed": _installed_at_import,
    "progress": 100.0 if _installed_at_import else 0.0,
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "current_file": "",
}


def runtime_state() -> dict:
    installed = _model_is_installed()
    with _state_lock:
        state = {**_state, "model_id": MODEL_ID, "model_path": str(MODEL_DIR)}
        state["installed"] = installed
        if installed and state["status"] == "not_installed":
            state.update(
                status="installed",
                detail="Mô hình đã được tải và sẵn sàng sử dụng.",
                progress=100.0,
            )
        return state


def _update_state(**values) -> None:
    with _state_lock:
        _state.update(values)


def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DownloadProgress:
    """Tqdm adapter for huggingface_hub."""

    _lock = threading.RLock()

    @classmethod
    def get_lock(cls):
        if not hasattr(cls, "_lock"):
            cls._lock = threading.RLock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def __init__(self, iterable=None, *args, total=None, desc=None, initial=0, **kwargs):
        self.iterable = iterable
        self.total = int(total or 0)
        self.n = int(initial or 0)
        self.desc = str(desc or "")
        _update_state(
            current_file=Path(self.desc).name,
            total_bytes=self.total,
            downloaded_bytes=self.n,
        )

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return self._iterate()

    def _iterate(self):
        for item in self.iterable:
            yield item

    def update(self, amount=1):
        self.n += int(amount or 0)
        progress = min(99.0, self.n * 100 / self.total) if self.total else 0.0
        _update_state(
            progress=round(progress, 1),
            downloaded_bytes=self.n,
            total_bytes=self.total,
            current_file=Path(self.desc).name,
            detail=f"Đang tải dữ liệu mô hình… {progress:.1f}%",
        )

    def set_description(self, desc=None, refresh=True):
        self.desc = str(desc or "")
        _update_state(current_file=Path(self.desc).name)

    def set_description_str(self, desc=None, refresh=True):
        self.set_description(desc, refresh)

    def set_postfix(self, *args, **kwargs):
        return None

    def refresh(self, *args, **kwargs):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _download_model_sync() -> None:
    import gdown

    ensure_directories()
    with _download_lock:
        if _model_is_installed():
            _update_state(
                status="installed", installed=True, progress=100.0,
                detail="Mô hình đã có sẵn trên thiết bị.",
            )
            return
        _update_state(
            status="downloading", installed=False, progress=0.0,
            downloaded_bytes=0, total_bytes=0, current_file="",
            detail="Đang kết nối để tải mô hình video từ Google Drive…",
        )
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            output_path = MODEL_DIR / "ltxv-2b-0.9.8-distilled-fp8.safetensors"
            gdown.download(
                id="1eyon1qAQJ-kbqH17bz4uIn50dWyhZ-FB",
                output=str(output_path),
                quiet=False
            )
            _marker_path.write_text(
                json.dumps({"model_id": MODEL_ID}, ensure_ascii=False),
                encoding="utf-8",
            )
            _update_state(
                status="installed", installed=True, progress=100.0,
                detail="Tải mô hình hoàn tất. Ứng dụng đã sẵn sàng.",
                current_file="",
            )
        except Exception as exc:
            _update_state(
                status="error", installed=False,
                detail=f"Không thể tải mô hình: {exc}",
            )
            raise


def start_model_download() -> dict:
    global _download_future
    if _model_is_installed():
        _update_state(
            status="installed", installed=True, progress=100.0,
            detail="Mô hình đã có sẵn trên thiết bị.",
        )
        return runtime_state()
    if _download_future is None or _download_future.done():
        _download_future = _executor.submit(_download_model_sync)
    return runtime_state()


def remove_model() -> dict:
    global _model
    if _download_future is not None and not _download_future.done():
        raise RuntimeError("Không thể xóa mô hình khi đang tải.")
    _model = None
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    _update_state(
        status="not_installed", installed=False, progress=0.0,
        detail="Cần tải mô hình video trước khi sử dụng.",
        device=None, current_file="", downloaded_bytes=0, total_bytes=0,
    )
    return runtime_state()


def _load_model():
    if not _model_is_installed():
        raise RuntimeError("Mô hình chưa được tải. Hãy hoàn tất bước cài đặt mô hình.")
    import torch
    from diffusers import LTXPipeline

    device = _best_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    _update_state(
        status="loading", detail=f"Đang nạp mô hình vào {device.upper()}…",
        device=device, installed=True,
    )
    try:
        single_file_path = MODEL_DIR / "ltxv-2b-0.9.8-distilled-fp8.safetensors"
        if single_file_path.exists():
            from transformers import T5EncoderModel
            _update_state(status="loading", detail=f"Đang tải Text Encoder...", device=device)
            text_encoder = T5EncoderModel.from_pretrained(
                "Lightricks/LTX-Video",
                subfolder="text_encoder",
                torch_dtype=dtype,
            )
            _update_state(status="loading", detail=f"Đang nạp Pipeline...", device=device)
            model = LTXPipeline.from_single_file(
                str(single_file_path),
                torch_dtype=dtype,
                text_encoder=text_encoder,
            )
        else:
            model = LTXPipeline.from_pretrained(
                str(MODEL_DIR),
                torch_dtype=dtype,
                local_files_only=True,
            )
        if device == "cuda":
            model.enable_model_cpu_offload() # Helps with memory on GPUs
    except Exception as exc:
        _update_state(status="error", detail=f"Không thể nạp mô hình: {exc}", device=device)
        raise
    _update_state(
        status="ready", detail=f"Mô hình đang hoạt động trên {device.upper()}.",
        device=device, installed=True, progress=100.0,
    )
    return model


async def get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is None:
            loop = asyncio.get_running_loop()
            _model = await loop.run_in_executor(_executor, _load_model)
    return _model


def inference_executor() -> ThreadPoolExecutor:
    return _executor
