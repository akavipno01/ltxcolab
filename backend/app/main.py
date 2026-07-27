from __future__ import annotations

import asyncio
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import APP_NAME, OUTPUTS_DIR, ensure_directories
from .database import db, initialize_database
from .model_runtime import (
    get_model,
    inference_executor,
    remove_model,
    runtime_state,
    start_model_download,
    _update_state,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_directories()
    initialize_database()
    yield


app = FastAPI(title=APP_NAME, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "file://"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Video-Id", "X-Seed", "X-Generation-Time", "X-Video-Format"],
)


def _row(row):
    return dict(row) if row else None


@app.get("/health")
def health():
    return {"ok": True, "name": APP_NAME, "runtime": runtime_state()}


@app.get("/runtime")
def model_status():
    return runtime_state()


@app.post("/model/download", status_code=202)
def download_model():
    return start_model_download()


@app.delete("/model")
def delete_model():
    try:
        return remove_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/history")
def history():
    with db() as connection:
        rows = connection.execute(
            "SELECT * FROM generations ORDER BY created_at DESC"
        ).fetchall()
    return [_row(row) for row in rows]


def _run_generation(model, *, prompt: str, num_inference_steps: int, guidance_scale: float, seed: int):
    import torch
    from diffusers.utils import export_to_video
    
    generator = torch.Generator(device=model.device).manual_seed(seed)
    
    video = model(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np"
    ).frames[0]
    
    return video


@app.post("/generate")
async def generate(
    text: Annotated[str, Form()],
    num_inference_steps: Annotated[int, Form(ge=1, le=100)] = 50,
    guidance_scale: Annotated[float, Form(ge=1.0, le=20.0)] = 3.0,
    seed: Annotated[int | None, Form(ge=0, le=2147483647)] = None,
):
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=422, detail="Vui lòng nhập văn bản để tạo video.")
        
    used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    started = time.perf_counter()
    
    try:
        model = await get_model()
        loop = asyncio.get_running_loop()
        
        _update_state(
            status="generating",
            progress=0.0,
            detail="Đang sinh video từ văn bản...",
        )
        
        video_frames = await loop.run_in_executor(
            inference_executor(),
            lambda: _run_generation(
                model,
                prompt=clean_text,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=used_seed,
            ),
        )
        
        _update_state(
            status="ready",
            progress=100.0,
            detail="Mô hình đã sẵn sàng.",
        )
        
        from diffusers.utils import export_to_video
        
        generation_time = round(time.perf_counter() - started, 2)
        video_id = uuid.uuid4().hex[:12]
        filename = f"{video_id}.mp4"
        output_path = OUTPUTS_DIR / filename
        
        # Export video and save to disk
        export_to_video(video_frames, str(output_path), fps=8)
        
        now = time.time()
        with db() as connection:
            connection.execute(
                """
                INSERT INTO generations
                (id, text, video_path, generation_time, seed, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id, clean_text, filename, generation_time,
                    used_seed, now,
                ),
            )
            
        return FileResponse(
            output_path,
            media_type="video/mp4",
            headers={
                "X-Video-Id": video_id,
                "X-Seed": str(used_seed),
                "X-Generation-Time": str(generation_time),
                "X-Video-Format": "mp4",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        detail = f"Không thể tạo video: {exc}"
        raise HTTPException(status_code=500, detail=detail) from exc
