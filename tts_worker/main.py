#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS Worker API Service

A lightweight FastAPI service that orchestrates book TTS processing.
Calls the TTS API at :6006 for audio generation instead of loading models directly.

Usage: python -m tts_worker.main
"""
import os
import sys
import asyncio
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

# Add project root to sys.path so utils.* imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Load .env from project root
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from tts_worker.processor import TaskState, process_book

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXTERNAL_API_HOST = os.getenv("EXTERNAL_API_HOST", "http://localhost:8000")
AUTODL_API_KEY = os.getenv("AUTODL_API_KEY", "autodl-tts-secret-key-2024")
TTS_API_URL = os.getenv("TTS_API_URL", "http://127.0.0.1:6006")
WORKER_HOST = os.getenv("WORKER_HOST", "0.0.0.0")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8080"))

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class BookTTSRequest(BaseModel):
    book_id: str
    lang: str = Field("zh", pattern="^(zh|en)$")
    force: bool = False
    modes: Optional[str] = Field(None, description="Comma-separated modes, e.g. 'summarize_20,guide'")
    spk: Optional[str] = Field(None, description="Speaker name for TTS API")


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    book_id: str
    progress: str
    chapters_completed: int
    chapters_total: int
    chapters_failed: int
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TTS Worker")

# In-memory task store
tasks: dict[str, TaskState] = {}
_current_task: asyncio.Task | None = None


async def verify_api_key(x_autodl_api_key: str = Header(...)):
    if x_autodl_api_key != AUTODL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_autodl_api_key


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process", response_model=TaskResponse)
async def start_process(
    request: BookTTSRequest,
    api_key: str = Depends(verify_api_key),
):
    global _current_task

    # Serial execution: reject if a task is already running
    if _current_task is not None and not _current_task.done():
        raise HTTPException(
            status_code=409,
            detail="A task is already in progress",
        )

    task_id = str(uuid.uuid4())
    task = TaskState(task_id=task_id, book_id=request.book_id)
    tasks[task_id] = task

    _current_task = asyncio.create_task(
        process_book(
            task,
            lang=request.lang,
            force=request.force,
            modes=request.modes,
            spk=request.spk,
            tts_api_url=TTS_API_URL,
            external_api_host=EXTERNAL_API_HOST,
            api_key=AUTODL_API_KEY,
        )
    )

    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Task created",
    )


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    api_key: str = Depends(verify_api_key),
):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        book_id=task.book_id,
        progress=task.progress,
        chapters_completed=task.chapters_completed,
        chapters_total=task.chapters_total,
        chapters_failed=task.chapters_failed,
        error=task.error,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT)
