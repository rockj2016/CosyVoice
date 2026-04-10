#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS API Service

Usage: python api.py --spk cn_male_1
"""
import os
import sys
import glob
import argparse
import uvicorn
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Generator, Optional
import asyncio
import uuid
from dotenv import load_dotenv
from starlette.background import BackgroundTask
from utils.normalize import smartread_text_normalize

# Add third_party path
sys.path.append('third_party/Matcha-TTS')

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='CosyVoice TTS API Service')
parser.add_argument('--spk', type=str, required=True, help='Default speaker name (filename without .pt in spk/ folder)')
parser.add_argument('--host', type=str, default=None, help='API host (overrides API_HOST env var)')
parser.add_argument('--port', type=int, default=None, help='API port (overrides API_PORT env var)')
args = parser.parse_args()

# Configuration from .env + CLI args
AUTODL_API_KEY = os.getenv('AUTODL_API_KEY', 'autodl-tts-secret-key-2024')
SPK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spk')
DEFAULT_SPK = args.spk
MODEL_DIR = os.getenv('MODEL_DIR', 'pretrained_models/Fun-CosyVoice3-0.5B')
GPU_CONCURRENT = int(os.getenv('GPU_CONCURRENT', '1'))
HOST = args.host or os.getenv('API_HOST', '0.0.0.0')
PORT = args.port or int(os.getenv('API_PORT', 6006))

# CosyVoice imports
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
try:
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
except ValueError:
    pass # Already registered

from cosyvoice.cli.cosyvoice import AutoModel

# Global CosyVoice instance
_cosyvoice_instance = None
_default_spk_name = None

app = FastAPI(title="CosyVoice TTS API")

class TTSRequest(BaseModel):
    text: str = Field(..., max_length=100, description="Text to synthesize (max 100 chars)")
    stream: bool = Field(False, description="If true, stream audio chunks as they are generated")
    spk: Optional[str] = Field(None, description="Speaker name. If omitted, uses the default speaker.")

def init_cosyvoice():
    """Initialize CosyVoice model and load speaker info (singleton)"""
    global _cosyvoice_instance, _default_spk_name

    if _cosyvoice_instance is not None:
        return _cosyvoice_instance, _default_spk_name

    print(f"Initializing CosyVoice model from {MODEL_DIR}...")
    _cosyvoice_instance = AutoModel(
        model_dir=MODEL_DIR,
        load_trt=True,
        load_vllm=True,
        fp16=False,
        gpu_concurrent=GPU_CONCURRENT,
    )

    # Load all .pt files from spk/ directory
    spk_files = sorted(glob.glob(os.path.join(SPK_DIR, '*.pt')))
    if not spk_files:
        raise RuntimeError(f"No .pt files found in {SPK_DIR}")

    for spk_path in spk_files:
        spk_name = os.path.splitext(os.path.basename(spk_path))[0]
        print(f"Loading speaker: {spk_name} from {spk_path}...")
        spk2info = torch.load(spk_path, map_location='cpu')
        # Use filename (without .pt) as the speaker name
        info = list(spk2info.values())[0]
        _cosyvoice_instance.frontend.spk2info[spk_name] = info

    _default_spk_name = DEFAULT_SPK
    if _default_spk_name not in _cosyvoice_instance.frontend.spk2info:
        raise RuntimeError(f"Default speaker '{_default_spk_name}' not found. Available: {list(_cosyvoice_instance.frontend.spk2info.keys())}")

    print(f"Default speaker: {_default_spk_name}")
    print(f"All speakers: {list(_cosyvoice_instance.frontend.spk2info.keys())}")

    return _cosyvoice_instance, _default_spk_name

async def verify_api_key(x_autodl_api_key: str = Header(...)):
    if x_autodl_api_key != AUTODL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_autodl_api_key

@app.on_event("startup")
async def startup_event():
    init_cosyvoice()

def _tts_full_generate(cosyvoice, text: str, spk_name: str, output_path: str) -> bool:
    """Run full TTS generation synchronously (called via asyncio.to_thread)."""
    for result in cosyvoice.inference_zero_shot(
        text,
        'You are a helpful assistant.<|endofprompt|>',
        '',
        zero_shot_spk_id=spk_name,
        stream=False
    ):
        torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
        return True
    return False


def _tts_stream_generator(cosyvoice, text: str, spk_name: str) -> Generator[bytes, None, None]:
    """Yield raw PCM float32 bytes as each audio chunk is generated."""
    for result in cosyvoice.inference_zero_shot(
        text,
        'You are a helpful assistant.<|endofprompt|>',
        '',
        zero_shot_spk_id=spk_name,
        stream=True
    ):
        chunk = result['tts_speech'].numpy()  # (1, num_samples)
        yield chunk.tobytes()


@app.get("/speakers")
async def list_speakers(api_key: str = Depends(verify_api_key)):
    cosyvoice, default_spk = init_cosyvoice()
    return {
        "default": default_spk,
        "speakers": list(cosyvoice.frontend.spk2info.keys()),
    }


@app.post("/tts")
async def generate_tts(request: TTSRequest, api_key: str = Depends(verify_api_key)):

    cosyvoice, default_spk = init_cosyvoice()
    text = smartread_text_normalize(request.text)

    spk_name = request.spk or default_spk
    if spk_name not in cosyvoice.frontend.spk2info:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown speaker '{spk_name}'. Available: {list(cosyvoice.frontend.spk2info.keys())}",
        )

    if request.stream:
        return StreamingResponse(
            _tts_stream_generator(cosyvoice, text, spk_name),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(cosyvoice.sample_rate),
                "X-Audio-Format": "float32",
                "X-Channels": "1",
            },
        )

    # Non-streaming: offload to thread to avoid blocking the event loop
    output_filename = f"tts_output_{uuid.uuid4()}.wav"
    output_path = os.path.join("/tmp", output_filename)

    generated = await asyncio.to_thread(
        _tts_full_generate, cosyvoice, text, spk_name, output_path
    )

    if not generated:
        raise HTTPException(status_code=500, detail="Failed to generate audio")

    return FileResponse(
        output_path, media_type="audio/wav", filename=output_filename,
        background=BackgroundTask(os.unlink, output_path),
    )

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
