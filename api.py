#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS API Service

Usage: python api.py
"""
import os
import sys
import uvicorn
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from dotenv import load_dotenv
from utils.normalize import smartread_text_normalize

# Add third_party path
sys.path.append('third_party/Matcha-TTS')

# Load environment variables
load_dotenv()

# Configuration from .env
AUTODL_API_KEY = os.getenv('AUTODL_API_KEY', 'autodl-tts-secret-key-2024')
SPK_INFO_PATH = os.getenv('SPK_INFO_PATH', './spkinfo.pt')
MODEL_DIR = os.getenv('MODEL_DIR', 'pretrained_models/Fun-CosyVoice3-0.5B')
HOST = os.getenv('API_HOST', '0.0.0.0')
PORT = int(os.getenv('API_PORT', 8000))

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
_spk_name = None

app = FastAPI(title="CosyVoice TTS API")

class TTSRequest(BaseModel):
    text: str = Field(..., max_length=100, description="Text to synthesize (max 100 chars)")

def init_cosyvoice():
    """Initialize CosyVoice model and load speaker info (singleton)"""
    global _cosyvoice_instance, _spk_name
    
    if _cosyvoice_instance is not None:
        return _cosyvoice_instance, _spk_name
    
    print(f"Initializing CosyVoice model from {MODEL_DIR}...")
    _cosyvoice_instance = AutoModel(
        model_dir=MODEL_DIR,
        load_trt=True,
        load_vllm=True,
        fp16=False
    )
    
    # Load speaker info
    print(f"Loading speaker info from {SPK_INFO_PATH}...")
    spk2info = torch.load(SPK_INFO_PATH, map_location='cpu')
    _spk_name = list(spk2info.keys())[0]
    _cosyvoice_instance.frontend.spk2info[_spk_name] = spk2info[_spk_name]
    print(f"Loaded speaker: {_spk_name}")
    
    return _cosyvoice_instance, _spk_name

async def verify_api_key(x_autodl_api_key: str = Header(...)):
    if x_autodl_api_key != AUTODL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_autodl_api_key

@app.on_event("startup")
async def startup_event():
    init_cosyvoice()

@app.post("/tts")
async def generate_tts(request: TTSRequest, api_key: str = Depends(verify_api_key)):
    try:
        cosyvoice, spk_name = init_cosyvoice()
        
        # Create a unique filename for the output
        output_filename = f"tts_output_{uuid.uuid4()}.wav"
        output_path = os.path.join("/tmp", output_filename)
        
        # Generate speech using pre-loaded speaker info
        # Using the same logic as run.py
        generated = False
        text = smartread_text_normalize(request.text)
        for i, result in enumerate(cosyvoice.inference_zero_shot(
            text,
            'You are a helpful assistant.<|endofprompt|>',
            '',  # Empty prompt_wav since we use cached spk_info
            zero_shot_spk_id=spk_name,
            stream=False
        )):
            torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
            generated = True
            break  # Only need first result
            
        if not generated:
             raise HTTPException(status_code=500, detail="Failed to generate audio")

        return FileResponse(output_path, media_type="audio/wav", filename=output_filename)

    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
