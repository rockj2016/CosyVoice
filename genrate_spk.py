#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('third_party/Matcha-TTS')

"""
Script to generate speaker info .pt file for CosyVoice inference.

Usage:
    python spk_example.py --wav_path <path_to_wav> \
                          --prompt_text <text_content> \
                          --spk_name <speaker_name> \
                          --output_path <output_pt_file> \
                          [--model_dir <model_directory>]

Example:
    python spk_example.py --wav_path /path/to/speaker.wav \
                          --prompt_text "这是一段示例文本" \
                          --spk_name "speaker1" \
                          --output_path ./speaker1.pt \
                          --model_dir pretrained_models/CosyVoice2-0.5B

CosyVoice3 NOTE:
    CosyVoice3 prompt is instruction-style and expects the delimiter token
    "<|endofprompt|>" (see `example.py` / `vllm_example.py`). If you don't
    include it, synthesis quality may degrade (e.g., intermittent / garbled audio).
    This script will auto-prepend "You are a helpful assistant.<|endofprompt|>"
    for CosyVoice3 when missing.
"""

import argparse
import os
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3, AutoModel


def load_wav(wav_path: str) -> torch.Tensor:
    """Load wav file and return tensor."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    
    speech, sample_rate = torchaudio.load(wav_path, backend='soundfile')
    # Convert to mono if stereo
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    # Flatten to 1D tensor
    speech = speech.squeeze()
    return speech


def generate_spk_info(
    model_dir: str,
    wav_path: str,
    prompt_text: str,
    spk_name: str,
    output_path: str
):
    """
    Generate speaker info .pt file from wav and prompt text.
    
    Args:
        model_dir: Path to the CosyVoice model directory
        wav_path: Path to the input wav file
        prompt_text: The text content corresponding to the wav file
        spk_name: Speaker name/id to use as key
        output_path: Path to save the output .pt file
    """
    print(f"Loading model from {model_dir}...")
    cosyvoice = AutoModel(model_dir=model_dir)

    # CosyVoice3 uses an instruction-style prompt. The repo examples always
    # include "<|endofprompt|>" to separate the instruction from the content
    # and to avoid text_frontend normalization changing the prompt text.
    if cosyvoice.__class__.__name__ == "CosyVoice3" and "<|endofprompt|>" not in prompt_text:
        prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        print("NOTE: Detected CosyVoice3 and '<|endofprompt|>' missing in --prompt_text; "
              "auto-prepended 'You are a helpful assistant.<|endofprompt|>'.")
    
    print(f"Generating speaker info for '{spk_name}' from {wav_path}...")
    # Use the frontend to extract speaker info (same as add_zero_shot_spk)
    # Note: frontend_zero_shot expects the wav file path, not a loaded tensor
    model_input = cosyvoice.frontend.frontend_zero_shot(
        '', prompt_text, wav_path, cosyvoice.sample_rate, ''
    )
    
    # Remove text-related keys, keep only speaker info
    del model_input['text']
    del model_input['text_len']
    
    # Create spk2info dict with the speaker info
    spk2info = {spk_name: model_input}
    
    # Save to .pt file
    print(f"Saving speaker info to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(spk2info, output_path)
    
    print(f"Successfully generated speaker info file: {output_path}")
    print(f"Speaker info contains keys: {list(model_input.keys())}")
    
    return spk2info


def main():
    parser = argparse.ArgumentParser(
        description="Generate speaker info .pt file for CosyVoice inference"
    )
    parser.add_argument(
        "--wav_path",
        type=str,
        required=True,
        help="Path to the input wav file"
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        required=True,
        help="Text content corresponding to the wav file"
    )
    parser.add_argument(
        "--spk_name",
        type=str,
        required=True,
        help="Speaker name/id to use as key in the spk2info dict"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output .pt file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="Path to the CosyVoice model directory (default: pretrained_models/CosyVoice2-0.5B)"
    )
    
    args = parser.parse_args()
    
    generate_spk_info(
        model_dir=args.model_dir,
        wav_path=args.wav_path,
        prompt_text=args.prompt_text,
        spk_name=args.spk_name,
        output_path=args.output_path
    )

"""
python spk_example.py --wav_path ./asset/nlcj.wav \
                    --prompt_text "这是一款为高效学习设计的智能听书软件。" \
                    --spk_name "nlcj" \
                    --output_path ./spkinfo.pt \
                    --model_dir pretrained_models/Fun-CosyVoice3-0.5B
"""
"""
python genrate_spk.py --wav_path ./asset/en_voice.wav \
                    --prompt_text "When I read these sections, I kept catching myself thinking." \
                    --spk_name "nlcj" \
                    --output_path ./spkinfo_en.pt \
                    --model_dir pretrained_models/Fun-CosyVoice3-0.5B
"""

if __name__ == "__main__":

    main()
