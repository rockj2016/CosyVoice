#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for TTS API (streaming & non-streaming)."""

import requests
import wave
import time
import numpy as np

BASE_URL = "https://u855844-dd-2bf2d013bd.westc.seetacloud.com:8443"
API_KEY = "smartread-2026"
HEADERS = {
    "X-Autodl-Api-Key": API_KEY,
    "Content-Type": "application/json",
}
TEST_TEXT = "你好，这是一段语音合成的测试文本。"


def test_non_streaming():
    """Non-streaming: get complete WAV file."""
    print("=== Non-streaming test ===")
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"text": TEST_TEXT, "stream": False},
    )
    elapsed = time.time() - start
    print(f"Status: {resp.status_code}  Time: {elapsed:.2f}s  Size: {len(resp.content)} bytes")

    if resp.status_code == 200:
        output = "output_non_stream.wav"
        with open(output, "wb") as f:
            f.write(resp.content)
        print(f"Saved to {output}")
    else:
        print(f"Error: {resp.text}")


def test_streaming():
    """Streaming: accumulate PCM chunks and save as WAV."""
    print("\n=== Streaming test ===")
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"text": TEST_TEXT, "stream": True},
        stream=True,
    )

    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text}")
        return

    sample_rate = int(resp.headers.get("X-Sample-Rate", 24000))
    audio_format = resp.headers.get("X-Audio-Format", "float32")
    print(f"Sample rate: {sample_rate}  Format: {audio_format}")

    chunks = []
    chunk_count = 0
    first_chunk_time = None
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
                print(f"First chunk received at {first_chunk_time:.2f}s")
            chunks.append(chunk)
            chunk_count += 1

    elapsed = time.time() - start
    raw_pcm = b"".join(chunks)
    print(f"Total chunks: {chunk_count}  Time: {elapsed:.2f}s  PCM size: {len(raw_pcm)} bytes")

    # Convert float32 PCM to int16 WAV
    samples_f32 = np.frombuffer(raw_pcm, dtype=np.float32)
    samples_i16 = np.clip(samples_f32 * 32767, -32768, 32767).astype(np.int16)

    output = "output_stream.wav"
    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.tobytes())
    print(f"Saved to {output}")


if __name__ == "__main__":
    test_non_streaming()
    test_streaming()
