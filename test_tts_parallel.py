#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test concurrent streaming + non-streaming TTS requests."""

import requests
import wave
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://u855844-dd-2bf2d013bd.westc.seetacloud.com:8443"
API_KEY = "smartread-2026"
HEADERS = {
    "X-Autodl-Api-Key": API_KEY,
    "Content-Type": "application/json",
}
TEST_TEXT = "你好，这是一段语音合成的测试文本。"


def run_streaming():
    """Streaming request, return timing info."""
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"text": TEST_TEXT, "stream": True},
        stream=True,
    )
    if resp.status_code != 200:
        return {"mode": "streaming", "error": resp.status_code}

    sample_rate = int(resp.headers.get("X-Sample-Rate", 24000))
    first_chunk_time = None
    chunks = []
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
            chunks.append(chunk)

    total_time = time.time() - start
    raw_pcm = b"".join(chunks)
    samples_f32 = np.frombuffer(raw_pcm, dtype=np.float32)
    samples_i16 = np.clip(samples_f32 * 32767, -32768, 32767).astype(np.int16)
    with wave.open("output_parallel_stream.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.tobytes())

    return {
        "mode": "streaming",
        "first_chunk": first_chunk_time,
        "total": total_time,
        "chunks": len(chunks),
        "pcm_bytes": len(raw_pcm),
    }


def run_non_streaming():
    """Non-streaming request, return timing info."""
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"text": TEST_TEXT, "stream": False},
    )
    total_time = time.time() - start
    if resp.status_code != 200:
        return {"mode": "non-streaming", "error": resp.status_code}

    with open("output_parallel_non_stream.wav", "wb") as f:
        f.write(resp.content)

    return {
        "mode": "non-streaming",
        "total": total_time,
        "size": len(resp.content),
    }


if __name__ == "__main__":
    print("Launching 2 concurrent requests (1 streaming + 1 non-streaming)...\n")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_streaming): "streaming",
            pool.submit(run_non_streaming): "non-streaming",
        }
        for future in as_completed(futures):
            result = future.result()
            mode = result["mode"]
            if "error" in result:
                print(f"[{mode}] Error: {result['error']}")
                continue
            if mode == "streaming":
                print(f"[streaming]     first chunk: {result['first_chunk']:.2f}s  "
                      f"total: {result['total']:.2f}s  "
                      f"chunks: {result['chunks']}  "
                      f"pcm: {result['pcm_bytes']} bytes")
            else:
                print(f"[non-streaming] total: {result['total']:.2f}s  "
                      f"size: {result['size']} bytes")

    print(f"\nWall time: {time.time() - t0:.2f}s")
