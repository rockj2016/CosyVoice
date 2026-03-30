#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test 5 concurrent TTS requests: 4 streaming + 1 non-streaming."""

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


def run_streaming(idx: int):
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"text": TEST_TEXT, "stream": True},
        stream=True,
    )
    if resp.status_code != 200:
        return {"mode": f"stream-{idx}", "error": resp.status_code}

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
    with wave.open(f"output_p5_stream_{idx}.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.tobytes())

    return {
        "mode": f"stream-{idx}",
        "first_chunk": first_chunk_time,
        "total": total_time,
        "chunks": len(chunks),
    }


def run_non_streaming():
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"text": TEST_TEXT, "stream": False},
    )
    total_time = time.time() - start
    if resp.status_code != 200:
        return {"mode": "non-stream", "error": resp.status_code}

    with open("output_p5_non_stream.wav", "wb") as f:
        f.write(resp.content)

    return {
        "mode": "non-stream",
        "total": total_time,
        "size": len(resp.content),
    }


if __name__ == "__main__":
    print("Launching 5 concurrent requests (4 streaming + 1 non-streaming)...\n")
    t0 = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = []
        for i in range(4):
            futures.append(pool.submit(run_streaming, i))
        futures.append(pool.submit(run_non_streaming))

        for future in as_completed(futures):
            results.append(future.result())

    # Sort: streaming by index, non-streaming last
    results.sort(key=lambda r: (0 if r["mode"].startswith("stream") else 1, r["mode"]))

    for r in results:
        if "error" in r:
            print(f"[{r['mode']:12s}] Error: {r['error']}")
        elif r["mode"] == "non-stream":
            print(f"[{r['mode']:12s}] total: {r['total']:.2f}s  size: {r['size']} bytes")
        else:
            print(f"[{r['mode']:12s}] first_chunk: {r['first_chunk']:.2f}s  "
                  f"total: {r['total']:.2f}s  chunks: {r['chunks']}")

    print(f"\nWall time: {time.time() - t0:.2f}s")
