"""
TTS 流式并发压测脚本

测试 CosyVoice HTTP 流式 TTS API 能支持多少并发用户流畅听书。
判断标准：音频到达速率 >= 播放速率（24kHz 16bit mono = 48,000 bytes/s）。

用法:
    python scripts/test_tts_concurrency.py --url http://localhost:8000/api/v1/test/tts
    python scripts/test_tts_concurrency.py --url http://localhost:8000/api/v1/test/tts --levels 1,2,5,10
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field

import aiohttp

# 24kHz, 16-bit mono PCM 的实时播放速率
PLAYBACK_RATE = 24000 * 2  # 48000 bytes/s

DEFAULT_TEXT = "人工智能正在深刻改变我们的生活方式。从智能手机上的语音助手，到自动驾驶汽车，再到医疗诊断中的影像识别，AI技术无处不在。"


@dataclass
class UserResult:
    """单个模拟用户的测试结果"""

    user_id: int
    success: bool = False
    error: str = ""
    first_byte_latency: float = 0.0  # 秒
    total_time: float = 0.0  # 秒
    total_bytes: int = 0
    chunk_count: int = 0
    throughput: float = 0.0  # bytes/s
    realtime_factor: float = 0.0  # throughput / PLAYBACK_RATE
    stutter_count: int = 0  # buffer 耗尽次数
    chunk_timestamps: list[float] = field(default_factory=list)


async def simulate_user(
    session: aiohttp.ClientSession,
    url: str,
    text: str,
    spk: str | None,
    user_id: int,
    api_key: str | None = None,
    raw_float32: bool = False,
) -> UserResult:
    """模拟一个用户发起流式 TTS 请求并收集指标"""
    result = UserResult(user_id=user_id)
    payload = {"text": text, "stream": True}
    if spk:
        payload["spk"] = spk

    headers = {}
    if api_key:
        headers["X-Autodl-Api-Key"] = api_key

    start = time.monotonic()
    first_byte_received = False
    # 模拟播放 buffer：追踪"已收到但未播放"的字节数
    buffer_bytes = 0.0
    last_chunk_time = start

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                result.error = f"HTTP {resp.status}: {error_body[:200]}"
                return result

            buf = b""
            async for chunk in resp.content.iter_any():
                now = time.monotonic()
                if not first_byte_received:
                    result.first_byte_latency = now - start
                    first_byte_received = True
                    last_chunk_time = now

                # CosyVoice 直连返回 float32 PCM，需要转为 int16 计算大小
                if raw_float32:
                    buf += chunk
                    usable = len(buf) - (len(buf) % 4)
                    if usable == 0:
                        continue
                    import numpy as np

                    samples = np.frombuffer(buf[:usable], dtype=np.float32)
                    buf = buf[usable:]
                    pcm_bytes = len(samples) * 2  # int16 = 2 bytes per sample
                else:
                    pcm_bytes = len(chunk)

                result.chunk_count += 1
                result.total_bytes += pcm_bytes
                result.chunk_timestamps.append(now)

                # 模拟播放 buffer 消耗
                elapsed_since_last = now - last_chunk_time
                played_bytes = elapsed_since_last * PLAYBACK_RATE
                buffer_bytes = buffer_bytes - played_bytes + len(chunk)
                if buffer_bytes < 0:
                    # buffer 耗尽 = 卡顿
                    result.stutter_count += 1
                    buffer_bytes = len(chunk)
                last_chunk_time = now

            result.success = True
            result.total_time = time.monotonic() - start
            if result.total_time > 0:
                result.throughput = result.total_bytes / result.total_time
                result.realtime_factor = result.throughput / PLAYBACK_RATE

    except asyncio.TimeoutError:
        result.error = "Timeout"
        result.total_time = time.monotonic() - start
    except aiohttp.ClientError as e:
        result.error = str(e)
        result.total_time = time.monotonic() - start
    except Exception as e:
        result.error = str(e)
        result.total_time = time.monotonic() - start

    return result


async def run_concurrency_level(
    url: str,
    text: str,
    spk: str | None,
    concurrency: int,
    api_key: str | None = None,
    raw_float32: bool = False,
) -> list[UserResult]:
    """对指定并发数执行一轮测试"""
    timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_read=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            simulate_user(session, url, text, spk, uid, api_key, raw_float32)
            for uid in range(concurrency)
        ]
        return await asyncio.gather(*tasks)


def print_round_report(concurrency: int, results: list[UserResult]):
    """打印单轮测试报告"""
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    smooth = [r for r in success if r.realtime_factor >= 1.0]
    stuttered = [r for r in success if r.stutter_count > 0]

    print(f"\n{'='*70}")
    print(f"  并发数: {concurrency}")
    print(f"{'='*70}")
    print(f"  成功/失败:        {len(success)}/{len(failed)}")

    if failed:
        errors = {}
        for r in failed:
            errors[r.error] = errors.get(r.error, 0) + 1
        for err, cnt in errors.items():
            print(f"    错误: {err} x{cnt}")

    if success:
        avg_fbl = sum(r.first_byte_latency for r in success) / len(success)
        max_fbl = max(r.first_byte_latency for r in success)
        avg_throughput = sum(r.throughput for r in success) / len(success)
        avg_rtf = sum(r.realtime_factor for r in success) / len(success)
        min_rtf = min(r.realtime_factor for r in success)
        avg_time = sum(r.total_time for r in success) / len(success)
        total_stutters = sum(r.stutter_count for r in success)

        print(f"  首字节延迟 (avg): {avg_fbl:.2f}s  (max: {max_fbl:.2f}s)")
        print(f"  平均耗时:         {avg_time:.2f}s")
        print(f"  平均吞吐率:       {avg_throughput:.0f} bytes/s")
        print(
            f"  实时因子 (avg):   {avg_rtf:.2f}x  (min: {min_rtf:.2f}x)"
        )
        print(f"  流畅用户:         {len(smooth)}/{len(success)}")
        print(f"  有卡顿用户:       {len(stuttered)}/{len(success)}  (共 {total_stutters} 次)")

        # 判定
        if min_rtf >= 1.0 and len(stuttered) == 0:
            verdict = "ALL SMOOTH"
        elif len(smooth) >= len(success) * 0.8:
            verdict = "MOSTLY SMOOTH"
        else:
            verdict = "DEGRADED"
        print(f"  结论:             {verdict}")
    else:
        print("  所有请求均失败！")

    print(f"{'='*70}")
    return len(success), len(failed)


async def main():
    parser = argparse.ArgumentParser(description="TTS 流式并发压测")
    parser.add_argument(
        "--url",
        default="http://54.187.177.148/tts",
        help="TTS API 地址",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="测试文本")
    parser.add_argument(
        "--levels",
        default="1,2,5,10,15,20,30,50",
        help="并发级别，逗号分隔",
    )
    parser.add_argument("--rounds", type=int, default=1, help="每个级别重复次数")
    parser.add_argument("--spk", default=None, help="Speaker 名称")
    parser.add_argument("--api-key", default="autodl-tts-secret-key-2024", help="X-Autodl-Api-Key (直连 CosyVoice 时需要)")
    parser.add_argument(
        "--raw-float32",
        action="store_true",
        help="CosyVoice 直连返回 float32 PCM，需转换为 int16 计算",
    )
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",")]

    print(f"TTS 流式并发压测")
    print(f"目标: {args.url}")
    print(f"文本长度: {len(args.text)} 字符")
    print(f"并发级别: {levels}")
    print(f"播放速率: {PLAYBACK_RATE} bytes/s (24kHz 16bit mono)")
    print(f"判定标准: 实时因子 >= 1.0 且无卡顿 = 流畅")
    if args.raw_float32:
        print(f"模式: float32 -> int16 转换")

    max_smooth_concurrency = 0
    results: list[UserResult] = []

    for level in levels:
        for round_num in range(1, args.rounds + 1):
            if args.rounds > 1:
                print(f"\n--- 并发 {level}, 第 {round_num}/{args.rounds} 轮 ---")

            results = await run_concurrency_level(
                args.url, args.text, args.spk, level, args.api_key, args.raw_float32
            )
            success_count, _ = print_round_report(level, results)

            # 判断是否全部流畅
            smooth_results = [
                r for r in results if r.success and r.realtime_factor >= 1.0
            ]
            if smooth_results and len(smooth_results) == success_count:
                max_smooth_concurrency = level

        # 如果连续两个级别全部失败，提前退出
        all_failed = all(not r.success for r in results)
        if all_failed:
            print("\n所有请求失败，停止测试。")
            break

    print(f"\n{'#'*70}")
    print(f"  最终结论: 最大全员流畅并发数 = {max_smooth_concurrency}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    asyncio.run(main())
