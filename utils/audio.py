import os
import subprocess
import traceback
import json
from decimal import Decimal
import numpy as np



# def merge_audio_and_generate_subtitles(audio_dir):
#     """
#     使用 ffmpeg 合并音频文件并记录每段音频的时长
#     优化版本，解决大量音频文件导致的内存问题
#     """
#     try:
        
#         # 获取音频文件列表
#         files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
#         files.sort(key=lambda x: int(x.split('.')[0]))
        
#         if not files:
#             print("No audio files found to merge")
#             return None
        
#         print(f"Processing {len(files)} audio files with FFmpeg...")
        
#         duration_dict = {}
#         total_duration = Decimal('0')
#         stop_duration = Decimal('0.2')  # 200ms停顿
#         title_stop_duration = Decimal('1.0')  # 2s停顿 for titles
#         # 目标采样率/声道（与TTS输出一致，避免重采样误差）
#         target_sample_rate = 24000
#         target_channels = 1
        
#         # 读取text_index来确定哪些是标题
#         text_index = {}
#         text_index_path = os.path.join(audio_dir, "text_index.json")
#         if os.path.exists(text_index_path):
#             with open(text_index_path, 'r', encoding='utf-8') as f:
#                 text_index = json.load(f)
        
#         # 创建静音文件（只创建一次）
#         silence_file = os.path.join(audio_dir, "silence.wav")
#         title_silence_file = os.path.join(audio_dir, "title_silence.wav")
        
#         if not os.path.exists(silence_file):
#             print("Creating silence file...")
#             result = subprocess.run([
#                 "ffmpeg", "-f", "lavfi", "-i", f"anullsrc=r={target_sample_rate}:cl=mono",
#                 "-t", str(float(stop_duration)), "-ac", str(target_channels), "-ar", str(target_sample_rate),
#                 silence_file, "-y"
#             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
#             if result.returncode != 0:
#                 print(f"Error creating silence file: {result.stderr}")
#                 return None
        
#         if not os.path.exists(title_silence_file):
#             print("Creating title silence file (2s)...")
#             result = subprocess.run([
#                 "ffmpeg", "-f", "lavfi", "-i", f"anullsrc=r={target_sample_rate}:cl=mono",
#                 "-t", str(float(title_stop_duration)), "-ac", str(target_channels), "-ar", str(target_sample_rate),
#                 title_silence_file, "-y"
#             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
#             if result.returncode != 0:
#                 print(f"Error creating title silence file: {result.stderr}")
#                 return None
        
#         # 创建文件列表用于 ffmpeg concat
#         list_file = os.path.join(audio_dir, "file_list.txt")
#         try:
#             with open(list_file, "w", encoding="utf-8") as f:
#                 for i, audio_file in enumerate(files):
#                     audio_path = os.path.join(audio_dir, audio_file)
#                     abs_audio_path = os.path.abspath(audio_path)
                    
#                     # 使用 ffprobe 获取音频采样数与采样率，优先用样本数计算时长，避免浮点误差
#                     result = subprocess.run([
#                         "ffprobe", "-v", "error", "-select_streams", "a:0",
#                         "-show_entries", "stream=nb_samples,sample_rate", "-of", "json",
#                         abs_audio_path
#                     ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
#                     if result.returncode != 0:
#                         print(f"Error getting duration for {audio_file}: {result.stderr}")
#                         continue
                    
#                     try:
#                         duration_dec = None
#                         try:
#                             probe = json.loads(result.stdout)
#                             streams = probe.get("streams", [])
#                             if streams:
#                                 s = streams[0]
#                                 nb_samples = s.get("nb_samples")
#                                 sample_rate_str = s.get("sample_rate")
#                                 if nb_samples is not None and sample_rate_str is not None:
#                                     # nb_samples may be string
#                                     nb = Decimal(str(nb_samples))
#                                     sr = Decimal(str(sample_rate_str))
#                                     if sr and sr != 0:
#                                         duration_dec = nb / sr
#                         except Exception:
#                             duration_dec = None
#                         if duration_dec is None:
#                             # 回退到format=duration
#                             res2 = subprocess.run([
#                                 "ffprobe", "-v", "error", "-show_entries",
#                                 "format=duration", "-of",
#                                 "default=noprint_wrappers=1:nokey=1", abs_audio_path
#                             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#                             duration_dec = Decimal(res2.stdout.strip())

#                         # 确定当前音频对应的文本索引
#                         audio_index = str(int(audio_file.split('.')[0]))
#                         current_text_item = text_index.get(audio_index, {})
#                         is_title = current_text_item.get('is_title', False)
                        
#                         # 根据是否为标题选择停顿时长
#                         if i < len(files) - 1:  # 仅在非最后一个片段后添加停顿
#                             pause_duration = title_stop_duration if is_title else stop_duration
#                             duration_with_pause = duration_dec + pause_duration
#                         else:
#                             duration_with_pause = duration_dec
                            
#                         duration_dict[audio_file] = str(duration_with_pause)
#                         total_duration += duration_with_pause
                        
#                         pause_type = "title(2s)" if is_title and i < len(files)-1 else ("normal(0.2s)" if i < len(files)-1 else "none")
#                         print(f"Audio {audio_file}: {float(duration_with_pause):.3f}s (pause={pause_type})")
                        
#                         # 添加音频文件到列表（使用绝对路径避免路径问题）
#                         f.write(f"file '{abs_audio_path}'\n")
                        
#                         # 在文件之间插入静音（除了最后一个文件）
#                         if i < len(files) - 1:
#                             if is_title:
#                                 abs_silence_path = os.path.abspath(title_silence_file)
#                             else:
#                                 abs_silence_path = os.path.abspath(silence_file)
#                             f.write(f"file '{abs_silence_path}'\n")
                            
#                     except ValueError as e:
#                         print(f"Error parsing duration for {audio_file}: {e}")
#                         continue
            
#             # 使用 ffmpeg concat 合并音频文件
#             output_file = os.path.join(audio_dir, "merged.mp3")
#             print("Merging audio files with FFmpeg...")
            
#             # Debug: 显示文件列表内容
#             try:
#                 with open(list_file, "r", encoding="utf-8") as debug_f:
#                     content = debug_f.read()
#                     print(f"File list content preview (first 500 chars):\n{content[:500]}")
#             except Exception as debug_e:
#                 print(f"Could not read file list for debug: {debug_e}")
            
#             result = subprocess.run([
#                 "ffmpeg", "-f", "concat", "-safe", "0",
#                 "-i", list_file,
#                 "-ar", str(target_sample_rate), "-ac", str(target_channels),
#                 "-c:a", "libmp3lame", "-b:a", "128k",
#                 "-write_xing", "1",
#                 output_file, "-y"
#             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
#             if result.returncode != 0:
#                 print(f"Error merging audio files: {result.stderr}")
#                 return None
            
#             print(f"Combined audio saved to: {output_file}")
#             print(f"Sum durations (with pauses): {float(total_duration):.3f}s")
#             # 读取合并后实际时长，用于后续字幕缩放以消除累计误差
#             probe_merged = subprocess.run([
#                 "ffprobe", "-v", "error", "-show_entries",
#                 "format=duration", "-of",
#                 "default=noprint_wrappers=1:nokey=1", output_file
#             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#             if probe_merged.returncode == 0:
#                 try:
#                     merged_duration = Decimal(probe_merged.stdout.strip())
#                     with open(os.path.join(audio_dir, "merged_duration.txt"), "w", encoding="utf-8") as f_md:
#                         f_md.write(str(merged_duration))
#                     print(f"Merged mp3 duration: {float(merged_duration):.3f}s")
#                 except Exception:
#                     pass
            
#             # 保存时长信息
#             with open(os.path.join(audio_dir, "text_duration.json"), "w", encoding="utf-8") as f:
#                 json.dump(duration_dict, f, ensure_ascii=False, indent=4)
            
#             # 生成字幕数据
#             merge_duration_and_text(audio_dir)
            
#             return int(float(total_duration))
            
#         finally:
#             # 清理临时文件
#             try:
#                 if os.path.exists(list_file):
#                     os.remove(list_file)
#                 if os.path.exists(silence_file):
#                     os.remove(silence_file)
#                 if os.path.exists(title_silence_file):
#                     os.remove(title_silence_file)
#             except Exception as cleanup_error:
#                 print(f"Error cleaning up temporary files: {cleanup_error}")
        
#     except Exception as e:
#         print(f"Error merging audio with FFmpeg: {str(e)}")
#         print(traceback.format_exc())
#         return None

def clean_wav_noise(wav_path, rms_threshold=0.015, window_ms=50):
    """
    清除单个 WAV 文件首尾的 TTS 噪声。
    使用 RMS 阈值检测有效语音边界，对静音段做淡入/淡出处理后写回原文件。

    Args:
        wav_path: WAV 文件的绝对路径
        rms_threshold: RMS 阈值，低于此值视为噪声（默认 0.015）
        window_ms: 检测窗口大小（毫秒，默认 5ms）

    Returns:
        True if file was modified, False otherwise
    """
    import wave as _wave
    try:
        with _wave.open(wav_path, 'rb') as wf:
            _sr = wf.getframerate()
            _sw = wf.getsampwidth()
            _raw = wf.readframes(wf.getnframes())
        if _sw != 2:
            return False
        _samples = np.frombuffer(_raw, dtype=np.int16).astype(np.float32) / 32768.0
        _chunk = int(_sr * window_ms / 1000)
        _modified = False
        _basename = os.path.basename(wav_path)

        # --- 清除开头噪声 ---
        _vstart = 0
        for _i in range(0, len(_samples) - _chunk, _chunk):
            _rms = np.sqrt(np.mean(_samples[_i:_i+_chunk]**2))
            if _rms >= rms_threshold:
                _vstart = max(0, _i - _chunk * 2)
                break
        if _vstart > 0:
            _samples[:_vstart] = 0.0
            _fade = min(int(_sr * window_ms / 1000), len(_samples) - _vstart)
            if _fade > 0:
                _samples[_vstart:_vstart+_fade] *= np.linspace(0, 1, _fade, dtype=np.float32)
            _modified = True
            print(f"  Cleaned {_basename}: leading noise silenced up to {_vstart/_sr*1000:.0f}ms")

        # --- 清除结尾噪声 ---
        _vend = len(_samples)
        for _i in range(len(_samples) - _chunk, _chunk, -_chunk):
            _rms = np.sqrt(np.mean(_samples[_i:_i+_chunk]**2))
            if _rms >= rms_threshold:
                _vend = min(len(_samples), _i + _chunk + _chunk * 2)
                break
        if _vend < len(_samples):
            _fade = min(int(_sr * window_ms / 1000), _vend)
            if _fade > 0:
                _samples[_vend-_fade:_vend] *= np.linspace(1, 0, _fade, dtype=np.float32)
            _samples[_vend:] = 0.0
            _modified = True
            _tail_ms = (len(_samples) - _vend) / _sr * 1000
            print(f"  Cleaned {_basename}: trailing noise silenced from {_vend/_sr*1000:.0f}ms ({_tail_ms:.0f}ms removed)")

        # --- 写回文件 ---
        if _modified:
            _pcm = np.clip(_samples, -1.0, 1.0)
            _pcm_out = (_pcm * 32767).astype(np.int16).tobytes()
            with _wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_sr)
                wf.writeframes(_pcm_out)

        return _modified
    except Exception as _e:
        print(f"  Warning: clean noise failed for {os.path.basename(wav_path)}: {_e}")
        return False


def load_wav_as_float32(wav_path, target_sr=24000):
    """
    读取 WAV 文件并返回 float32 numpy 数组和采样率。
    如果采样率与 target_sr 不一致，用 ffmpeg 重采样。
    """
    import struct
    import wave

    # 先用 ffprobe 检查采样率
    probe_result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,sample_fmt",
        "-of", "json", wav_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    need_resample = False
    if probe_result.returncode == 0:
        try:
            info = json.loads(probe_result.stdout)
            streams = info.get("streams", [])
            if streams:
                sr_actual = int(streams[0].get("sample_rate", target_sr))
                if sr_actual != target_sr:
                    need_resample = True
        except Exception:
            pass

    if need_resample:
        # 重采样到目标采样率
        tmp_path = wav_path + ".resampled.wav"
        subprocess.run([
            "ffmpeg", "-i", wav_path, "-ar", str(target_sr), "-ac", "1",
            "-sample_fmt", "s16", tmp_path, "-y"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        read_path = tmp_path
    else:
        read_path = wav_path

    # 用 wave 模块读取 PCM 数据
    with wave.open(read_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # 转为 numpy float32
    if sampwidth == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # 取第一个声道（如果是多声道）
    if n_channels > 1:
        samples = samples[::n_channels]

    # 清理临时文件
    if need_resample and os.path.exists(read_path):
        os.remove(read_path)

    return samples, sr


def crossfade_segments(seg_a, seg_b, crossfade_samples):
    """
    对两段音频进行 crossfade 拼接。
    使用 equal-power (cosine) 曲线实现平滑过渡。

    Args:
        seg_a: 前一段音频 (numpy float32 array)
        seg_b: 后一段音频 (numpy float32 array)
        crossfade_samples: crossfade 的采样点数

    Returns:
        拼接后的音频 (numpy float32 array)
    """
    # 确保 crossfade 不超过任一片段长度
    cf = min(crossfade_samples, len(seg_a), len(seg_b))

    if cf <= 0:
        # 无法 crossfade，直接拼接
        return np.concatenate([seg_a, seg_b])

    # equal-power crossfade 曲线
    t = np.linspace(0, np.pi / 2, cf, dtype=np.float32)
    fade_out = np.cos(t)    # 1 → 0
    fade_in = np.sin(t)     # 0 → 1

    # 非重叠部分
    result_parts = []
    if len(seg_a) > cf:
        result_parts.append(seg_a[:-cf])

    # 重叠部分：混合
    overlap = seg_a[-cf:] * fade_out + seg_b[:cf] * fade_in
    result_parts.append(overlap)

    # seg_b 剩余部分
    if len(seg_b) > cf:
        result_parts.append(seg_b[cf:])

    return np.concatenate(result_parts)
    

def merge_audio_and_generate_subtitles(audio_dir, crossfade_ms=10):
    """
    使用 Crossfade 方法合并音频文件并记录每段音频的时长。
    通过在每个音频片段连接处应用交叉淡入淡出（crossfade），
    消除拼接点处的杂音（pop/click）。

    Args:
        audio_dir: 包含音频文件的目录
        crossfade_ms: crossfade 时长（毫秒），默认 10ms
    """
    try:

        # 获取音频文件列表
        files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        files.sort(key=lambda x: int(x.split('.')[0]))

        if not files:
            print("No audio files found to merge")
            return None

        print(f"Processing {len(files)} audio files with Crossfade method...")
        print(f"  Crossfade duration: {crossfade_ms}ms")

        # 目标采样率/声道（与TTS输出一致）
        target_sample_rate = 24000
        target_channels = 1
        crossfade_samples = int(target_sample_rate * crossfade_ms / 1000)

        duration_dict = {}
        total_duration = Decimal('0')
        stop_duration = Decimal('0.2')       # 200ms 停顿
        title_stop_duration = Decimal('1.0')  # 1s 停顿 for titles

        # 读取 text_index 来确定哪些是标题
        text_index = {}
        text_index_path = os.path.join(audio_dir, "text_index.json")
        if os.path.exists(text_index_path):
            with open(text_index_path, 'r', encoding='utf-8') as f:
                text_index = json.load(f)

        # ── 清除每个片段开头和结尾的 TTS 噪声 ──
        print("Cleaning leading/trailing noise from audio segments...")
        for audio_file in files:
            audio_path = os.path.join(audio_dir, audio_file)
            abs_path = os.path.abspath(audio_path)
            clean_wav_noise(abs_path)

        # ── 读取所有音频片段到内存 ──
        print("Loading audio segments...")
        segments = []
        segment_infos = []  # (audio_file, is_title) tuples

        for i, audio_file in enumerate(files):
            audio_path = os.path.join(audio_dir, audio_file)
            abs_audio_path = os.path.abspath(audio_path)

            try:
                samples, sr = load_wav_as_float32(abs_audio_path, target_sr=target_sample_rate)
                segments.append(samples)

                # 确定当前音频对应的文本索引
                audio_index = str(int(audio_file.split('.')[0]))
                current_text_item = text_index.get(audio_index, {})
                is_title = current_text_item.get('is_title', False)
                segment_infos.append((audio_file, is_title))

                # 计算精确时长
                duration_dec = Decimal(str(len(samples))) / Decimal(str(target_sample_rate))

                # 根据是否为标题选择停顿时长
                if i < len(files) - 1:
                    pause_duration = title_stop_duration if is_title else stop_duration
                    duration_with_pause = duration_dec + pause_duration
                else:
                    duration_with_pause = duration_dec

                duration_dict[audio_file] = str(duration_with_pause)
                total_duration += duration_with_pause

                pause_type = "title(1s)" if is_title and i < len(files)-1 else ("normal(0.2s)" if i < len(files)-1 else "none")
                print(f"  Audio {audio_file}: {float(duration_with_pause):.3f}s (pause={pause_type})")

            except Exception as e:
                print(f"  Error loading {audio_file}: {e}")
                continue

        if not segments:
            print("No audio segments loaded successfully")
            return None

        # ── 使用 Crossfade 拼接所有片段 ──
        print(f"\nApplying crossfade ({crossfade_ms}ms) between {len(segments)} segments...")

        # 生成静音间隔的 numpy 数组
        silence_normal = np.zeros(int(target_sample_rate * float(stop_duration)), dtype=np.float32)
        silence_title = np.zeros(int(target_sample_rate * float(title_stop_duration)), dtype=np.float32)

        # 逐步拼接：片段 + 静音（带 fade-out / fade-in，不裁剪采样点）
        combined = segments[0].copy()

        for i in range(1, len(segments)):
            audio_file, is_title_prev = segment_infos[i - 1]

            # 选择上一个片段后面应该插入的静音
            if is_title_prev:
                silence = silence_title
            else:
                silence = silence_normal

            # 对 combined 尾部做 fade-out（原地修改振幅，不裁剪采样点）
            fade_len = min(crossfade_samples, len(combined))
            if fade_len > 0:
                t = np.linspace(0, np.pi / 2, fade_len, dtype=np.float32)
                combined[-fade_len:] *= np.cos(t)  # 1 → 0

            # 对下一段头部做 fade-in
            seg = segments[i].copy()
            fade_len_b = min(crossfade_samples, len(seg))
            if fade_len_b > 0:
                t_b = np.linspace(0, np.pi / 2, fade_len_b, dtype=np.float32)
                seg[:fade_len_b] *= np.sin(t_b)  # 0 → 1

            # 直接拼接：combined + silence + seg（静音时长完整保留）
            combined = np.concatenate([combined, silence, seg])

            if (i % 50 == 0) or (i == len(segments) - 1):
                print(f"  Merged {i+1}/{len(segments)} segments...")

        # ── 导出为 MP3 ──
        output_file = os.path.join(audio_dir, "merged.mp3")
        print(f"\nExporting merged audio to: {output_file}")

        # 先写入临时 WAV，再用 ffmpeg 转 MP3
        tmp_wav = os.path.join(audio_dir, "_merged_tmp.wav")
        try:
            import wave

            # 转为 int16 PCM
            pcm_int16 = np.clip(combined, -1.0, 1.0)
            pcm_int16 = (pcm_int16 * 32767).astype(np.int16)

            with wave.open(tmp_wav, 'wb') as wf:
                wf.setnchannels(target_channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(target_sample_rate)
                wf.writeframes(pcm_int16.tobytes())

            # ffmpeg WAV → MP3
            result = subprocess.run([
                "ffmpeg", "-i", tmp_wav,
                "-ar", str(target_sample_rate), "-ac", str(target_channels),
                "-c:a", "libmp3lame", "-b:a", "128k",
                "-write_xing", "1",
                output_file, "-y"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                print(f"Error encoding MP3: {result.stderr}")
                return None

        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        print(f"Combined audio saved to: {output_file}")
        print(f"Sum durations (with pauses): {float(total_duration):.3f}s")

        # 读取合并后实际时长
        probe_merged = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", output_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if probe_merged.returncode == 0:
            try:
                merged_duration = Decimal(probe_merged.stdout.strip())
                with open(os.path.join(audio_dir, "merged_duration.txt"), "w", encoding="utf-8") as f_md:
                    f_md.write(str(merged_duration))
                print(f"Merged mp3 duration: {float(merged_duration):.3f}s")
            except Exception:
                pass

        # 保存时长信息
        with open(os.path.join(audio_dir, "text_duration.json"), "w", encoding="utf-8") as f:
            json.dump(duration_dict, f, ensure_ascii=False, indent=4)

        # 生成字幕数据
        merge_duration_and_text(audio_dir)

        return int(float(total_duration))

    except Exception as e:
        print(f"Error merging audio with Crossfade: {str(e)}")
        print(traceback.format_exc())
        return None



def merge_duration_and_text(audio_dir):
    """
    合并时长和文本数据，生成字幕文件（支持子章节信息）
    """
    try:
        # 使用高精度累加，避免逐段四舍五入导致的累计误差
        start_at = Decimal('0')
        data = []
        
        # 读取文本索引
        with open(f'{audio_dir}/text_index.json', 'r', encoding='utf-8') as f:
            text_index = json.load(f)
        
        # 读取时长信息
        with open(f'{audio_dir}/text_duration.json', 'r', encoding='utf-8') as f:
            duration_dict = json.load(f)
        
        # 生成字幕数据
        # 按数字顺序遍历，确保与音频片段顺序一致
        ordered_files = sorted(duration_dict.keys(), key=lambda x: int(x.split('.')[0]))
        # 如果存在实际合并后的时长，则按比例缩放，消除累计误差（例如编码器填充帧）
        merged_duration_path = os.path.join(audio_dir, "merged_duration.txt")
        scale_ratio = None
        if os.path.exists(merged_duration_path):
            try:
                with open(merged_duration_path, 'r', encoding='utf-8') as fmd:
                    merged_total = Decimal(fmd.read().strip())
                computed_total = sum(Decimal(str(duration_dict[fn])) for fn in ordered_files)
                if computed_total > 0:
                    scale_ratio = merged_total / computed_total
                    # 仅在比例明显偏离1时缩放
                    if abs(scale_ratio - Decimal('1')) < Decimal('0.002'):
                        scale_ratio = None
            except Exception:
                scale_ratio = None

        for file_name in ordered_files:
            duration = Decimal(str(duration_dict[file_name]))
            if scale_ratio is not None:
                duration = (duration * scale_ratio)
            end_at = start_at + duration
            
            # 获取对应的文本信息
            index = file_name.split('.')[0]
            text_item = text_index.get(index, {})
            
            caption_item = {
                'start_at': format(start_at, '.2f'),
                'end_at': format(end_at, '.2f'),
                'text': text_item.get('text', ''),
                'sub_chapter_id': text_item.get('sub_chapter_id', None)
            }
            
            data.append(caption_item)
            start_at += duration
        
        # 保存字幕数据
        with open(f'{audio_dir}/data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"Generated subtitle data with {len(data)} segments")
        
    except Exception as e:
        print(f"Error generating subtitle data: {str(e)}")
        print(traceback.format_exc())
