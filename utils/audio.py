import os
import subprocess
import traceback
import json
from decimal import Decimal


def merge_audio_and_generate_subtitles(audio_dir):
    """
    使用 ffmpeg 合并音频文件并记录每段音频的时长
    优化版本，解决大量音频文件导致的内存问题
    """
    try:
        
        # 获取音频文件列表
        files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        files.sort(key=lambda x: int(x.split('.')[0]))
        
        if not files:
            print("No audio files found to merge")
            return None
        
        print(f"Processing {len(files)} audio files with FFmpeg...")
        
        duration_dict = {}
        total_duration = Decimal('0')
        stop_duration = Decimal('0.2')  # 200ms停顿
        title_stop_duration = Decimal('1.0')  # 2s停顿 for titles
        # 目标采样率/声道（与TTS输出一致，避免重采样误差）
        target_sample_rate = 24000
        target_channels = 1
        
        # 读取text_index来确定哪些是标题
        text_index = {}
        text_index_path = os.path.join(audio_dir, "text_index.json")
        if os.path.exists(text_index_path):
            with open(text_index_path, 'r', encoding='utf-8') as f:
                text_index = json.load(f)
        
        # 创建静音文件（只创建一次）
        silence_file = os.path.join(audio_dir, "silence.wav")
        title_silence_file = os.path.join(audio_dir, "title_silence.wav")
        
        if not os.path.exists(silence_file):
            print("Creating silence file...")
            result = subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", f"anullsrc=r={target_sample_rate}:cl=mono",
                "-t", str(float(stop_duration)), "-ac", str(target_channels), "-ar", str(target_sample_rate),
                silence_file, "-y"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                print(f"Error creating silence file: {result.stderr}")
                return None
        
        if not os.path.exists(title_silence_file):
            print("Creating title silence file (2s)...")
            result = subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", f"anullsrc=r={target_sample_rate}:cl=mono",
                "-t", str(float(title_stop_duration)), "-ac", str(target_channels), "-ar", str(target_sample_rate),
                title_silence_file, "-y"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                print(f"Error creating title silence file: {result.stderr}")
                return None
        
        # 创建文件列表用于 ffmpeg concat
        list_file = os.path.join(audio_dir, "file_list.txt")
        try:
            with open(list_file, "w", encoding="utf-8") as f:
                for i, audio_file in enumerate(files):
                    audio_path = os.path.join(audio_dir, audio_file)
                    abs_audio_path = os.path.abspath(audio_path)
                    
                    # 使用 ffprobe 获取音频采样数与采样率，优先用样本数计算时长，避免浮点误差
                    result = subprocess.run([
                        "ffprobe", "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=nb_samples,sample_rate", "-of", "json",
                        abs_audio_path
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if result.returncode != 0:
                        print(f"Error getting duration for {audio_file}: {result.stderr}")
                        continue
                    
                    try:
                        duration_dec = None
                        try:
                            probe = json.loads(result.stdout)
                            streams = probe.get("streams", [])
                            if streams:
                                s = streams[0]
                                nb_samples = s.get("nb_samples")
                                sample_rate_str = s.get("sample_rate")
                                if nb_samples is not None and sample_rate_str is not None:
                                    # nb_samples may be string
                                    nb = Decimal(str(nb_samples))
                                    sr = Decimal(str(sample_rate_str))
                                    if sr and sr != 0:
                                        duration_dec = nb / sr
                        except Exception:
                            duration_dec = None
                        if duration_dec is None:
                            # 回退到format=duration
                            res2 = subprocess.run([
                                "ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", abs_audio_path
                            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            duration_dec = Decimal(res2.stdout.strip())

                        # 确定当前音频对应的文本索引
                        audio_index = str(int(audio_file.split('.')[0]))
                        current_text_item = text_index.get(audio_index, {})
                        is_title = current_text_item.get('is_title', False)
                        
                        # 根据是否为标题选择停顿时长
                        if i < len(files) - 1:  # 仅在非最后一个片段后添加停顿
                            pause_duration = title_stop_duration if is_title else stop_duration
                            duration_with_pause = duration_dec + pause_duration
                        else:
                            duration_with_pause = duration_dec
                            
                        duration_dict[audio_file] = str(duration_with_pause)
                        total_duration += duration_with_pause
                        
                        pause_type = "title(2s)" if is_title and i < len(files)-1 else ("normal(0.2s)" if i < len(files)-1 else "none")
                        print(f"Audio {audio_file}: {float(duration_with_pause):.3f}s (pause={pause_type})")
                        
                        # 添加音频文件到列表（使用绝对路径避免路径问题）
                        f.write(f"file '{abs_audio_path}'\n")
                        
                        # 在文件之间插入静音（除了最后一个文件）
                        if i < len(files) - 1:
                            if is_title:
                                abs_silence_path = os.path.abspath(title_silence_file)
                            else:
                                abs_silence_path = os.path.abspath(silence_file)
                            f.write(f"file '{abs_silence_path}'\n")
                            
                    except ValueError as e:
                        print(f"Error parsing duration for {audio_file}: {e}")
                        continue
            
            # 使用 ffmpeg concat 合并音频文件
            output_file = os.path.join(audio_dir, "merged.mp3")
            print("Merging audio files with FFmpeg...")
            
            # Debug: 显示文件列表内容
            try:
                with open(list_file, "r", encoding="utf-8") as debug_f:
                    content = debug_f.read()
                    print(f"File list content preview (first 500 chars):\n{content[:500]}")
            except Exception as debug_e:
                print(f"Could not read file list for debug: {debug_e}")
            
            result = subprocess.run([
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-ar", str(target_sample_rate), "-ac", str(target_channels),
                "-c:a", "libmp3lame", "-b:a", "128k",
                "-write_xing", "1",
                output_file, "-y"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                print(f"Error merging audio files: {result.stderr}")
                return None
            
            print(f"Combined audio saved to: {output_file}")
            print(f"Sum durations (with pauses): {float(total_duration):.3f}s")
            # 读取合并后实际时长，用于后续字幕缩放以消除累计误差
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
            
        finally:
            # 清理临时文件
            try:
                if os.path.exists(list_file):
                    os.remove(list_file)
                if os.path.exists(silence_file):
                    os.remove(silence_file)
                if os.path.exists(title_silence_file):
                    os.remove(title_silence_file)
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary files: {cleanup_error}")
        
    except Exception as e:
        print(f"Error merging audio with FFmpeg: {str(e)}")
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
