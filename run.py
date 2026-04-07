#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS Book Processing Script

Usage: python run.py --book_id <book_id> --spk <speaker_name> [--lang zh|en]

从外部服务获取书本信息，生成 TTS 音频和字幕，并将结果上传回外部服务。
"""
import os
import sys
import json
import glob
import argparse
import requests
import torch
import torchaudio

# Add third_party path
sys.path.append('third_party/Matcha-TTS')

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
EXTERNAL_API_HOST = os.getenv('EXTERNAL_API_HOST', 'http://localhost:8000')
AUTODL_API_KEY = os.getenv('AUTODL_API_KEY', 'autodl-tts-secret-key-2024')
SPK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spk')
MODEL_DIR = os.getenv('MODEL_DIR', 'pretrained_models/Fun-CosyVoice3-0.5B')

# Import utilities
from utils.utils import split_into_sentences, split_into_sentences_en
from utils.audio import merge_audio_and_generate_subtitles
from utils.normalize import smartread_text_normalize
from utils.s3 import S3


class BookDeletedException(Exception):
    """Raised when the book has been deleted (API returns 404)."""
    pass

# CosyVoice imports
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel


# Global CosyVoice instance
_cosyvoice_instance = None
_spk_name = None


def init_cosyvoice(default_spk_name):
    """Initialize CosyVoice model and load all speakers from spk/ directory (singleton)"""
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

    # Load all .pt files from spk/ directory
    spk_files = sorted(glob.glob(os.path.join(SPK_DIR, '*.pt')))
    if not spk_files:
        raise RuntimeError(f"No .pt files found in {SPK_DIR}")

    for spk_path in spk_files:
        spk_name = os.path.splitext(os.path.basename(spk_path))[0]
        print(f"Loading speaker: {spk_name} from {spk_path}...")
        spk2info = torch.load(spk_path, map_location='cpu')
        info = list(spk2info.values())[0]
        _cosyvoice_instance.frontend.spk2info[spk_name] = info

    _spk_name = default_spk_name
    if _spk_name not in _cosyvoice_instance.frontend.spk2info:
        raise RuntimeError(f"Speaker '{_spk_name}' not found. Available: {list(_cosyvoice_instance.frontend.spk2info.keys())}")

    print(f"Default speaker: {_spk_name}")
    print(f"All speakers: {list(_cosyvoice_instance.frontend.spk2info.keys())}")

    return _cosyvoice_instance, _spk_name


def cosyvoice_tts(text: str, output_path: str) -> bool:
    """
    Generate TTS audio for given text using CosyVoice3
    
    Args:
        text: Text to synthesize
        output_path: Path to save the generated wav file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cosyvoice, spk_name = _cosyvoice_instance, _spk_name
        text = smartread_text_normalize(text)

        # Generate speech using pre-loaded speaker info
        for i, result in enumerate(cosyvoice.inference_zero_shot(
            text,
            'You are a helpful assistant.<|endofprompt|>',
            '',  # Empty prompt_wav since we use cached spk_info
            zero_shot_spk_id=spk_name,
            stream=False
        )):
            torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
            break  # Only need first result
        
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False


def fetch_book_content(book_id: str, force: bool = False) -> dict:
    """
    Fetch book content from external API

    Args:
        book_id: The book ID to fetch
        force: If True, include COMPLETED chapters for re-processing

    Returns:
        Book content data dict
    """
    url = f"{EXTERNAL_API_HOST}/autodl/book/{book_id}/content"
    if force:
        url += "?force=true"
    headers = {
        "X-AutoDL-API-Key": AUTODL_API_KEY
    }
    
    print(f"Fetching book content from {url}...")
    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        raise BookDeletedException(f"Book {book_id} not found (deleted?)")
    response.raise_for_status()
    
    return response.json()


def submit_chapter_audio(chapter_id: str, s3_key: str, audio_duration: int, caption_data: str, voice_id: str = None) -> bool:
    """
    Submit chapter audio data to external API

    Args:
        chapter_id: Book version chapter ID
        s3_key: S3 key of the uploaded audio
        audio_duration: Audio duration in seconds
        caption_data: JSON string of caption data
        voice_id: Voice UUID for multi-voice support

    Returns:
        True if successful, False otherwise
    """
    url = f"{EXTERNAL_API_HOST}/autodl/chapter/audio"
    headers = {
        "X-AutoDL-API-Key": AUTODL_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "book_version_chapter_id": chapter_id,
        "s3_key": s3_key,
        "audio_duration": audio_duration,
        "caption_data": caption_data,
    }
    if voice_id:
        payload["voice_id"] = voice_id

    print(f"Submitting audio for chapter {chapter_id} (voice_id={voice_id})...")
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    print(f"Submit result: {result.get('message', 'success')}")
    return result.get('success', False)


def process_chapter(book_id: str, chapter_data: dict, version_id: str, lang: str = 'zh', s3: S3 = None, voice_id: str = None) -> bool:
    """
    Process a single chapter: split text, generate TTS, merge audio, upload

    Args:
        book_id: The book ID
        chapter_data: Chapter content data
        version_id: Book version ID
        voice_id: Voice UUID for multi-voice support

    Returns:
        True if successful, False otherwise
    """
    chapter_id = chapter_data['chapter_id']
    chapter_title = chapter_data['title']
    content = chapter_data['content']
    
    print(f"\n{'='*60}")
    print(f"Processing chapter: {chapter_title} (ID: {chapter_id})")
    print(f"length: {len(content)}")
    print(f"{'='*60}")
    
    # Create audio directory
    audio_dir = f"output/{book_id}/{version_id}/{chapter_id}"
    os.makedirs(audio_dir, exist_ok=True)
    
    text_index = {}
    sentence_index = 1
    
    # Try to parse content as JSON (sub_chapters_data)
    try:
        sub_chapters = json.loads(content) if content else None
        if isinstance(sub_chapters, list) and len(sub_chapters) > 0:
            # Has sub-chapters data
            print(f"Processing {len(sub_chapters)} sub-chapters")
            
            for sub_chapter in sub_chapters:
                sub_chapter_id = sub_chapter.get("id")
                sub_chapter_title = sub_chapter.get("title", "")
                sub_summary = sub_chapter.get("summary", "")
                
                if sub_summary:
                    # Add title as a sentence
                    text_index[str(sentence_index)] = {
                        "text": sub_chapter_title,
                        "sub_chapter_id": sub_chapter_id,
                        "sub_chapter_title": sub_chapter_title,
                        "is_title": True
                    }
                    sentence_index += 1
                    
                    # Split summary into sentences
                    split_fn = split_into_sentences_en if lang == 'en' else split_into_sentences
                    sub_sentences = split_fn(sub_summary)
                    print(f"Sub-chapter '{sub_chapter_title}': {len(sub_sentences)} sentences")
                    
                    for sentence in sub_sentences:
                        sentence = sentence.strip()
                        if lang == 'zh':
                            sentence = sentence.replace("\n", "").replace(" ", "")
                        text_index[str(sentence_index)] = {
                            "text": sentence,
                            "sub_chapter_id": sub_chapter_id,
                            "sub_chapter_title": sub_chapter_title
                        }
                        sentence_index += 1
            
            print(f"Total sentences across all sub-chapters: {sentence_index - 1}")
        else:
            raise ValueError("Not a valid sub-chapters list")
            
    except (json.JSONDecodeError, ValueError, TypeError):
        # No sub-chapters, use content as summary
        if content:
            split_fn = split_into_sentences_en if lang == 'en' else split_into_sentences
            sentences = split_fn(content)
            print(f"Split chapter content into {len(sentences)} sentences")
            
            for i, sentence in enumerate(sentences, 1):
                text_index[str(i)] = {
                    "text": sentence,
                    "sub_chapter_id": None,
                    "sub_chapter_title": None
                }
        else:
            print(f"Warning: No content for chapter {chapter_id}")
            return False
    
    if not text_index:
        print(f"Warning: No text to process for chapter {chapter_id}")
        return False
    
    # Generate audio for each sentence
    total_sentences = len(text_index)
    for i in range(1, total_sentences + 1):
        text_item = text_index[str(i)]
        sentence = text_item["text"]
        audio_path = f"{audio_dir}/{i}.wav"

        print(f"Generating audio for sentence {i}/{total_sentences}: {len(sentence)} {sentence}...")

        success = cosyvoice_tts(sentence, audio_path)

        if not success:
            print(f"Failed to generate audio for sentence {i}, removing from text_index")
            del text_index[str(i)]
            continue

    # Save text_index.json (after removing any failed entries)
    with open(f'{audio_dir}/text_index.json', 'w', encoding='utf-8') as f:
        json.dump(text_index, f, ensure_ascii=False, indent=4)

    # Merge audio and generate subtitles
    audio_duration = merge_audio_and_generate_subtitles(audio_dir)
    
    if audio_duration is None:
        print(f"Failed to merge audio for chapter {chapter_id}")
        return False
    
    # Upload to S3 (voice_id 区分路径避免覆盖)
    if s3 is None:
        s3 = S3()
    merged_audio_path = f"{audio_dir}/merged.mp3"
    if voice_id:
        s3_url, s3_key = s3.upload_audio(
            merged_audio_path, book_id, f"{chapter_id}/{voice_id}"
        )
    else:
        s3_url, s3_key = s3.upload_audio(merged_audio_path, book_id, chapter_id)
    print(f"Uploaded to S3: {s3_url}")

    # Read caption data
    caption_data = "[]"
    data_path = f"{audio_dir}/data.json"
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            caption_data = json.dumps(json.load(f), ensure_ascii=False)

    # Submit to external API
    success = submit_chapter_audio(
        chapter_id=chapter_id,
        s3_key=s3_key,
        audio_duration=audio_duration,
        caption_data=caption_data,
        voice_id=voice_id,
    )
    
    if success:
        print(f"Cleaning up audio files for chapter {chapter_id}...")
        try:
            # 删除合成前的音频片段
            for wav_path in glob.glob(f"{audio_dir}/*.wav"):
                os.remove(wav_path)
            # 删除合成后的音频
            if os.path.exists(merged_audio_path):
                os.remove(merged_audio_path)
            print("Cleanup completed.")
        except Exception as e:
            print(f"Failed to clean up audio files: {e}")
            
    return success


def _version_matches_modes(version, allowed_modes):
    """Check if a version matches the allowed modes filter."""
    mode = version.get('mode', '')
    ratio = version.get('ratio', 0)
    if mode == 'summarize':
        return f"summarize_{ratio}" in allowed_modes
    return mode in allowed_modes


def main():
    parser = argparse.ArgumentParser(description='TTS Book Processing Script')
    parser.add_argument('--book_id', type=str, required=True, help='Book ID to process')
    parser.add_argument('--spk', type=str, required=True, help='Speaker name (e.g. cn_male_1), resolved to spk/<name>.pt')
    parser.add_argument('--lang', type=str, default='zh', choices=['zh', 'en'], help='Language for text splitting: zh (default) or en')
    parser.add_argument('--force', action='store_true', default=False, help='Force re-process completed chapters (silent update)')
    parser.add_argument('--modes', type=str, default=None, help='Comma-separated modes to process, e.g. "summarize_20,guide"')
    parser.add_argument('--voice_id', type=str, default=None, help='Voice UUID for multi-voice audio callback')
    parser.add_argument('--host', type=str, default=None, help='Override EXTERNAL_API_HOST from .env')
    args = parser.parse_args()

    global EXTERNAL_API_HOST
    if args.host:
        EXTERNAL_API_HOST = args.host

    book_id = args.book_id
    voice_id = args.voice_id

    print(f"Speaker: {args.spk}, Language: {args.lang}, Voice ID: {voice_id}")
    print(f"Starting TTS processing for book: {book_id}")

    # Initialize CosyVoice model and load all speakers
    init_cosyvoice(default_spk_name=args.spk)

    import time
    s3 = S3()

    # Force mode: fetch once, process, exit
    if args.force:
        print("Force mode: processing completed chapters for re-generation...")
        try:
            book_data = fetch_book_content(book_id, force=True)
        except Exception as e:
            print(f"Failed to fetch book content in force mode: {e}")
            return

        versions = book_data.get('versions', [])
        if args.modes:
            allowed = set(args.modes.split(","))
            versions = [v for v in versions if _version_matches_modes(v, allowed)]
            print(f"Filtered to modes {allowed}: {len(versions)} versions")

        print(f"Book title: {book_data.get('title', 'Unknown')}")
        print(f"Number of versions to process: {len(versions)}")

        for version in versions:
            version_id = version['version_id']
            mode = version.get('mode', 'unknown')
            ratio = version.get('ratio', 0)
            print(f"\n{'#'*60}")
            print(f"Processing version: {version_id} (mode={mode}, ratio={ratio}%)")
            print(f"{'#'*60}")
            for chapter in version.get('version_chapters', []):
                try:
                    process_chapter(book_id, chapter, version_id, lang=args.lang, s3=s3, voice_id=voice_id)
                except Exception as e:
                    print(f"Error processing chapter {chapter.get('chapter_id')}: {e}")
                    continue

        print("\nForce mode processing completed!")
        return

    poll_interval = 60  # 轮询间隔（秒）

    while True:
        try:
            book_data = fetch_book_content(book_id)
        except BookDeletedException as e:
            print(f"Book deleted: {e}. Stopping container.")
            break
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch book content: {e}, retrying in {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        versions = book_data.get('versions', [])
        content_complete = book_data.get('content_generation_complete', False)

        if not versions:
            if content_complete:
                print("All versions processed and content generation complete. Exiting.")
                break
            else:
                print(f"No versions ready yet, waiting {poll_interval}s...")
                time.sleep(poll_interval)
                continue

        print(f"Book title: {book_data.get('title', 'Unknown')}")
        print(f"Number of versions to process: {len(versions)}")

        # Process available versions
        for version in versions:
            version_id = version['version_id']
            mode = version.get('mode', 'unknown')
            ratio = version.get('ratio', 0)

            print(f"\n{'#'*60}")
            print(f"Processing version: {version_id} (mode={mode}, ratio={ratio}%)")
            print(f"{'#'*60}")

            for chapter in version.get('version_chapters', []):
                try:
                    process_chapter(book_id, chapter, version_id, lang=args.lang, s3=s3, voice_id=voice_id)
                except Exception as e:
                    print(f"Error processing chapter {chapter.get('chapter_id')}: {e}")
                    continue

        # 处理完当前批次后，如果内容已全部生成，再检查一次是否还有剩余
        if content_complete:
            print("Content generation complete, doing final check...")
            try:
                final_data = fetch_book_content(book_id)
                if not final_data.get('versions', []):
                    print("All done. Exiting.")
                    break
            except BookDeletedException:
                print("Book deleted during final check. Exiting.")
                break
            except requests.exceptions.RequestException:
                print("Final check failed, exiting anyway.")
                break

    print("\n" + "="*60)
    print("TTS processing completed!")
    print("="*60)


if __name__ == '__main__':
    main()
