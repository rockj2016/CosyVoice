#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS Book Processing Script

Usage: python run.py --book_id <book_id> --spk <spk_file.pt> [--lang zh|en]

从外部服务获取书本信息，生成 TTS 音频和字幕，并将结果上传回外部服务。
"""
import os
import sys
import json
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
SPK_INFO_PATH = os.getenv('SPK_INFO_PATH', './spkinfo.pt')
MODEL_DIR = os.getenv('MODEL_DIR', 'pretrained_models/Fun-CosyVoice3-0.5B')

# Import utilities
from utils.utils import split_into_sentences, split_into_sentences_en
from utils.audio import merge_audio_and_generate_subtitles
from utils.s3 import S3

# CosyVoice imports
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel


# Global CosyVoice instance
_cosyvoice_instance = None
_spk_name = None


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
        cosyvoice, spk_name = init_cosyvoice()
        
        # Generate speech using pre-loaded speaker info
        for i, result in enumerate(cosyvoice.inference_zero_shot(
            text,
            'You are a helpful assistant.<|endofprompt|>',
            '',  # Empty prompt_wav since we use cached spk_info
            zero_shot_spk_id=spk_name,
            stream=False
        )):
            torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
            # 清除首尾噪声
            from utils.audio import clean_wav_noise
            clean_wav_noise(os.path.abspath(output_path))
            break  # Only need first result
        
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False


def fetch_book_content(book_id: str) -> dict:
    """
    Fetch book content from external API
    
    Args:
        book_id: The book ID to fetch
        
    Returns:
        Book content data dict
    """
    url = f"{EXTERNAL_API_HOST}/autodl/book/{book_id}/content"
    headers = {
        "X-AutoDL-API-Key": AUTODL_API_KEY
    }
    
    print(f"Fetching book content from {url}...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()


def submit_chapter_audio(chapter_id: str, s3_key: str, audio_duration: int, caption_data: str) -> bool:
    """
    Submit chapter audio data to external API
    
    Args:
        chapter_id: Book version chapter ID
        s3_key: S3 key of the uploaded audio
        audio_duration: Audio duration in seconds
        caption_data: JSON string of caption data
        
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
        "caption_data": caption_data
    }
    
    print(f"Submitting audio for chapter {chapter_id}...")
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    print(f"Submit result: {result.get('message', 'success')}")
    return result.get('success', False)


def process_chapter(book_id: str, chapter_data: dict, version_id: str, lang: str = 'zh') -> bool:
    """
    Process a single chapter: split text, generate TTS, merge audio, upload
    
    Args:
        book_id: The book ID
        chapter_data: Chapter content data
        version_id: Book version ID
        
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
    
    # Save text_index.json
    with open(f'{audio_dir}/text_index.json', 'w', encoding='utf-8') as f:
        json.dump(text_index, f, ensure_ascii=False, indent=4)
    
    # Generate audio for each sentence
    for i in range(1, len(text_index) + 1):
        text_item = text_index[str(i)]
        sentence = text_item["text"]
        audio_path = f"{audio_dir}/{i}.wav"
        
        print(f"Generating audio for sentence {i}/{len(text_index)}: {len(sentence)} {sentence}...")
        
        success = cosyvoice_tts(sentence, audio_path)
        
        if not success:
            print(f"Failed to generate audio for sentence {i}")
            continue
    
    # Merge audio and generate subtitles
    audio_duration = merge_audio_and_generate_subtitles(audio_dir)
    
    if audio_duration is None:
        print(f"Failed to merge audio for chapter {chapter_id}")
        return False
    
    # Upload to S3
    s3 = S3()
    merged_audio_path = f"{audio_dir}/merged.mp3"
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
        caption_data=caption_data
    )
    
    if success:
        print(f"Cleaning up audio files for chapter {chapter_id}...")
        try:
            import glob
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


def main():
    global SPK_INFO_PATH
    
    parser = argparse.ArgumentParser(description='TTS Book Processing Script')
    parser.add_argument('--book_id', type=str, required=True, help='Book ID to process')
    parser.add_argument('--spk', type=str, required=True, help='Path to speaker info .pt file')
    parser.add_argument('--lang', type=str, default='zh', choices=['zh', 'en'], help='Language for text splitting: zh (default) or en')
    args = parser.parse_args()
    
    book_id = args.book_id
    
    # Set speaker info path from --spk argument
    SPK_INFO_PATH = args.spk
    print(f"Speaker info: {SPK_INFO_PATH}, Language: {args.lang}")
    print(f"Starting TTS processing for book: {book_id}")
    
    # Fetch book content
    try:
        book_data = fetch_book_content(book_id)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch book content: {e}")
        sys.exit(1)
    
    print(f"Book title: {book_data.get('title', 'Unknown')}")
    print(f"Number of versions: {len(book_data.get('versions', []))}")
    
    # Initialize CosyVoice model (one-time loading)
    init_cosyvoice()
    
    # Process each version and chapter
    for version in book_data.get('versions', []):
        version_id = version['version_id']
        mode = version.get('mode', 'unknown')
        ratio = version.get('ratio', 0)
        
        print(f"\n{'#'*60}")
        print(f"Processing version: {version_id} (mode={mode}, ratio={ratio}%)")
        print(f"{'#'*60}")
        
        for chapter in version.get('version_chapters', []):
            try:
                process_chapter(book_id, chapter, version_id, lang=args.lang)
            except Exception as e:
                print(f"Error processing chapter {chapter.get('chapter_id')}: {e}")
                continue
    
    print("\n" + "="*60)
    print("TTS processing completed!")
    print("="*60)


if __name__ == '__main__':
    main()
