#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS Worker Processor

Book/chapter processing logic. Calls the TTS API at :6006 for audio generation
instead of loading CosyVoice models directly.
"""
import os
import sys
import json
import glob
import asyncio
from datetime import datetime
from dataclasses import dataclass, field

# Add project root to sys.path so utils.* imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import requests

from utils.utils import split_into_sentences, split_into_sentences_en
from utils.audio import merge_audio_and_generate_subtitles
from utils.s3 import S3


@dataclass
class TaskState:
    task_id: str
    book_id: str
    status: str = "pending"  # pending, processing, completed, failed
    chapters_completed: int = 0
    chapters_total: int = 0
    chapters_failed: int = 0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def progress(self) -> str:
        if self.chapters_total == 0:
            return "0/0 chapters"
        return f"{self.chapters_completed}/{self.chapters_total} chapters"

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.updated_at = datetime.now().isoformat()


class BookDeletedException(Exception):
    """Raised when the book has been deleted (API returns 404)."""
    pass


# ---------------------------------------------------------------------------
# TTS HTTP Client
# ---------------------------------------------------------------------------

def generate_tts(
    text: str,
    output_path: str,
    *,
    tts_api_url: str,
    api_key: str,
    spk: str | None = None,
) -> bool:
    """
    Call the TTS API to generate audio for a single sentence.

    The TTS API already applies smartread_text_normalize, so we send raw text.
    """
    try:
        payload = {"text": text, "stream": False}
        if spk:
            payload["spk"] = spk

        resp = requests.post(
            f"{tts_api_url}/tts",
            json=payload,
            headers={"X-AutoDL-API-Key": api_key},
            timeout=120,
        )
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(resp.content)

        return True
    except Exception as e:
        print(f"TTS API Error: {e}")
        return False


# ---------------------------------------------------------------------------
# External API helpers (mirrored from run.py)
# ---------------------------------------------------------------------------

def fetch_book_content(
    book_id: str,
    *,
    external_api_host: str,
    api_key: str,
    force: bool = False,
) -> dict:
    url = f"{external_api_host}/autodl/book/{book_id}/content"
    if force:
        url += "?force=true"

    print(f"Fetching book content from {url}...")
    resp = requests.get(url, headers={"X-AutoDL-API-Key": api_key})
    if resp.status_code == 404:
        raise BookDeletedException(f"Book {book_id} not found (deleted?)")
    resp.raise_for_status()
    return resp.json()


def submit_chapter_audio(
    chapter_id: str,
    s3_key: str,
    audio_duration: int,
    caption_data: str,
    *,
    external_api_host: str,
    api_key: str,
) -> bool:
    url = f"{external_api_host}/autodl/chapter/audio"
    payload = {
        "book_version_chapter_id": chapter_id,
        "s3_key": s3_key,
        "audio_duration": audio_duration,
        "caption_data": caption_data,
    }

    print(f"Submitting audio for chapter {chapter_id}...")
    resp = requests.post(
        url,
        json=payload,
        headers={
            "X-AutoDL-API-Key": api_key,
            "Content-Type": "application/json",
        },
    )
    resp.raise_for_status()

    result = resp.json()
    print(f"Submit result: {result.get('message', 'success')}")
    return result.get("success", False)


# ---------------------------------------------------------------------------
# Chapter processing (mirrors run.py process_chapter)
# ---------------------------------------------------------------------------

def process_chapter(
    book_id: str,
    chapter_data: dict,
    version_id: str,
    *,
    lang: str = "zh",
    spk: str | None = None,
    s3: S3,
    tts_api_url: str,
    external_api_host: str,
    api_key: str,
) -> bool:
    chapter_id = chapter_data["chapter_id"]
    chapter_title = chapter_data["title"]
    content = chapter_data["content"]

    print(f"\n{'='*60}")
    print(f"Processing chapter: {chapter_title} (ID: {chapter_id})")
    print(f"length: {len(content)}")
    print(f"{'='*60}")

    audio_dir = f"output/{book_id}/{version_id}/{chapter_id}"
    os.makedirs(audio_dir, exist_ok=True)

    text_index = {}
    sentence_index = 1

    # Try to parse content as JSON (sub_chapters_data)
    try:
        sub_chapters = json.loads(content) if content else None
        if isinstance(sub_chapters, list) and len(sub_chapters) > 0:
            print(f"Processing {len(sub_chapters)} sub-chapters")

            for sub_chapter in sub_chapters:
                sub_chapter_id = sub_chapter.get("id")
                sub_chapter_title = sub_chapter.get("title", "")
                sub_summary = sub_chapter.get("summary", "")

                if sub_summary:
                    text_index[str(sentence_index)] = {
                        "text": sub_chapter_title,
                        "sub_chapter_id": sub_chapter_id,
                        "sub_chapter_title": sub_chapter_title,
                        "is_title": True,
                    }
                    sentence_index += 1

                    split_fn = split_into_sentences_en if lang == "en" else split_into_sentences
                    sub_sentences = split_fn(sub_summary)
                    print(f"Sub-chapter '{sub_chapter_title}': {len(sub_sentences)} sentences")

                    for sentence in sub_sentences:
                        sentence = sentence.strip()
                        if lang == "zh":
                            sentence = sentence.replace("\n", "").replace(" ", "")
                        text_index[str(sentence_index)] = {
                            "text": sentence,
                            "sub_chapter_id": sub_chapter_id,
                            "sub_chapter_title": sub_chapter_title,
                        }
                        sentence_index += 1

            print(f"Total sentences across all sub-chapters: {sentence_index - 1}")
        else:
            raise ValueError("Not a valid sub-chapters list")

    except (json.JSONDecodeError, ValueError, TypeError):
        if content:
            split_fn = split_into_sentences_en if lang == "en" else split_into_sentences
            sentences = split_fn(content)
            print(f"Split chapter content into {len(sentences)} sentences")

            for i, sentence in enumerate(sentences, 1):
                text_index[str(i)] = {
                    "text": sentence,
                    "sub_chapter_id": None,
                    "sub_chapter_title": None,
                }
        else:
            print(f"Warning: No content for chapter {chapter_id}")
            return False

    if not text_index:
        print(f"Warning: No text to process for chapter {chapter_id}")
        return False

    # Generate audio for each sentence via TTS API
    total_sentences = len(text_index)
    for i in range(1, total_sentences + 1):
        key = str(i)
        if key not in text_index:
            continue
        text_item = text_index[key]
        sentence = text_item["text"]
        audio_path = f"{audio_dir}/{i}.wav"

        print(f"Generating audio for sentence {i}/{total_sentences}: {len(sentence)} {sentence}...")

        success = generate_tts(
            sentence,
            audio_path,
            tts_api_url=tts_api_url,
            api_key=api_key,
            spk=spk,
        )

        if not success:
            print(f"Failed to generate audio for sentence {i}, removing from text_index")
            del text_index[key]
            continue

    # Save text_index.json
    with open(f"{audio_dir}/text_index.json", "w", encoding="utf-8") as f:
        json.dump(text_index, f, ensure_ascii=False, indent=4)

    # Merge audio and generate subtitles
    audio_duration = merge_audio_and_generate_subtitles(audio_dir)

    if audio_duration is None:
        print(f"Failed to merge audio for chapter {chapter_id}")
        return False

    # Upload to S3
    merged_audio_path = f"{audio_dir}/merged.mp3"
    s3_url, s3_key = s3.upload_audio(merged_audio_path, book_id, chapter_id)
    print(f"Uploaded to S3: {s3_url}")

    # Read caption data
    caption_data = "[]"
    data_path = f"{audio_dir}/data.json"
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            caption_data = json.dumps(json.load(f), ensure_ascii=False)

    # Submit to external API
    success = submit_chapter_audio(
        chapter_id=chapter_id,
        s3_key=s3_key,
        audio_duration=audio_duration,
        caption_data=caption_data,
        external_api_host=external_api_host,
        api_key=api_key,
    )

    if success:
        print(f"Cleaning up audio files for chapter {chapter_id}...")
        try:
            for wav_path in glob.glob(f"{audio_dir}/*.wav"):
                os.remove(wav_path)
            if os.path.exists(merged_audio_path):
                os.remove(merged_audio_path)
            print("Cleanup completed.")
        except Exception as e:
            print(f"Failed to clean up audio files: {e}")

    return success


# ---------------------------------------------------------------------------
# Version filtering (from run.py)
# ---------------------------------------------------------------------------

def _version_matches_modes(version: dict, allowed_modes: set) -> bool:
    mode = version.get("mode", "")
    ratio = version.get("ratio", 0)
    if mode == "summarize":
        return f"summarize_{ratio}" in allowed_modes
    return mode in allowed_modes


# ---------------------------------------------------------------------------
# Book-level processing (background entry point)
# ---------------------------------------------------------------------------

async def process_book(
    task: TaskState,
    *,
    lang: str = "zh",
    force: bool = False,
    modes: str | None = None,
    spk: str | None = None,
    tts_api_url: str,
    external_api_host: str,
    api_key: str,
):
    """Main background task: fetch book, process all chapters."""
    book_id = task.book_id
    task.update(status="processing")

    try:
        book_data = await asyncio.to_thread(
            fetch_book_content,
            book_id,
            external_api_host=external_api_host,
            api_key=api_key,
            force=force,
        )
    except BookDeletedException as e:
        task.update(status="failed", error=str(e))
        return
    except Exception as e:
        task.update(status="failed", error=f"Failed to fetch book content: {e}")
        return

    versions = book_data.get("versions", [])
    if modes:
        allowed = set(modes.split(","))
        versions = [v for v in versions if _version_matches_modes(v, allowed)]
        print(f"Filtered to modes {allowed}: {len(versions)} versions")

    print(f"Book title: {book_data.get('title', 'Unknown')}")
    print(f"Number of versions to process: {len(versions)}")

    # Count total chapters
    total_chapters = sum(len(v.get("version_chapters", [])) for v in versions)
    task.update(chapters_total=total_chapters)

    s3 = S3()

    for version in versions:
        version_id = version["version_id"]
        mode = version.get("mode", "unknown")
        ratio = version.get("ratio", 0)

        print(f"\n{'#'*60}")
        print(f"Processing version: {version_id} (mode={mode}, ratio={ratio}%)")
        print(f"{'#'*60}")

        for chapter in version.get("version_chapters", []):
            try:
                success = await asyncio.to_thread(
                    process_chapter,
                    book_id,
                    chapter,
                    version_id,
                    lang=lang,
                    spk=spk,
                    s3=s3,
                    tts_api_url=tts_api_url,
                    external_api_host=external_api_host,
                    api_key=api_key,
                )
                if success:
                    task.update(chapters_completed=task.chapters_completed + 1)
                else:
                    task.update(chapters_failed=task.chapters_failed + 1)
            except Exception as e:
                print(f"Error processing chapter {chapter.get('chapter_id')}: {e}")
                task.update(chapters_failed=task.chapters_failed + 1)
                continue

    task.update(status="completed")
    print(f"\nBook processing completed! {task.progress}")
