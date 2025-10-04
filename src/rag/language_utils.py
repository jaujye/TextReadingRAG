"""Language detection and text processing utilities for multilingual RAG support."""

import re
from typing import List, Tuple

import jieba
from langdetect import detect, LangDetectException

from src.core.config import settings


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Args:
        text: Input text to analyze

    Returns:
        Language code ('en', 'zh', etc.)
    """
    if not text or not text.strip():
        return settings.rag.default_language

    # For longer texts, use character-based detection (more reliable)
    # For shorter texts, use langdetect
    text_length = len([c for c in text if not c.isspace()])

    if text_length >= 50:
        # Longer text: prioritize character-based detection
        if is_chinese(text):
            return 'zh'

    # Use langdetect as primary or fallback method
    try:
        lang = detect(text)
        # Map langdetect codes to our supported languages
        if lang.startswith('zh') or lang == 'ko':  # langdetect sometimes confuses Chinese with Korean
            # Double-check with character pattern
            if is_chinese(text):
                return 'zh'
        if lang == 'en':
            return 'en'
        # For other languages, return default
        return settings.rag.default_language
    except LangDetectException:
        # If langdetect fails, try character-based detection
        if is_chinese(text):
            return 'zh'
        return settings.rag.default_language


def is_chinese(text: str) -> bool:
    """
    Check if text contains significant Chinese characters.

    Args:
        text: Input text to check

    Returns:
        True if text is primarily Chinese
    """
    if not text:
        return False

    # Count CJK characters (Common Chinese ranges)
    # \u4e00-\u9fff: CJK Unified Ideographs
    # \u3400-\u4dbf: CJK Extension A
    # \uf900-\ufaff: CJK Compatibility Ideographs
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
    cjk_chars = len(cjk_pattern.findall(text))
    total_chars = len([c for c in text if not c.isspace()])

    if total_chars == 0:
        return False

    # Consider text Chinese if >30% CJK characters
    return (cjk_chars / total_chars) > 0.3


def tokenize_chinese(text: str) -> List[str]:
    """
    Tokenize Chinese text using jieba.

    Args:
        text: Chinese text to tokenize

    Returns:
        List of word tokens
    """
    return list(jieba.cut(text))


def get_sentence_boundaries_chinese(text: str) -> List[int]:
    """
    Find sentence boundaries in Chinese text based on punctuation.

    Args:
        text: Chinese text

    Returns:
        List of character positions where sentences end
    """
    # Chinese sentence-ending punctuation
    pattern = r'[。！？\n]'
    boundaries = [m.end() for m in re.finditer(pattern, text)]
    return boundaries


def split_chinese_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[str]:
    """
    Split Chinese text into chunks based on semantic boundaries.

    Args:
        text: Chinese text to split
        chunk_size: Target chunk size in characters (uses config default if None)
        chunk_overlap: Overlap size in characters (uses config default if None)

    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = settings.rag.chinese_chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.rag.chinese_chunk_overlap

    # Get sentence boundaries
    boundaries = get_sentence_boundaries_chinese(text)

    if not boundaries:
        # No sentence boundaries, fall back to character-based splitting
        return _split_by_chars(text, chunk_size, chunk_overlap)

    # Add start and end positions
    boundaries = [0] + boundaries
    if boundaries[-1] < len(text):
        boundaries.append(len(text))

    # Group sentences into chunks
    chunks = []
    current_chunk_start = 0
    current_chunk_end = 0

    for i in range(1, len(boundaries)):
        sentence_start = boundaries[i - 1]
        sentence_end = boundaries[i]
        sentence = text[sentence_start:sentence_end]

        potential_end = sentence_end
        potential_chunk = text[current_chunk_start:potential_end]

        if len(potential_chunk) <= chunk_size:
            current_chunk_end = potential_end
        else:
            # Current chunk is full, save it
            if current_chunk_end > current_chunk_start:
                chunks.append(text[current_chunk_start:current_chunk_end])
                # Start new chunk with overlap
                current_chunk_start = max(current_chunk_start, current_chunk_end - chunk_overlap)
                current_chunk_end = sentence_end
            else:
                # Single sentence exceeds chunk size, add it anyway
                chunks.append(sentence)
                current_chunk_start = sentence_end
                current_chunk_end = sentence_end

    # Add final chunk
    if current_chunk_end > current_chunk_start:
        chunks.append(text[current_chunk_start:current_chunk_end])

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fall back to simple character-based splitting."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def split_text_by_language(
    text: str,
    language: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> Tuple[List[str], str]:
    """
    Split text into chunks using language-appropriate method.

    Args:
        text: Text to split
        language: Language code (auto-detected if None)
        chunk_size: Target chunk size
        chunk_overlap: Overlap size

    Returns:
        Tuple of (list of chunks, detected language)
    """
    # Detect language if not provided
    if language is None and settings.rag.enable_language_detection:
        language = detect_language(text)
    elif language is None:
        language = settings.rag.default_language

    # Use Chinese-specific splitting for Chinese text
    if language == 'zh':
        chunks = split_chinese_text(text, chunk_size, chunk_overlap)
    else:
        # For English and other languages, fall back to character-based
        # (LlamaIndex's SentenceSplitter will be used in ingestion.py)
        if chunk_size is None:
            chunk_size = settings.rag.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.rag.chunk_overlap
        chunks = _split_by_chars(text, chunk_size, chunk_overlap)

    return chunks, language
