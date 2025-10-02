"""Tests for Traditional Chinese language support in RAG system."""

import pytest
from unittest.mock import Mock, patch

from src.rag.language_utils import (
    detect_language,
    is_chinese,
    tokenize_chinese,
    split_chinese_text,
    get_sentence_boundaries_chinese,
)
from src.rag.retrieval import multilingual_tokenizer


class TestLanguageDetection:
    """Test language detection functionality."""

    def test_detect_english(self):
        """Test detection of English text."""
        text = "This is a test document in English."
        assert detect_language(text) == 'en'

    def test_detect_chinese(self):
        """Test detection of Traditional Chinese text."""
        text = "這是一份繁體中文的測試文檔。"
        lang = detect_language(text)
        assert lang == 'zh'

    def test_detect_simplified_chinese(self):
        """Test detection of Simplified Chinese text."""
        text = "这是一份简体中文的测试文档。"
        lang = detect_language(text)
        assert lang == 'zh'

    def test_detect_empty_string(self):
        """Test detection of empty string."""
        text = ""
        lang = detect_language(text)
        assert lang == 'en'  # Should return default language

    def test_is_chinese_traditional(self):
        """Test Chinese text detection for Traditional Chinese."""
        text = "繁體中文測試文檔內容"
        assert is_chinese(text) is True

    def test_is_chinese_simplified(self):
        """Test Chinese text detection for Simplified Chinese."""
        text = "简体中文测试文档内容"
        assert is_chinese(text) is True

    def test_is_chinese_english(self):
        """Test Chinese text detection for English text."""
        text = "This is English text"
        assert is_chinese(text) is False

    def test_is_chinese_mixed(self):
        """Test Chinese text detection for mixed text."""
        text = "This is a mix 這是混合文本 of languages"
        # Mixed text with >30% Chinese should be detected as Chinese
        assert is_chinese(text) is True


class TestChineseTokenization:
    """Test Chinese tokenization functionality."""

    def test_tokenize_chinese_simple(self):
        """Test basic Chinese tokenization."""
        text = "我喜歡閱讀書籍"
        tokens = tokenize_chinese(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Jieba should split into words
        assert all(isinstance(token, str) for token in tokens)

    def test_tokenize_chinese_with_punctuation(self):
        """Test Chinese tokenization with punctuation."""
        text = "你好，世界！這是測試。"
        tokens = tokenize_chinese(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_chinese_mixed_content(self):
        """Test tokenization of mixed Chinese and English."""
        text = "我喜歡Python編程"
        tokens = tokenize_chinese(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0


class TestChineseSentenceBoundaries:
    """Test Chinese sentence boundary detection."""

    def test_sentence_boundaries_single_sentence(self):
        """Test boundary detection for single sentence."""
        text = "這是一個句子。"
        boundaries = get_sentence_boundaries_chinese(text)
        assert len(boundaries) == 1
        assert boundaries[0] == len(text)

    def test_sentence_boundaries_multiple_sentences(self):
        """Test boundary detection for multiple sentences."""
        text = "這是第一句。這是第二句！這是第三句？"
        boundaries = get_sentence_boundaries_chinese(text)
        assert len(boundaries) == 3

    def test_sentence_boundaries_with_newlines(self):
        """Test boundary detection with newlines."""
        text = "第一行\n第二行\n第三行"
        boundaries = get_sentence_boundaries_chinese(text)
        assert len(boundaries) == 3

    def test_sentence_boundaries_no_punctuation(self):
        """Test boundary detection without punctuation."""
        text = "沒有標點符號的文本"
        boundaries = get_sentence_boundaries_chinese(text)
        assert len(boundaries) == 0


class TestChineseTextSplitting:
    """Test Chinese text splitting functionality."""

    def test_split_chinese_text_basic(self):
        """Test basic Chinese text splitting."""
        text = "這是一份測試文檔。" * 20  # Create longer text
        chunks = split_chinese_text(text, chunk_size=100, chunk_overlap=20)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some flexibility

    def test_split_chinese_text_with_overlap(self):
        """Test Chinese text splitting with overlap."""
        text = "第一句話。" * 10 + "第二句話。" * 10
        chunks = split_chinese_text(text, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1
        # Check that chunks have reasonable sizes
        for chunk in chunks:
            assert len(chunk) > 0

    def test_split_chinese_text_short_text(self):
        """Test splitting of short Chinese text."""
        text = "很短的文本。"
        chunks = split_chinese_text(text, chunk_size=100, chunk_overlap=20)

        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_split_chinese_text_long_sentence(self):
        """Test splitting when single sentence exceeds chunk size."""
        text = "這" * 300 + "。"  # Single very long sentence
        chunks = split_chinese_text(text, chunk_size=100, chunk_overlap=20)

        # Should still create chunks even if sentence is too long
        assert len(chunks) > 0

    def test_split_chinese_text_realistic(self):
        """Test splitting with realistic Traditional Chinese content."""
        text = """
        繁體中文是中文的一種書寫系統。它主要在臺灣、香港和澳門使用。
        與簡體中文相比，繁體中文保留了更多的傳統字形。
        這種文字系統有著悠久的歷史和深厚的文化底蘊。
        許多古典文學作品都是用繁體中文書寫的。
        學習繁體中文可以幫助我們更好地理解中華文化。
        """
        chunks = split_chinese_text(text.strip(), chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 0
        # All chunks should be non-empty
        assert all(chunk.strip() for chunk in chunks)
        # Chunks should not be too long
        assert all(len(chunk) <= 120 for chunk in chunks)


class TestMultilingualTokenizer:
    """Test multilingual tokenizer for BM25."""

    def test_multilingual_tokenizer_english(self):
        """Test tokenizer with English text."""
        text = "This is a test document"
        tokens = multilingual_tokenizer(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # English should be split by whitespace
        assert "test" in tokens or "this" in tokens

    def test_multilingual_tokenizer_chinese(self):
        """Test tokenizer with Chinese text."""
        text = "這是一個測試文檔"
        tokens = multilingual_tokenizer(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should use jieba tokenization

    def test_multilingual_tokenizer_mixed(self):
        """Test tokenizer with mixed content."""
        text = "這是Python測試"
        tokens = multilingual_tokenizer(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0


class TestIntegration:
    """Integration tests for Chinese RAG support."""

    def test_language_detection_and_splitting_workflow(self):
        """Test complete workflow of language detection and splitting."""
        # Traditional Chinese text
        text = """
        人工智能技術正在快速發展。機器學習和深度學習已經成為主流技術。
        自然語言處理是人工智能的重要分支。它可以幫助計算機理解和生成人類語言。
        """

        # Detect language
        lang = detect_language(text)
        assert lang == 'zh'

        # Split text
        chunks = split_chinese_text(text.strip(), chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 0

        # Tokenize first chunk
        if chunks:
            tokens = tokenize_chinese(chunks[0])
            assert len(tokens) > 0

    def test_english_vs_chinese_processing(self):
        """Test that English and Chinese are processed differently."""
        english_text = "This is an English document. It has multiple sentences. Each sentence should be processed correctly."
        chinese_text = "這是中文文檔。它有多個句子。每個句子都應該正確處理。"

        # Detect languages
        assert detect_language(english_text) == 'en'
        assert detect_language(chinese_text) == 'zh'

        # Check text classification
        assert not is_chinese(english_text)
        assert is_chinese(chinese_text)

        # Both should produce chunks
        en_chunks = split_chinese_text(english_text, chunk_size=100, chunk_overlap=20)
        zh_chunks = split_chinese_text(chinese_text, chunk_size=100, chunk_overlap=20)

        assert len(en_chunks) > 0
        assert len(zh_chunks) > 0


@pytest.mark.parametrize("text,expected_lang", [
    ("Hello world", "en"),
    ("你好世界", "zh"),
    ("這是繁體中文", "zh"),
    ("This is English", "en"),
])
def test_language_detection_parametrized(text, expected_lang):
    """Parametrized test for language detection."""
    assert detect_language(text) == expected_lang


@pytest.mark.parametrize("text,is_chinese_expected", [
    ("完全中文文本", True),
    ("Completely English text", False),
    ("中文English混合Mix", True),  # >30% Chinese
    ("Mix with 少量中文", False),  # <30% Chinese
])
def test_is_chinese_parametrized(text, is_chinese_expected):
    """Parametrized test for Chinese text detection."""
    assert is_chinese(text) == is_chinese_expected
