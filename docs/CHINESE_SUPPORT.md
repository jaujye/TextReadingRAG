# Traditional Chinese Language Support

## Overview

The TextReadingRAG system now supports Traditional Chinese (繁體中文) and Simplified Chinese (简体中文) documents alongside English. The system automatically detects the language of documents and queries, applying language-specific processing for optimal results.

## Features

### 1. Automatic Language Detection
- Documents and queries are automatically detected as English or Chinese
- Uses `langdetect` library for accurate language identification
- Falls back to configured default language if detection fails

### 2. Language-Specific Text Processing

#### English Processing
- Space-based tokenization
- Sentence-aware chunking using LlamaIndex SentenceSplitter
- Default chunk size: 512 characters
- WordNet-based synonym expansion

#### Chinese Processing
- **Word Segmentation**: Uses `jieba` for accurate Chinese word tokenization
- **Semantic Chunking**: Respects Chinese sentence boundaries (。！？)
- **Optimized Chunk Sizes**: 256 characters (vs 512 for English) due to information density
- **BM25 Support**: Custom tokenizer for Chinese sparse retrieval
- **Query Expansion**: LLM-based only (WordNet doesn't support Chinese)

### 3. Hybrid Retrieval
The system uses both dense (embedding) and sparse (BM25) retrieval:
- **Dense retrieval**: Uses OpenAI's `text-embedding-3-small` (supports Chinese)
- **Sparse retrieval**: Custom multilingual tokenizer for BM25
  - English: whitespace tokenization
  - Chinese: jieba word segmentation

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Language Support Settings
SUPPORTED_LANGUAGES="en,zh"  # en=English, zh=Chinese
DEFAULT_LANGUAGE="en"
ENABLE_LANGUAGE_DETECTION=True
CHINESE_CHUNK_SIZE=256  # Characters for Chinese text
CHINESE_CHUNK_OVERLAP=64  # Overlap for Chinese text

# Query Expansion (use "llm" for multilingual support)
QUERY_EXPANSION_METHODS="llm"  # Don't use "synonym" for Chinese
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SUPPORTED_LANGUAGES` | `["en", "zh"]` | List of supported languages |
| `DEFAULT_LANGUAGE` | `"en"` | Fallback language |
| `ENABLE_LANGUAGE_DETECTION` | `True` | Auto-detect document language |
| `CHINESE_CHUNK_SIZE` | `256` | Chunk size for Chinese (characters) |
| `CHINESE_CHUNK_OVERLAP` | `64` | Overlap size for Chinese |
| `CHUNK_SIZE` | `512` | Chunk size for English (characters) |
| `CHUNK_OVERLAP` | `128` | Overlap size for English |

## Usage Examples

### Uploading Chinese Documents

The system automatically detects and processes Chinese documents:

```python
# Upload a Traditional Chinese PDF
POST /api/v1/documents/upload
Content-Type: multipart/form-data

{
  "file": "繁體中文文檔.pdf",
  "collection_name": "chinese_docs"
}
```

The document will be:
1. Detected as Chinese (`zh`)
2. Split using Chinese sentence boundaries
3. Chunked into ~256 character segments
4. Embedded using multilingual model
5. Stored with language metadata

### Querying in Chinese

```python
# Query in Traditional Chinese
POST /api/v1/query
Content-Type: application/json

{
  "query": "什麼是人工智能？",
  "collection_name": "chinese_docs",
  "top_k": 5
}
```

The query will be:
1. Detected as Chinese
2. Tokenized using jieba for BM25
3. Embedded using multilingual model
4. Retrieved using hybrid search
5. Re-ranked if enabled

### Mixed Language Collections

You can store both English and Chinese documents in the same collection:

```python
# The system handles mixed languages automatically
collection_stats = {
  "english_docs": 150,
  "chinese_docs": 80,
  "total_chunks": 5420
}
```

Each chunk is tagged with its language in metadata:
```python
{
  "text": "人工智能技術正在快速發展。",
  "language": "zh",
  "chunk_size": 256,
  ...
}
```

## Implementation Details

### Text Splitting Algorithm

#### Chinese Text Splitting
```python
def split_chinese_text(text, chunk_size=256, chunk_overlap=64):
    1. Find sentence boundaries (。！？\n)
    2. Group sentences into chunks ≤ chunk_size
    3. Apply overlap between chunks
    4. Handle long sentences that exceed chunk_size
```

#### Sentence Boundary Detection
Chinese uses different punctuation:
- Period: `。` (not `.`)
- Exclamation: `！` (not `!`)
- Question: `？` (not `?`)

### Tokenization

#### jieba Tokenizer
```python
# Example tokenization
text = "我喜歡閱讀技術文檔"
tokens = jieba.cut(text)
# Result: ['我', '喜歡', '閱讀', '技術', '文檔']
```

### BM25 Integration

The multilingual tokenizer automatically selects the correct tokenization:

```python
def multilingual_tokenizer(text: str) -> List[str]:
    language = detect_language(text)
    if language == 'zh':
        return tokenize_chinese(text)  # jieba
    else:
        return text.lower().split()    # whitespace
```

## Testing

Comprehensive tests are provided in `tests/test_rag/test_chinese_support.py`:

```bash
# Run Chinese support tests
pytest tests/test_rag/test_chinese_support.py -v

# Run specific test class
pytest tests/test_rag/test_chinese_support.py::TestLanguageDetection -v

# Run integration tests
pytest tests/test_rag/test_chinese_support.py::TestIntegration -v
```

Test coverage includes:
- Language detection (English, Traditional Chinese, Simplified Chinese)
- Chinese tokenization with jieba
- Sentence boundary detection
- Text chunking with various edge cases
- Multilingual tokenizer for BM25
- Integration workflows

## Performance Considerations

### Chunk Size Rationale

Chinese characters carry more information than English characters:
- **English**: ~5-6 characters per word average
- **Chinese**: 1-2 characters per word average

Therefore, Chinese chunk size (256) is half of English (512) to maintain similar semantic content per chunk.

### Embedding Model

OpenAI's `text-embedding-3-small` supports both English and Chinese:
- Dimension: 1536
- Multilingual: Yes
- Chinese performance: High quality

For better Chinese performance, consider:
- `text-embedding-3-large` (higher dimension, better quality)
- Specialized Chinese models: `m3e-base`, `bge-large-zh`

## Limitations and Future Work

### Current Limitations
1. **Query Expansion**: Synonym-based expansion only works for English
2. **Traditional vs Simplified**: Both detected as `zh` (no differentiation)
3. **Mixed Language Chunks**: Each chunk is assigned one language only

### Planned Enhancements
1. Chinese-specific synonym expansion using Chinese WordNet
2. Traditional ↔ Simplified conversion using OpenCC
3. Language-specific reranking models
4. Better handling of mixed-language content
5. Support for more languages (Japanese, Korean, etc.)

## Troubleshooting

### Issue: Chinese text not being detected

**Solution**: Check if text contains sufficient Chinese characters (>30% threshold)

```python
from src.rag.language_utils import is_chinese

text = "你的文本"
if not is_chinese(text):
    # Adjust detection threshold or check text encoding
```

### Issue: Poor chunking quality

**Solution**: Adjust Chinese chunk size and overlap

```bash
# In .env file
CHINESE_CHUNK_SIZE=200  # Smaller chunks
CHINESE_CHUNK_OVERLAP=50  # More overlap
```

### Issue: BM25 not working well for Chinese

**Solution**: Verify jieba is installed and tokenizer is being used

```bash
pip install jieba>=0.42.1
```

### Issue: Query expansion fails for Chinese

**Solution**: Use only LLM-based expansion (not synonym)

```bash
# In .env file
QUERY_EXPANSION_METHODS="llm"  # Remove "synonym"
```

## Dependencies

Chinese support requires these additional packages:

```bash
jieba>=0.42.1                      # Chinese word segmentation
opencc-python-reimplemented>=0.1.7 # Traditional/Simplified conversion (future use)
langdetect==1.0.9                  # Language detection (already included)
```

Install with:
```bash
pip install -r requirements.txt
```

## References

- [jieba - Chinese text segmentation](https://github.com/fxsjy/jieba)
- [OpenAI Embeddings - Multilingual support](https://platform.openai.com/docs/guides/embeddings)
- [LangDetect - Language detection library](https://github.com/Mimino666/langdetect)

## Support

For issues or questions about Chinese language support:
1. Check this documentation
2. Review test cases in `tests/test_rag/test_chinese_support.py`
3. Open an issue with example Chinese text that's not working correctly
