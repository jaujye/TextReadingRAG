# Chinese Support - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install jieba>=0.42.1 opencc-python-reimplemented>=0.1.7
```

Or update all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Update Configuration

Add to your `.env` file:
```bash
# Enable Chinese support
ENABLE_LANGUAGE_DETECTION=True
CHINESE_CHUNK_SIZE=256
CHINESE_CHUNK_OVERLAP=64

# Use LLM-based query expansion (works for Chinese)
QUERY_EXPANSION_METHODS="llm"
```

### 3. Test It!

```bash
# Run Chinese support tests
pytest tests/test_rag/test_chinese_support.py -v
```

## Usage

### Upload Chinese Document
```python
# API call - system auto-detects Chinese
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: "繁體中文文檔.pdf"
```

### Query in Chinese
```python
# API call - system auto-detects Chinese query
POST /api/v1/query
{
  "query": "什麼是人工智能？",
  "top_k": 5
}
```

### Programmatic Usage
```python
from src.rag.language_utils import detect_language, split_chinese_text

# Detect language
text = "這是繁體中文"
lang = detect_language(text)  # Returns: 'zh'

# Split Chinese text
chunks = split_chinese_text(text, chunk_size=256, chunk_overlap=64)
```

## What Changed?

| Component | English | Chinese |
|-----------|---------|---------|
| **Tokenization** | Whitespace | jieba word segmentation |
| **Chunk Size** | 512 chars | 256 chars |
| **Chunk Overlap** | 128 chars | 64 chars |
| **Punctuation** | . ! ? | 。！？ |
| **Query Expansion** | LLM + Synonym | LLM only |
| **BM25** | Standard | Custom tokenizer |

## Key Functions

```python
# Language utilities (src/rag/language_utils.py)
detect_language(text: str) -> str              # Auto-detect 'en' or 'zh'
is_chinese(text: str) -> bool                  # Check if text is Chinese
tokenize_chinese(text: str) -> List[str]       # jieba tokenization
split_chinese_text(text, size, overlap)        # Semantic chunking

# Multilingual retrieval (src/rag/retrieval.py)
multilingual_tokenizer(text: str) -> List[str] # Auto-select tokenizer
```

## Configuration Reference

```bash
# Language Settings
SUPPORTED_LANGUAGES="en,zh"     # Supported languages
DEFAULT_LANGUAGE="en"           # Fallback language
ENABLE_LANGUAGE_DETECTION=True  # Auto-detect (recommended)

# Chinese-Specific
CHINESE_CHUNK_SIZE=256          # Chunk size for Chinese
CHINESE_CHUNK_OVERLAP=64        # Overlap for Chinese

# Query Expansion
QUERY_EXPANSION_METHODS="llm"   # Don't use "synonym" for Chinese
```

## Metadata

Documents and chunks are tagged with language:
```json
{
  "text": "人工智能技術發展",
  "language": "zh",
  "chunk_size": 256,
  "chunk_overlap": 64,
  "parsing_method": "simple_reader"
}
```

## Testing

```bash
# All tests
pytest tests/test_rag/test_chinese_support.py -v

# Specific test
pytest tests/test_rag/test_chinese_support.py::TestLanguageDetection -v

# Integration only
pytest tests/test_rag/test_chinese_support.py::TestIntegration -v
```

## Troubleshooting

### Chinese not detected?
Check character ratio (needs >30% Chinese):
```python
from src.rag.language_utils import is_chinese
is_chinese("你的文本")  # Should return True
```

### Poor chunking?
Adjust chunk size in `.env`:
```bash
CHINESE_CHUNK_SIZE=200  # Smaller
CHINESE_CHUNK_OVERLAP=50  # More overlap
```

### Query expansion failing?
Use LLM-only (not synonym):
```bash
QUERY_EXPANSION_METHODS="llm"
```

## Examples

### Traditional Chinese
```python
query = "什麼是深度學習？"
# Automatically:
# - Detected as 'zh'
# - Tokenized with jieba
# - Retrieved with hybrid search
# - Returns relevant Chinese chunks
```

### Simplified Chinese
```python
query = "什么是深度学习？"
# Automatically:
# - Detected as 'zh' (same as Traditional)
# - Processed identically
# - Works seamlessly
```

### Mixed Collection
```python
# Store both English and Chinese docs
upload("english_doc.pdf")   # Tagged as 'en'
upload("chinese_doc.pdf")   # Tagged as 'zh'

# Query in either language
query_en = "What is AI?"    # Retrieves from both
query_zh = "什麼是人工智能？"  # Retrieves from both
```

## Architecture

```
Document Upload
    ↓
[Language Detection] → metadata["language"] = "zh"
    ↓
[Chinese Text Splitting] → jieba + sentence boundaries
    ↓
[Embedding] → OpenAI multilingual model
    ↓
[Storage] → ChromaDB with language tags

Query Processing
    ↓
[Language Detection] → "zh"
    ↓
[Dense: Embedding] + [Sparse: jieba BM25]
    ↓
[Hybrid Fusion]
    ↓
[Reranking]
    ↓
Results
```

## Performance

- **Language detection**: ~1-5ms per document
- **jieba tokenization**: ~10-50ms (first load ~200ms for dictionary)
- **No impact** on English documents
- **Memory**: +30MB for jieba dictionary (shared)

## Best Practices

✅ **Do:**
- Enable language detection for mixed collections
- Use LLM-based query expansion for multilingual
- Keep default chunk sizes (256 for Chinese, 512 for English)
- Tag documents with language metadata

❌ **Don't:**
- Use synonym expansion for Chinese queries
- Mix languages in single chunk
- Disable language detection for multilingual collections
- Use English-only chunk sizes for Chinese

## Next Steps

1. Read full docs: `docs/CHINESE_SUPPORT.md`
2. Review implementation: `CHINESE_SUPPORT_SUMMARY.md`
3. Run tests: `pytest tests/test_rag/test_chinese_support.py`
4. Try with your Chinese documents!

## Support

- Full docs: `docs/CHINESE_SUPPORT.md`
- Tests: `tests/test_rag/test_chinese_support.py`
- Code: `src/rag/language_utils.py`
