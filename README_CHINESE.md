# 繁體中文支援 / Traditional Chinese Support

[English](#english) | [繁體中文](#繁體中文)

---

## 繁體中文

### 概述
TextReadingRAG 系統現已全面支援繁體中文和簡體中文文檔檢索。系統會自動檢測文檔語言，並應用針對性的處理策略以獲得最佳效果。

### 主要功能

✅ **自動語言檢測** - 自動識別英文和中文文檔
✅ **中文分詞** - 使用 jieba 進行準確的中文分詞
✅ **語義分塊** - 尊重中文句子邊界（。！？）
✅ **優化的分塊大小** - 針對中文信息密度優化（256字符 vs 英文512字符）
✅ **混合檢索** - 結合向量檢索和 BM25 稀疏檢索
✅ **多語言查詢** - 支持中英文查詢

### 快速開始

#### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

新增的中文支援依賴：
- `jieba>=0.42.1` - 中文分詞
- `opencc-python-reimplemented>=0.1.7` - 繁簡轉換支援

#### 2. 配置

在 `.env` 文件中添加：
```bash
# 啟用語言檢測
ENABLE_LANGUAGE_DETECTION=True

# 中文分塊參數
CHINESE_CHUNK_SIZE=256
CHINESE_CHUNK_OVERLAP=64

# 查詢擴展（使用 LLM 以支持中文）
QUERY_EXPANSION_METHODS="llm"
```

#### 3. 使用示例

**上傳中文文檔：**
```python
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: "繁體中文文檔.pdf"
```

**中文查詢：**
```python
POST /api/v1/query
{
  "query": "什麼是機器學習？",
  "top_k": 5
}
```

### 技術特點

#### 英文處理
- 分詞：空格分割
- 分塊大小：512字符
- 句子邊界：. ! ?
- 查詢擴展：LLM + 同義詞

#### 中文處理
- 分詞：jieba 分詞
- 分塊大小：256字符
- 句子邊界：。！？
- 查詢擴展：僅 LLM

### 示例代碼

```python
from src.rag.language_utils import detect_language, split_chinese_text

# 檢測語言
text = "這是繁體中文測試文本。"
language = detect_language(text)  # 返回: 'zh'

# 分割中文文本
chunks = split_chinese_text(
    text,
    chunk_size=256,
    chunk_overlap=64
)
```

### 運行示例

```bash
# 運行中文支援示例
python examples/chinese_example.py

# 運行測試
pytest tests/test_rag/test_chinese_support.py -v
```

### 文檔

- **📖 文檔中心**: [docs/README.md](docs/README.md) - 完整文檔導覽
- **完整指南**: [docs/guides/CHINESE_SUPPORT.md](docs/guides/CHINESE_SUPPORT.md) - 包含快速開始與詳細說明
- **示例代碼**: [examples/chinese_example.py](examples/chinese_example.py)

### 性能

- 語言檢測：~1-5毫秒/文檔
- jieba 分詞：~10-50毫秒/文檔
- 對英文文檔無影響
- 內存開銷：~30MB（jieba 詞典，共享）

### 支援的語言

| 語言 | 代碼 | 分詞方式 | 分塊大小 |
|------|------|----------|----------|
| 英文 | en | 空格分割 | 512字符 |
| 繁體中文 | zh | jieba | 256字符 |
| 簡體中文 | zh | jieba | 256字符 |

---

## English

### Overview
TextReadingRAG now fully supports Traditional Chinese (繁體中文) and Simplified Chinese (简体中文) document retrieval. The system automatically detects document language and applies language-specific processing for optimal results.

### Key Features

✅ **Automatic Language Detection** - Auto-identifies English and Chinese documents
✅ **Chinese Tokenization** - Uses jieba for accurate Chinese word segmentation
✅ **Semantic Chunking** - Respects Chinese sentence boundaries (。！？)
✅ **Optimized Chunk Sizes** - Optimized for Chinese information density (256 vs 512 chars)
✅ **Hybrid Retrieval** - Combines vector search and BM25 sparse retrieval
✅ **Multilingual Queries** - Supports both English and Chinese queries

### Quick Start

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

New Chinese support dependencies:
- `jieba>=0.42.1` - Chinese word segmentation
- `opencc-python-reimplemented>=0.1.7` - Traditional/Simplified conversion support

#### 2. Configuration

Add to your `.env` file:
```bash
# Enable language detection
ENABLE_LANGUAGE_DETECTION=True

# Chinese chunking parameters
CHINESE_CHUNK_SIZE=256
CHINESE_CHUNK_OVERLAP=64

# Query expansion (use LLM for multilingual support)
QUERY_EXPANSION_METHODS="llm"
```

#### 3. Usage Examples

**Upload Chinese Document:**
```python
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: "traditional_chinese_doc.pdf"
```

**Chinese Query:**
```python
POST /api/v1/query
{
  "query": "What is machine learning?",
  "top_k": 5
}
```

### Technical Details

#### English Processing
- Tokenization: Whitespace splitting
- Chunk size: 512 characters
- Sentence boundaries: . ! ?
- Query expansion: LLM + Synonym

#### Chinese Processing
- Tokenization: jieba word segmentation
- Chunk size: 256 characters
- Sentence boundaries: 。！？
- Query expansion: LLM only

### Example Code

```python
from src.rag.language_utils import detect_language, split_chinese_text

# Detect language
text = "這是繁體中文測試文本。"
language = detect_language(text)  # Returns: 'zh'

# Split Chinese text
chunks = split_chinese_text(
    text,
    chunk_size=256,
    chunk_overlap=64
)
```

### Run Examples

```bash
# Run Chinese support examples
python examples/chinese_example.py

# Run tests
pytest tests/test_rag/test_chinese_support.py -v
```

### Documentation

- **📖 Documentation Center**: [docs/README.md](docs/README.md) - Complete documentation index
- **Complete Guide**: [docs/guides/CHINESE_SUPPORT.md](docs/guides/CHINESE_SUPPORT.md) - Quick start + full documentation
- **Example Code**: [examples/chinese_example.py](examples/chinese_example.py)

### Performance

- Language detection: ~1-5ms per document
- jieba tokenization: ~10-50ms per document
- No impact on English documents
- Memory overhead: ~30MB (jieba dictionary, shared)

### Supported Languages

| Language | Code | Tokenization | Chunk Size |
|----------|------|--------------|------------|
| English | en | Whitespace | 512 chars |
| Traditional Chinese | zh | jieba | 256 chars |
| Simplified Chinese | zh | jieba | 256 chars |

### Architecture

```
Document Upload → Language Detection → Language-Specific Processing
                                               ↓
                     English ←→ [Router] ←→ Chinese
                        ↓                      ↓
                  SentenceSplitter      jieba + Boundaries
                        ↓                      ↓
                        └──→ Embeddings ←──────┘
                                ↓
                           ChromaDB Storage

Query → Detection → Multilingual BM25 + Dense Retrieval → Results
```

### Testing

```bash
# All tests
pytest tests/test_rag/test_chinese_support.py -v

# Specific test class
pytest tests/test_rag/test_chinese_support.py::TestLanguageDetection -v

# Integration tests
pytest tests/test_rag/test_chinese_support.py::TestIntegration -v
```

### Resources

- **jieba**: https://github.com/fxsjy/jieba
- **OpenAI Embeddings**: Supports multilingual (including Chinese)
- **LangDetect**: Language detection library

### Support

For issues or questions:
1. Check documentation: [docs/guides/CHINESE_SUPPORT.md](docs/guides/CHINESE_SUPPORT.md)
2. Review test cases: [tests/test_rag/test_chinese_support.py](tests/test_rag/test_chinese_support.py)
3. Run examples: `python examples/chinese_example.py`
4. ChromaDB troubleshooting: [docs/troubleshooting/TROUBLESHOOTING_CHROMADB.md](docs/troubleshooting/TROUBLESHOOTING_CHROMADB.md)

---

## 技術支持 / Technical Support

- 📖 文檔中心 / Documentation Center: [docs/README.md](docs/README.md)
- 📖 完整指南 / Complete Guide: [docs/guides/CHINESE_SUPPORT.md](docs/guides/CHINESE_SUPPORT.md)
- 💻 示例代碼 / Examples: [examples/chinese_example.py](examples/chinese_example.py)
- 🧪 測試 / Tests: [tests/test_rag/test_chinese_support.py](tests/test_rag/test_chinese_support.py)
