# API 測試腳本

本目錄包含 TextReadingRAG API 的測試腳本。

## 測試檔案

### 1. test_document_upload.py
測試文檔上傳相關的 API 端點：

- **單一文檔上傳** (`POST /api/documents/upload`)
  - 成功上傳測試
  - 無效文件類型測試
  - 帶有 metadata 的上傳測試

- **批次文檔上傳** (`POST /api/documents/upload/batch`)
  - 批次上傳測試
  - 超過限制測試

- **處理進度** (`GET /api/documents/progress/{document_id}`)
  - 獲取處理進度
  - 不存在的文檔測試

- **文檔管理**
  - 列出文檔 (`GET /api/documents/list`)
  - 獲取文檔詳情 (`GET /api/documents/{document_id}`)
  - 刪除文檔 (`DELETE /api/documents/{document_id}`)

- **集合管理**
  - 創建集合 (`POST /api/documents/collections`)
  - 列出集合 (`GET /api/documents/collections`)

### 2. test_query.py
測試查詢相關的 API 端點：

- **基本查詢** (`POST /api/query/`)
  - 成功查詢測試
  - 不同檢索策略測試（VECTOR_ONLY, BM25_ONLY, HYBRID）
  - 無結果查詢測試

- **查詢選項**
  - 啟用/禁用查詢擴展
  - 啟用/禁用重排序
  - 自定義參數測試

- **批次查詢** (`POST /api/query/batch`)
  - 並行處理多個查詢

- **文檔比較** (`POST /api/query/compare`)
  - 比較多個文檔的相似性和差異

- **文檔摘要** (`POST /api/query/summarize`)
  - 生成文檔摘要
  - 提取關鍵點

- **查詢統計** (`GET /api/query/stats`)
  - 獲取查詢性能統計

- **驗證測試**
  - 空查詢測試
  - 無效參數測試

## 運行測試

### 運行所有測試

```bash
# 使用 pytest 運行所有 API 測試
python -m pytest tests/test_api/ -v -s
```

### 運行特定測試文件

```bash
# 只測試文檔上傳
python -m pytest tests/test_api/test_document_upload.py -v -s

# 只測試查詢功能
python -m pytest tests/test_api/test_query.py -v -s
```

### 運行特定測試用例

```bash
# 運行特定的測試類
python -m pytest tests/test_api/test_document_upload.py::TestDocumentUpload -v -s

# 運行特定的測試方法
python -m pytest tests/test_api/test_document_upload.py::TestDocumentUpload::test_upload_single_document_success -v -s
```

### 運行時查看詳細輸出

```bash
# -v: 詳細模式，顯示每個測試的名稱
# -s: 顯示 print 語句的輸出
# --tb=short: 簡短的錯誤追蹤
python -m pytest tests/test_api/ -v -s --tb=short
```

### 生成測試覆蓋率報告

```bash
# 安裝 pytest-cov
pip install pytest-cov

# 生成覆蓋率報告
python -m pytest tests/test_api/ --cov=src.api --cov-report=html

# 在瀏覽器中打開 htmlcov/index.html 查看報告
```

## 測試架構

### Fixtures
測試使用 pytest fixtures 來提供：
- **client**: FastAPI 測試客戶端
- **sample_pdf_file**: 單個測試 PDF 文件
- **multiple_pdf_files**: 多個測試 PDF 文件
- **mock_retrieval_nodes**: 模擬的檢索節點

### Mocking
測試使用 `unittest.mock` 來模擬：
- 文檔攝取服務 (DocumentIngestionService)
- 向量存儲 (ChromaVectorStore)
- 檢索服務 (HybridRetrievalService)
- 查詢擴展服務 (QueryExpansionService)
- 重排序服務 (RerankingService)
- LLM (OpenAI)

這確保測試快速運行且不依賴外部服務。

## 測試輸出範例

```
=== Testing Single Document Upload ===
Status Code: 200
Response: {
  "document_id": "abc-123",
  "filename": "test_document.pdf",
  "size": 410,
  "content_type": "application/pdf",
  "processing_status": "pending",
  "collection_name": "test_collection"
}
✓ Document uploaded successfully with ID: abc-123
```

## 注意事項

1. **模擬模式**: 測試使用模擬對象，不會實際調用 OpenAI API 或存儲真實數據
2. **臨時文件**: PDF 測試文件在內存中創建，測試後自動清理
3. **獨立性**: 每個測試都是獨立的，不會相互影響
4. **詳細日誌**: 使用 `-s` 標誌可查看測試過程中的詳細輸出

## 添加新測試

要添加新的測試用例：

1. 在適當的測試類中創建新方法
2. 方法名以 `test_` 開頭
3. 使用 fixtures 獲取所需的測試依賴
4. 使用 `with patch()` 模擬外部依賴
5. 發送 API 請求並驗證響應
6. 使用 `assert` 語句驗證結果
7. 添加 `print` 語句以提供詳細輸出

範例：

```python
def test_my_new_feature(self, client: TestClient):
    """
    Test my new feature.

    Verifies:
    - Feature works correctly
    - Response is valid
    """
    print("\n=== Testing My New Feature ===")

    with patch("src.api.endpoints.module.dependency") as mock_dep:
        mock_dep.return_value = Mock()

        response = client.post("/api/endpoint", json={"data": "test"})

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        assert response.status_code == 200
        assert "expected_field" in response.json()

        print("✓ Feature tested successfully")
```

## 故障排除

### 測試失敗
- 檢查 mock 配置是否正確
- 驗證 API 端點路徑是否正確
- 確認請求數據格式符合 API 要求

### Import 錯誤
- 確保已安裝所有依賴：`pip install -r requirements.txt`
- 確保 PYTHONPATH 包含項目根目錄

### 依賴問題
- 更新 pytest: `pip install --upgrade pytest`
- 清除 pytest 緩存: `pytest --cache-clear`
