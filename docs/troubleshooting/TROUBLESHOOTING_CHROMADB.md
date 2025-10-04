# ChromaDB 權限問題排除

## 問題描述
API容器無法連接到ChromaDB，錯誤訊息：
```
Failed to connect to ChromaDB: unable to open database file
```

## 已修正的配置問題

### 1. docker-compose.yml 中的 volume 重複掛載
**修正前:**
```yaml
volumes:
  - chromadb_data:/chroma/chroma
  - ./data/chroma_db:/chroma/chroma:rw
```

**修正後:**
```yaml
volumes:
  - ./data/chroma_db:/chroma/chroma
```

### 2. API容器的 CHROMA_PORT 配置錯誤
**修正前:**
```yaml
environment:
  - CHROMA_HOST=chromadb
  - CHROMA_PORT=7999
```

**修正後:**
```yaml
environment:
  - CHROMA_HOST=chromadb
  - CHROMA_PORT=8000
```

## 需要在伺服器 (192.168.0.118) 上執行的步驟

### 步驟 1: 停止所有容器
```bash
cd /path/to/TextReadingRAG
docker compose down
```

### 步驟 2: 修正數據目錄權限
```bash
# 創建必要的目錄
mkdir -p data/chroma_db
mkdir -p data/uploads
mkdir -p logs

# 設置權限（讓Docker容器可以寫入）
chmod -R 777 data/
chmod -R 777 logs/
```

### 步驟 3: 清理舊的 volume（可選）
```bash
# 如果需要完全重置ChromaDB數據
docker volume rm textreadingrag_chromadb_data 2>/dev/null || true
```

### 步驟 4: 重新啟動容器
```bash
# 重新build並啟動API容器
docker compose up -d --build api

# 檢查容器狀態
docker compose ps

# 檢查API日誌
docker compose logs -f api
```

### 步驟 5: 驗證服務
```bash
# 測試API健康狀態
curl http://localhost:8080/health

# 測試ChromaDB連接
curl http://localhost:7999/api/v1/heartbeat
```

## 驗證修復

修復完成後，應該能夠成功上傳文件：
```bash
curl -X POST "http://192.168.0.118:8080/api/documents/upload" \
  -F "file=@test_document.txt" \
  -F "collection_name=test_collection"
```

預期回應應該包含 document_id 而不是錯誤訊息。
