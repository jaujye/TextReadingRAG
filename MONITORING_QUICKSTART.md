# 🚀 監控快速啟動指南

## 一鍵啟動

```bash
# 1. 啟動所有監控服務
docker-compose --profile monitoring up -d

# 2. 驗證服務狀態
docker-compose ps

# 3. 安裝 Python 依賴 (如果還沒安裝)
pip install prometheus-client prometheus-fastapi-instrumentator

# 4. 啟動 FastAPI 應用
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## 訪問監控介面

| 服務 | URL | 預設帳密 |
|------|-----|----------|
| **Grafana Dashboard** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | 無需認證 |
| **API Metrics** | http://localhost:8080/metrics | 無需認證 |
| **cAdvisor** | http://localhost:8081 | 無需認證 |

## 匯入 Grafana 儀表板

1. 登入 Grafana (http://localhost:3000)
2. 左側選單 → **Dashboards** → **Import**
3. 點擊 **Upload JSON file**
4. 選擇 `grafana-dashboard.json`
5. 選擇 **Prometheus** 作為數據源
6. 點擊 **Import**

## 驗證監控正常運作

```bash
# 檢查 Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# 檢查 API metrics
curl http://localhost:8080/metrics | grep "http_requests_total"

# 測試查詢
curl -X POST http://localhost:8080/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "query_type": "question_answer"}'
```

## 停止監控服務

```bash
# 停止監控服務但保留數據
docker-compose --profile monitoring stop

# 完全移除監控服務和數據
docker-compose --profile monitoring down -v
```

## 常見問題

**Q: Grafana 顯示 "No Data"?**
- 確認 Prometheus 正在運行: `docker-compose ps prometheus`
- 檢查數據源配置: Grafana → Configuration → Data Sources
- 測試 Prometheus 查詢: http://localhost:9090/graph

**Q: API metrics 未出現?**
- 確認已安裝依賴: `pip list | grep prometheus`
- 檢查 API 是否暴露 metrics: `curl http://localhost:8080/metrics`
- 查看 Prometheus targets: http://localhost:9090/targets

**詳細文件**: 請參考 [docs/MONITORING.md](docs/MONITORING.md)
