# TextReadingRAG 監控指南

## 📊 監控架構概覽

本專案實現了完整的 Prometheus + Grafana 監控解決方案,提供全方位的系統可觀測性。

### 監控組件

| 組件 | 用途 | 端口 |
|------|------|------|
| **Prometheus** | 指標收集與儲存 | 9090 |
| **Grafana** | 視覺化儀表板 | 3000 |
| **Redis Exporter** | Redis 指標匯出 | 9121 |
| **Node Exporter** | 系統指標匯出 | 9100 |
| **cAdvisor** | 容器指標監控 | 8081 |
| **FastAPI App** | 應用程式指標 | 8080/metrics |

---

## 🚀 快速開始

### 1. 啟動監控服務

```bash
# 啟動基礎服務 (ChromaDB + Redis)
docker-compose up -d

# 啟動監控堆疊
docker-compose --profile monitoring up -d

# 驗證所有服務運行中
docker-compose ps
```

### 2. 訪問監控介面

- **Grafana Dashboard**: http://localhost:3000
  - 預設帳號: `admin`
  - 預設密碼: `admin` (首次登入後會要求修改)

- **Prometheus**: http://localhost:9090
  - 查詢指標和設定告警規則

- **cAdvisor**: http://localhost:8081
  - 即時容器資源使用情況

### 3. 配置 Grafana

首次登入 Grafana 後:

1. 數據源已自動配置 (Prometheus)
2. 匯入預設儀表板:
   - 導航至 **Dashboards** → **Import**
   - 上傳 `grafana-dashboard.json`
   - 選擇 Prometheus 數據源
   - 點擊 **Import**

---

## 📈 監控指標說明

### API 指標

由 `prometheus-fastapi-instrumentator` 自動收集:

| 指標名稱 | 類型 | 說明 |
|---------|------|------|
| `http_requests_total` | Counter | HTTP 請求總數 (按方法、路徑、狀態碼) |
| `http_request_duration_seconds` | Histogram | HTTP 請求延遲分佈 |
| `http_requests_in_progress` | Gauge | 當前處理中的請求數 |

**範例查詢**:
```promql
# 每秒請求率
rate(http_requests_total{job="textreadingrag-api"}[5m])

# P95 延遲
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# 錯誤率
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```

### RAG 系統指標

#### 1. 文件處理 (Document Processing)

| 指標名稱 | 類型 | 說明 |
|---------|------|------|
| `rag_document_processing_duration_seconds` | Histogram | 文件處理耗時 |
| `rag_document_processing_errors_total` | Counter | 處理錯誤總數 |
| `rag_documents_indexed_total` | Counter | 已索引文件數 |
| `rag_documents_pending` | Gauge | 待處理文件數 |
| `rag_documents_processing` | Gauge | 處理中文件數 |

**範例查詢**:
```promql
# 平均處理時間
rate(rag_document_processing_duration_seconds_sum[5m]) /
rate(rag_document_processing_duration_seconds_count[5m])

# 處理錯誤率
rate(rag_document_processing_errors_total[5m])
```

#### 2. 檢索 (Retrieval)

| 指標名稱 | 類型 | 說明 |
|---------|------|------|
| `rag_retrieval_duration_seconds` | Histogram | 檢索耗時 |
| `rag_retrieval_errors_total` | Counter | 檢索錯誤數 |
| `rag_retrieval_quality_score` | Histogram | 檢索品質分數 |
| `rag_documents_retrieved` | Histogram | 每次檢索的文件數 |

**範例查詢**:
```promql
# 檢索 P95 延遲 (按策略)
histogram_quantile(0.95,
  rate(rag_retrieval_duration_seconds_bucket{retrieval_strategy="hybrid"}[5m]))

# 平均檢索品質
avg(rag_retrieval_quality_score)
```

#### 3. Query Expansion

| 指標名稱 | 類型 | 說明 |
|---------|------|------|
| `rag_query_expansion_duration_seconds` | Histogram | Query expansion 耗時 |
| `rag_query_expansion_errors_total` | Counter | 錯誤數 |
| `rag_expanded_queries_count` | Histogram | 擴展查詢數量 |

#### 4. Reranking

| 指標名稱 | 類型 | 說明 |
|---------|------|------|
| `rag_reranking_duration_seconds` | Histogram | Reranking 耗時 |
| `rag_reranking_errors_total` | Counter | Reranking 錯誤數 |
| `rag_reranking_score_improvement` | Histogram | 分數改善幅度 |
| `rag_documents_reranked` | Histogram | Rerank 的文件數 |

#### 5. 回應生成 (Generation)

| 指標名稱 | 類型 | 說明 |
|---------|------|------|
| `rag_generation_duration_seconds` | Histogram | 生成耗時 |
| `rag_generation_errors_total` | Counter | 生成錯誤數 |
| `rag_tokens_generated` | Histogram | 生成的 token 數 |

### 系統資源指標

#### Node Exporter 指標

```promql
# CPU 使用率
100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 記憶體使用率
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# 磁碟使用率
100 - ((node_filesystem_avail_bytes{mountpoint="/"} /
         node_filesystem_size_bytes{mountpoint="/"}) * 100)
```

#### Redis 指標

```promql
# Cache hit rate
rate(redis_keyspace_hits_total[5m]) /
(rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m]))

# 記憶體使用
redis_memory_used_bytes / redis_memory_max_bytes

# Key 驅逐率
rate(redis_evicted_keys_total[5m])
```

#### Container 指標 (cAdvisor)

```promql
# 容器 CPU 使用率
rate(container_cpu_usage_seconds_total{name!=""}[5m]) * 100

# 容器記憶體使用率
(container_memory_usage_bytes{name!=""} /
 container_spec_memory_limit_bytes{name!=""}) * 100
```

---

## 🔔 告警規則

告警規則定義在 [`alert_rules.yml`](../alert_rules.yml) 中。

### 關鍵告警

#### API 告警

| 告警名稱 | 條件 | 嚴重性 | 說明 |
|---------|------|--------|------|
| `APIHighErrorRate` | 5xx 錯誤率 > 5% (持續 2 分鐘) | Critical | API 錯誤率過高 |
| `APIHighLatency` | P95 延遲 > 2 秒 (持續 5 分鐘) | Warning | API 回應過慢 |
| `APIDown` | API 無法連接 (持續 1 分鐘) | Critical | API 服務中斷 |

#### RAG 系統告警

| 告警名稱 | 條件 | 嚴重性 | 說明 |
|---------|------|--------|------|
| `HighRetrievalLatency` | P95 檢索時間 > 1.5 秒 | Warning | 檢索速度過慢 |
| `LowRetrievalQuality` | 平均品質分數 < 0.6 (持續 10 分鐘) | Warning | 檢索品質下降 |
| `HighDocumentProcessingFailureRate` | 處理錯誤率 > 10% | Warning | 文件處理失敗率高 |

#### 基礎設施告警

| 告警名稱 | 條件 | 嚴重性 | 說明 |
|---------|------|--------|------|
| `ChromaDBDown` | ChromaDB 無法連接 | Critical | 向量資料庫中斷 |
| `RedisDown` | Redis 無法連接 | Warning | 快取服務中斷 |
| `HighCPUUsage` | CPU 使用率 > 80% (持續 5 分鐘) | Warning | CPU 資源不足 |
| `HighMemoryUsage` | 記憶體使用率 > 85% (持續 5 分鐘) | Warning | 記憶體資源不足 |
| `LowDiskSpace` | 可用磁碟空間 < 15% | Warning | 磁碟空間不足 |

### 測試告警

```bash
# 手動觸發測試告警
curl -X POST http://localhost:9090/-/reload
```

---

## 📊 Grafana 儀表板

預設儀表板包含以下面板:

### 1. 總覽 (Overview)
- 服務健康狀態
- 活躍請求數
- 總文件索引數
- Cache 使用情況

### 2. API 效能
- 請求率趨勢
- 回應時間 (P50/P95/P99)
- 錯誤率
- 請求分佈 (按端點)

### 3. RAG 效能
- 檢索延遲
- Reranking 延遲
- 生成延遲
- 端到端查詢時間

### 4. 品質指標
- 檢索品質分數
- Reranking 改善幅度
- 生成 token 數
- 文件檢索數量分佈

### 5. 系統資源
- CPU 使用率
- 記憶體使用率
- 磁碟使用率
- 網路 I/O

### 6. 錯誤追蹤
- 各組件錯誤率
- 錯誤類型分佈
- 錯誤趨勢

---

## 🛠️ 進階配置

### 自訂指標保留時間

編輯 `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# 在 Prometheus 啟動命令中設定
command:
  - '--storage.tsdb.retention.time=30d'  # 改為 90d 保留 90 天
```

### 增加告警通知

1. **配置 Alertmanager** (可選)

創建 `alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'team-notifications'

receivers:
  - name: 'team-notifications'
    email_configs:
      - to: 'team@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@gmail.com'
        auth_password: 'your-app-password'
```

2. **在 docker-compose.yml 中加入 Alertmanager**:

```yaml
alertmanager:
  image: prom/alertmanager:latest
  ports:
    - "9093:9093"
  volumes:
    - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
  command:
    - '--config.file=/etc/alertmanager/alertmanager.yml'
```

### 自訂 Grafana 儀表板

1. 在 Grafana UI 中編輯儀表板
2. 點擊 **Share** → **Export** → **Save to file**
3. 將 JSON 檔案覆蓋 `grafana-dashboard.json`

### 增加監控端點

編輯 `prometheus.yml`:

```yaml
scrape_configs:
  # 新增自訂服務
  - job_name: 'custom-service'
    static_configs:
      - targets: ['custom-service:9090']
        labels:
          service: 'custom-service'
```

---

## 🔍 常見監控查詢

### 效能分析

```promql
# Top 5 最慢的 API 端點
topk(5,
  histogram_quantile(0.95,
    rate(http_request_duration_seconds_bucket[5m])))

# 各 RAG 階段耗時分佈
sum(rate(rag_retrieval_duration_seconds_sum[5m])) by (retrieval_strategy)
sum(rate(rag_reranking_duration_seconds_sum[5m])) by (reranking_model)
sum(rate(rag_generation_duration_seconds_sum[5m])) by (model_name)

# 查詢吞吐量
sum(rate(rag_queries_total[5m]))
```

### 錯誤追蹤

```promql
# 各組件錯誤率排行
topk(5, sum(rate(rag_retrieval_errors_total[5m])) by (error_type))
topk(5, sum(rate(rag_document_processing_errors_total[5m])) by (error_type))

# 錯誤趨勢
sum(rate(rag_retrieval_errors_total[1h])) by (error_type)
```

### 資源使用

```promql
# 各容器記憶體使用
sum(container_memory_usage_bytes{name!=""}) by (name) / 1024 / 1024 / 1024

# 各容器 CPU 使用
sum(rate(container_cpu_usage_seconds_total{name!=""}[5m])) by (name) * 100
```

---

## 🐛 監控故障排除

### Prometheus 無法啟動

```bash
# 檢查配置文件語法
docker run --rm -v $(pwd)/prometheus.yml:/prometheus.yml prom/prometheus:latest \
  promtool check config /prometheus.yml

# 查看日誌
docker-compose logs prometheus
```

### Grafana 無法連接 Prometheus

1. 檢查 Prometheus 是否運行:
   ```bash
   curl http://localhost:9090/-/healthy
   ```

2. 檢查 Grafana 數據源配置:
   - 登入 Grafana
   - **Configuration** → **Data Sources** → **Prometheus**
   - 測試連接

### 指標未出現

1. 確認 API 正在運行並暴露 `/metrics` 端點:
   ```bash
   curl http://localhost:8080/metrics
   ```

2. 檢查 Prometheus targets:
   - 訪問 http://localhost:9090/targets
   - 確認所有 targets 狀態為 UP

3. 查看 scrape 錯誤:
   ```bash
   # 在 Prometheus UI 中查詢
   up{job="textreadingrag-api"}
   ```

### Windows 上的 cAdvisor 問題

Windows 上 cAdvisor 可能無法正常運行。解決方案:

1. 使用 Windows 容器版本
2. 或移除 cAdvisor,使用 Docker Desktop 內建監控

---

## 📚 最佳實踐

### 1. 監控策略

- **Golden Signals**: 監控延遲、流量、錯誤、飽和度
- **SLI/SLO**: 定義服務水準指標和目標
- **RED Method**: Rate, Errors, Duration

### 2. 告警策略

- **避免告警疲勞**: 只對可行動的問題設置告警
- **分級管理**: Critical → Immediate action, Warning → Investigation
- **抑制規則**: 相關告警只發送一次

### 3. 儀表板設計

- **分層次**: Overview → Deep Dive → Debugging
- **相關性**: 相關指標放在一起
- **可行動**: 每個圖表都應該能回答一個問題

### 4. 效能優化

- **減少高基數標籤**: 避免使用 user_id 等高基數標籤
- **適當的 bucket 設定**: Histogram buckets 應涵蓋實際分佈
- **定期清理**: 刪除不再使用的指標

---

## 🔐 安全建議

### 1. 生產環境配置

```yaml
# Grafana 安全設定
environment:
  - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
  - GF_USERS_ALLOW_SIGN_UP=false
  - GF_AUTH_ANONYMOUS_ENABLED=false
  - GF_SECURITY_SECRET_KEY=${GRAFANA_SECRET_KEY}
```

### 2. Prometheus 認證

使用 reverse proxy (如 Nginx) 增加基本認證:

```nginx
location /prometheus {
    auth_basic "Prometheus";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://prometheus:9090;
}
```

### 3. 網路隔離

生產環境建議:
- 監控服務放在內部網路
- 透過 VPN 或 bastion host 訪問
- 使用防火牆限制訪問

---

## 📞 支援與資源

- **Prometheus 文件**: https://prometheus.io/docs/
- **Grafana 文件**: https://grafana.com/docs/
- **PromQL 教學**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Grafana Dashboard Library**: https://grafana.com/grafana/dashboards/

---

## 🎯 監控檢查清單

啟動監控後,確認以下項目:

- [ ] Prometheus 正在收集指標 (檢查 targets)
- [ ] Grafana 儀表板顯示數據
- [ ] 告警規則已載入
- [ ] Redis exporter 正常運作
- [ ] Node exporter 報告系統指標
- [ ] cAdvisor 顯示容器資源
- [ ] API `/metrics` 端點可訪問
- [ ] 自訂 RAG 指標正在記錄

---

**監控是保障系統健康的關鍵!** 定期檢查儀表板,主動發現並解決問題。
