# TextReadingRAG 文檔中心

## 📚 文檔導覽

本目錄包含 TextReadingRAG 專案的完整文檔，依照類別組織如下：

---

## 🚀 快速開始

### 新手入門
- **[專案說明](../README.md)** - 專案概述、功能特色、系統架構
- **[中文支援指南](guides/CHINESE_SUPPORT.md)** - 包含快速開始與完整文檔

### 英文版本
- **[English README](../README.md)** - Project overview and features
- **[中文版說明](../README_CHINESE.md)** - 中文版專案說明

---

## 📖 使用指南 (guides/)

### 部署相關
- **[生產環境部署指南](guides/PRODUCTION_DEPLOYMENT.md)**
  - Docker 容器化部署
  - 生產環境安全設定
  - 效能調優建議
  - 監控與日誌配置

### 監控與運維
- **[監控系統完整指南](guides/MONITORING.md)**
  - 快速啟動 (5分鐘)
  - Prometheus + Grafana 配置
  - 關鍵指標說明
  - 告警規則設定

---

## 🔧 疑難排解 (troubleshooting/)

### 資料庫相關
- **[ChromaDB 疑難排解](troubleshooting/TROUBLESHOOTING_CHROMADB.md)**
  - 常見錯誤診斷
  - 連接問題解決
  - 效能優化建議

---

## 🧪 測試與效能 (testing/)

### 效能測試報告
- **[RAG 系統效能測試報告](testing/RAG_PERFORMANCE_TEST_REPORT.md)**
  - 學術論文測試案例
  - 準確率評估 (95/100)
  - 效能指標分析
  - 優化建議與實施計劃

---

## 📁 文檔結構

```
docs/
├── README.md                          # 本文檔索引
├── guides/                            # 使用指南
│   ├── CHINESE_SUPPORT.md            # 中文支援完整指南 (含快速開始)
│   ├── MONITORING.md                 # 監控系統完整指南 (含快速啟動)
│   └── PRODUCTION_DEPLOYMENT.md      # 生產環境部署指南
├── troubleshooting/                   # 疑難排解
│   └── TROUBLESHOOTING_CHROMADB.md   # ChromaDB 問題排解
└── testing/                          # 測試文檔
    └── RAG_PERFORMANCE_TEST_REPORT.md # 效能測試報告
```

---

## 🔗 相關資源

### 配置文件
- [環境變數範例](../.env.example) - 環境配置說明
- [Docker Compose](../docker-compose.yml) - 容器編排配置
- [Nginx 配置](../nginx.conf) - 反向代理設定
- [Prometheus 配置](../prometheus.yml) - 監控配置
- [告警規則](../alert_rules.yml) - 告警設定

### 範例與測試
- [測試數據](../examples/test_data/) - 測試文件與範例
- [Jupyter Notebook](../examples/test_data/test_rag_system.ipynb) - RAG 系統測試筆記本

### 開發資源
- [需求文件](../requirements.txt) - Python 依賴套件
- [Dockerfile](../Dockerfile) - 容器映像建構
- [License](../LICENSE) - Apache 2.0 授權

---

## 📝 文檔維護

### 文檔更新規範
1. 所有新增文檔需更新本索引
2. 文檔間互相引用需使用相對路徑
3. 重要更新需註明日期與版本

### 文檔分類原則
- **guides/** - 操作指南、使用教學
- **troubleshooting/** - 問題診斷、解決方案
- **testing/** - 測試報告、效能分析
- **api/** (未來) - API 文檔、介面說明
- **architecture/** (未來) - 系統架構、設計文檔

---

## ❓ 獲取幫助

### 問題回報
- GitHub Issues: [提交問題](https://github.com/your-repo/issues)
- 文檔問題請標註 `documentation` 標籤

### 貢獻文檔
1. Fork 專案
2. 在適當分類下新增或修改文檔
3. 更新本索引文件
4. 提交 Pull Request

---

**最後更新**: 2025-10-04
**維護者**: TextReadingRAG Team
