# TextReadingRAG 生產環境部署指南 (Ubuntu Server)

本文檔詳細說明如何在 Ubuntu Server 生產環境中使用 Docker Compose 部署 TextReadingRAG 系統。

## 📋 系統需求

### 硬體需求
- **CPU**: 4核心或以上（推薦 8核心）
- **記憶體**: 8GB RAM 或以上（推薦 16GB）
- **儲存空間**: 50GB 或以上（用於 Docker 映像、向量資料庫和文件儲存）
- **網路**: 穩定的網路連接（用於下載 Docker 映像和 AI 模型）

### 軟體需求
- **作業系統**: Ubuntu Server 20.04 LTS 或更高版本
- **Docker**: 24.0.0 或更高版本
- **Docker Compose**: 2.20.0 或更高版本
- **OpenAI API Key**: 用於嵌入和語言模型

## 🚀 部署步驟

### 1. 系統準備

#### 1.1 更新系統套件
```bash
sudo apt update && sudo apt upgrade -y
```

#### 1.2 安裝必要工具
```bash
sudo apt install -y curl wget git vim ufw
```

#### 1.3 安裝 Docker
```bash
# 安裝 Docker 官方 GPG 金鑰
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 添加 Docker 官方 APT 倉庫
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安裝 Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 啟動 Docker 服務
sudo systemctl enable docker
sudo systemctl start docker

# 驗證安裝
sudo docker --version
sudo docker compose version
```

#### 1.4 配置 Docker 使用者權限（可選）
```bash
# 將當前使用者加入 docker 群組
sudo usermod -aG docker $USER

# 重新登入以套用群組變更
newgrp docker

# 驗證權限
docker ps
```

### 2. 下載專案

#### 2.1 克隆儲存庫
```bash
# 建立專案目錄
sudo mkdir -p /opt/textreadingrag
sudo chown $USER:$USER /opt/textreadingrag
cd /opt/textreadingrag

# 克隆專案（替換為實際的儲存庫 URL）
git clone <repository-url> .
```

或者手動上傳專案檔案：
```bash
# 在本地壓縮專案
tar -czf textreadingrag.tar.gz .

# 上傳到伺服器
scp textreadingrag.tar.gz user@server:/opt/textreadingrag/

# 在伺服器上解壓縮
cd /opt/textreadingrag
tar -xzf textreadingrag.tar.gz
rm textreadingrag.tar.gz
```

### 3. 配置環境變數

#### 3.1 建立 .env 檔案
```bash
cd /opt/textreadingrag
cp .env.example .env
vim .env
```

#### 3.2 編輯 .env 檔案（最小配置）
```env
# OpenAI API 配置（必填）
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# ChromaDB 配置
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# Redis 配置
REDIS_HOST=redis
REDIS_PORT=6379
ENABLE_CACHE=true

# 應用配置
DEBUG=False
LOG_LEVEL=INFO
MAX_FILE_SIZE=50
CHUNK_SIZE=512
CHUNK_OVERLAP=128

# 檢索配置
DENSE_TOP_K=10
SPARSE_TOP_K=10
ALPHA=0.5
RERANK_TOP_N=3
```

#### 3.3 配置檔案權限
```bash
# 保護敏感配置檔案
chmod 600 .env
```

### 4. 建立資料目錄

```bash
# 建立必要的資料目錄
mkdir -p data/uploads data/processed data/chroma_db logs

# 設定目錄權限
chmod 755 data logs
chmod 777 data/uploads data/processed data/chroma_db
```

### 5. 配置防火牆

```bash
# 啟用 UFW 防火牆
sudo ufw enable

# 允許 SSH（重要！避免被鎖定）
sudo ufw allow 22/tcp

# 允許 HTTP 和 HTTPS（如果使用 Nginx）
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 允許應用 API 埠（可選，用於直接訪問）
sudo ufw allow 8080/tcp

# 檢查防火牆狀態
sudo ufw status
```

### 6. 啟動服務

#### 6.1 僅啟動基礎服務（ChromaDB + Redis）
```bash
cd /opt/textreadingrag
docker compose up -d
```

#### 6.2 啟動完整堆疊（包含 API 服務）
```bash
cd /opt/textreadingrag
docker compose --profile full-stack up -d
```

#### 6.3 使用生產配置（包含 Nginx）
```bash
cd /opt/textreadingrag
docker compose --profile production up -d
```

#### 6.4 驗證服務狀態
```bash
# 檢查所有容器狀態
docker compose ps

# 檢查服務健康狀態
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# 查看服務日誌
docker compose logs -f
```

### 7. 驗證部署

#### 7.1 檢查 ChromaDB
```bash
curl http://localhost:8000/api/v1/heartbeat
# 預期輸出: {"nanosecond heartbeat": ...}
```

#### 7.2 檢查 Redis
```bash
docker exec textreadingrag-redis redis-cli ping
# 預期輸出: PONG
```

#### 7.3 檢查 API（如果使用 full-stack）
```bash
curl http://localhost:8080/health
# 預期輸出: {"status": "healthy", ...}

# 檢查 API 文檔
curl http://localhost:8080/docs
```

#### 7.4 測試完整流程
```bash
# 上傳測試文件
curl -X POST "http://localhost:8080/api/documents/upload" \
  -F "file=@test.pdf" \
  -F "collection_name=test_collection"

# 執行查詢
curl -X POST "http://localhost:8080/api/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "測試查詢",
    "query_type": "question_answer",
    "retrieval_strategy": "hybrid",
    "top_k": 5
  }'
```

## 🔒 生產環境安全設定

### 1. 使用 Nginx 反向代理

#### 1.1 建立 Nginx 配置檔案
```bash
vim /opt/textreadingrag/nginx.conf
```

#### 1.2 Nginx 配置內容
```nginx
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/ssl/certs/your-cert.crt;
        ssl_certificate_key /etc/ssl/certs/your-key.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security Headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000" always;

        # Client body size limit
        client_max_body_size 50M;

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;

            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 300s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://api_backend/health;
            access_log off;
        }

        # API docs
        location /docs {
            proxy_pass http://api_backend/docs;
        }

        location /openapi.json {
            proxy_pass http://api_backend/openapi.json;
        }
    }
}
```

#### 1.3 配置 SSL 憑證
```bash
# 建立 SSL 目錄
mkdir -p /opt/textreadingrag/ssl

# 使用 Let's Encrypt（推薦）
sudo apt install -y certbot
sudo certbot certonly --standalone -d your-domain.com

# 複製憑證到專案目錄
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /opt/textreadingrag/ssl/your-cert.crt
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /opt/textreadingrag/ssl/your-key.key
sudo chown $USER:$USER /opt/textreadingrag/ssl/*
```

### 2. 配置資料持久化

#### 2.1 使用外部儲存（推薦）
```bash
# 編輯 docker-compose.override.yml
vim docker-compose.override.yml
```

```yaml
version: '3.8'

services:
  chromadb:
    volumes:
      # 使用外部掛載點確保資料持久化
      - /mnt/data/chromadb:/chroma/chroma:rw

  api:
    volumes:
      - /mnt/data/uploads:/app/data/uploads:rw
      - /mnt/data/processed:/app/data/processed:rw
      - /var/log/textreadingrag:/app/logs:rw
```

#### 2.2 建立備份腳本
```bash
vim /opt/textreadingrag/scripts/backup.sh
```

```bash
#!/bin/bash

BACKUP_DIR="/mnt/backups/textreadingrag"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/backup_$TIMESTAMP"

# 建立備份目錄
mkdir -p $BACKUP_PATH

# 備份 ChromaDB 資料
docker compose exec -T chromadb tar czf - /chroma/chroma > $BACKUP_PATH/chromadb.tar.gz

# 備份上傳的檔案
tar czf $BACKUP_PATH/uploads.tar.gz -C /opt/textreadingrag data/uploads

# 備份配置檔案
cp /opt/textreadingrag/.env $BACKUP_PATH/.env.backup
cp /opt/textreadingrag/docker-compose.yml $BACKUP_PATH/

# 刪除 7 天前的備份
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_PATH"
```

#### 2.3 設定定期備份
```bash
# 授予執行權限
chmod +x /opt/textreadingrag/scripts/backup.sh

# 設定 cron 任務（每天凌晨 2 點備份）
crontab -e

# 添加以下行
0 2 * * * /opt/textreadingrag/scripts/backup.sh >> /var/log/textreadingrag-backup.log 2>&1
```

### 3. 日誌管理

#### 3.1 配置日誌輪轉
```bash
sudo vim /etc/logrotate.d/textreadingrag
```

```
/var/log/textreadingrag/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 root root
    sharedscripts
    postrotate
        docker compose -f /opt/textreadingrag/docker-compose.yml restart api > /dev/null 2>&1 || true
    endscript
}
```

#### 3.2 查看日誌
```bash
# 查看所有服務日誌
docker compose logs -f

# 查看特定服務日誌
docker compose logs -f api
docker compose logs -f chromadb
docker compose logs -f redis

# 查看最近 100 行日誌
docker compose logs --tail=100 api

# 查看應用日誌檔案
tail -f /opt/textreadingrag/logs/*.log
```

## 📊 監控與維護

### 1. 系統監控

#### 1.1 安裝監控工具
```bash
# 安裝 Docker stats 工具
sudo apt install -y docker-compose-plugin

# 查看容器資源使用情況
docker stats

# 查看磁碟使用情況
df -h
du -sh /opt/textreadingrag/data/*
```

#### 1.2 設定健康檢查腳本
```bash
vim /opt/textreadingrag/scripts/health_check.sh
```

```bash
#!/bin/bash

# 檢查服務健康狀態
check_service() {
    SERVICE_NAME=$1
    HEALTH_URL=$2

    if curl -f -s $HEALTH_URL > /dev/null; then
        echo "✓ $SERVICE_NAME is healthy"
        return 0
    else
        echo "✗ $SERVICE_NAME is unhealthy"
        return 1
    fi
}

# 檢查 ChromaDB
check_service "ChromaDB" "http://localhost:8000/api/v1/heartbeat"

# 檢查 API
check_service "API" "http://localhost:8080/health"

# 檢查 Redis
docker exec textreadingrag-redis redis-cli ping > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ Redis is healthy"
else
    echo "✗ Redis is unhealthy"
fi

# 檢查磁碟空間
DISK_USAGE=$(df -h /opt/textreadingrag | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "⚠ Disk usage is high: ${DISK_USAGE}%"
fi
```

```bash
chmod +x /opt/textreadingrag/scripts/health_check.sh
```

### 2. 效能調優

#### 2.1 調整 Docker Compose 資源限制
```yaml
# docker-compose.override.yml
version: '3.8'

services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '2'
          memory: 2G
    environment:
      - WORKERS=4  # 根據 CPU 核心數調整

  chromadb:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  redis:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

#### 2.2 橫向擴展（多實例）
```bash
# 啟動多個 API 實例
docker compose --profile full-stack up -d --scale api=3

# 配置 Nginx 負載均衡
# 修改 nginx.conf upstream 部分：
upstream api_backend {
    least_conn;
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
}
```

### 3. 更新與維護

#### 3.1 更新應用程式
```bash
cd /opt/textreadingrag

# 拉取最新程式碼
git pull origin main

# 重新建構映像
docker compose --profile full-stack build --no-cache

# 重新啟動服務（零停機時間）
docker compose --profile full-stack up -d --force-recreate --no-deps api
```

#### 3.2 更新 Docker 映像
```bash
# 拉取最新的基礎映像
docker compose pull

# 重新啟動服務
docker compose --profile full-stack up -d
```

#### 3.3 清理 Docker 資源
```bash
# 清理未使用的映像
docker image prune -a -f

# 清理未使用的容器
docker container prune -f

# 清理未使用的卷
docker volume prune -f

# 完整清理（小心使用）
docker system prune -a -f --volumes
```

## 🔧 故障排除

### 1. 常見問題

#### 問題：容器無法啟動
```bash
# 檢查容器日誌
docker compose logs api

# 檢查容器狀態
docker compose ps

# 重新啟動特定服務
docker compose restart api
```

#### 問題：ChromaDB 連接失敗
```bash
# 檢查 ChromaDB 是否運行
docker compose ps chromadb

# 檢查網路連接
docker compose exec api ping chromadb

# 重新建立網路
docker compose down
docker network prune -f
docker compose up -d
```

#### 問題：記憶體不足
```bash
# 檢查記憶體使用
docker stats

# 增加 swap 空間
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久啟用 swap
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 問題：磁碟空間不足
```bash
# 檢查磁碟使用情況
df -h
du -sh /opt/textreadingrag/*

# 清理日誌
find /opt/textreadingrag/logs -name "*.log" -mtime +7 -delete

# 清理舊的上傳檔案
find /opt/textreadingrag/data/uploads -mtime +30 -delete
```

### 2. 效能優化

#### 2.1 資料庫優化
```bash
# 定期重建 ChromaDB 索引
docker compose exec chromadb chroma optimize

# 清理未使用的集合
# 透過 API 刪除舊的集合
```

#### 2.2 快取優化
```bash
# 監控 Redis 快取命中率
docker exec textreadingrag-redis redis-cli info stats | grep hit

# 調整快取過期時間（在 .env 中）
CACHE_TTL=7200  # 2 小時
```

## 📝 最佳實踐

### 1. 安全性
- ✅ 定期更新系統和 Docker
- ✅ 使用強密碼和 API 金鑰
- ✅ 啟用 HTTPS 和 SSL 憑證
- ✅ 限制 API 存取速率
- ✅ 定期備份資料
- ✅ 監控異常存取日誌

### 2. 可靠性
- ✅ 設定健康檢查和自動重啟
- ✅ 使用容器編排（Docker Swarm 或 Kubernetes）
- ✅ 實施負載均衡
- ✅ 配置監控告警
- ✅ 建立災難恢復計畫

### 3. 效能
- ✅ 根據負載調整資源限制
- ✅ 啟用快取減少重複計算
- ✅ 使用 SSD 儲存向量資料庫
- ✅ 優化查詢參數（top_k、chunk_size）
- ✅ 實施橫向擴展

### 4. 維護
- ✅ 定期檢查日誌
- ✅ 監控資源使用情況
- ✅ 定期備份和測試恢復
- ✅ 保持文檔更新
- ✅ 建立變更管理流程

## 🆘 支援與協助

如遇到問題，請參考：
- **專案文檔**: [README.md](../README.md)
- **API 文檔**: http://your-domain.com/docs
- **GitHub Issues**: 提交問題報告
- **日誌檔案**: `/opt/textreadingrag/logs/`

---

**部署完成後，您的 TextReadingRAG 系統已準備好為生產環境服務！** 🎉
