# AI Lesson Builder Service — Hướng dẫn chạy

## Tổng quan hệ thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MÁY CỦA BẠN                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────────┐         ┌──────────────────────────────┐    │
│   │  AI Lesson API   │         │       Langfuse Stack         │    │
│   │  (Python/FastAPI)│ ──────► │  (Quản lý prompt + Camera)   │    │
│   │  Port: 30001     │         │  Port: 3010                  │    │
│   └────────┬─────────┘         └──────────────────────────────┘    │
│            │                                                        │
│            │ gọi API                                                │
│            ▼                                                        │
│   ┌──────────────────┐                                             │
│   │    OpenAI API    │  (bên ngoài, trả tiền theo token)           │
│   │  - Terra (ảnh)   │                                             │
│   │  - Luna (guard)  │                                             │
│   └──────────────────┘                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Các service cần chạy

| Service | Mục đích | Port | Bắt buộc? |
|---------|----------|------|-----------|
| **AI Lesson API** | Nhận ảnh → trả bài học | 30001 | ✅ Có |
| **Langfuse** | Dashboard xem AI làm gì + quản lý prompt | 3010 | Không (nhưng nên có) |

---

## 1. Chạy AI Lesson API (service chính)

### Lần đầu tiên

```bash
cd /Users/tranhieu/ai-lesson-builder-service

# Tạo môi trường ảo Python
python3 -m venv venv
source venv/bin/activate

# Cài thư viện
pip install -r requirements.txt

# Copy file cấu hình mẫu (nếu chưa có .env)
cp .env.example .env
# Sau đó mở .env, điền OPENAI_API_KEY
```

### Chạy hằng ngày

```bash
cd /Users/tranhieu/ai-lesson-builder-service
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 30001 --reload
```

**Kiểm tra hoạt động:**
```bash
curl http://localhost:30001/health
# Phải trả về: {"status": "ok"}
```

### Chạy nền (không chiếm terminal)

```bash
cd /Users/tranhieu/ai-lesson-builder-service
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 30001 --reload > /tmp/server.log 2>&1 &

# Xem log
tail -f /tmp/server.log

# Tắt service
pkill -f "uvicorn app.main:app"
```

---

## 2. Chạy Langfuse (dashboard theo dõi)

### Langfuse để làm gì?

1. **Xem AI đang làm gì** — mỗi request tạo bài học sẽ hiện trên dashboard, kèm:
   - Thời gian từng bước (lấy profile, kiểm tra ảnh, soạn bài...)
   - Input/output của AI
   - Chi phí token
   - Lỗi (nếu có) — tô đỏ dễ thấy

2. **Quản lý prompt (kịch bản cho AI)** — sửa prompt trên web, không cần deploy lại code

### Lần đầu tiên

```bash
cd /Users/tranhieu/ai-lesson-builder-service/deploy/langfuse

# Khởi động (lần đầu sẽ pull image, mất 5-10 phút)
docker compose up -d

# Đợi khoảng 1-2 phút cho database khởi tạo
# Kiểm tra tất cả container healthy
docker compose ps
```

**Mở dashboard:** http://localhost:3010

**Đăng nhập:**
- Email: `admin@pika.local`
- Mật khẩu: xem trong file `deploy/langfuse/.env`, dòng `LANGFUSE_INIT_USER_PASSWORD`

### Chạy / Tắt hằng ngày

```bash
cd /Users/tranhieu/ai-lesson-builder-service/deploy/langfuse

# Bật
docker compose up -d

# Tắt (dữ liệu vẫn giữ)
docker compose down

# Tắt + xóa dữ liệu (reset hoàn toàn)
docker compose down -v
```

### Xem log khi có lỗi

```bash
cd /Users/tranhieu/ai-lesson-builder-service/deploy/langfuse

# Xem log tất cả container
docker compose logs -f

# Xem log 1 container cụ thể
docker compose logs -f langfuse-web
```

---

## 3. Test nhanh toàn bộ hệ thống

```bash
# Gửi 1 ảnh, yêu cầu tạo bài học
curl -X POST http://localhost:30001/v3/lessons/generate \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "test-123",
    "image_urls": ["https://images.unsplash.com/photo-1588072432836-e10032774350?w=600"],
    "custom_prompt": "Tạo bài học từ vựng tiếng Anh",
    "optional_parent_config": {
      "child_name": "Bé Bin",
      "child_age": 7,
      "language": "vi"
    }
  }'
```

**Kết quả mong đợi:** JSON chứa `status: success` và mảng `lessons` có ít nhất 1 bài.

**Sau đó mở Langfuse** (http://localhost:3010 → Tracing → Traces) — sẽ thấy request vừa gửi với sơ đồ thời gian từng bước.

---

## 4. Cấu hình quan trọng trong .env

| Biến | Ý nghĩa | Bắt buộc? |
|------|---------|-----------|
| `OPENAI_API_KEY` | Key OpenAI để gọi AI | ✅ Có |
| `OPENAI_VISION_MODEL` | Model đọc ảnh (mặc định: gpt-5.6-terra) | Không |
| `OPENAI_GUARDRAIL_MODEL` | Model kiểm duyệt (mặc định: gpt-5.6-luna) | Không |
| `LANGFUSE_BASE_URL` | URL Langfuse (http://localhost:3010) | Không (nếu không dùng) |
| `LANGFUSE_PUBLIC_KEY` | Key public của Langfuse | Không |
| `LANGFUSE_SECRET_KEY` | Key secret của Langfuse | Không |
| `PROFILE_API_BASE_URL` | URL API lấy thông tin bé | Không (fail-open) |
| `MEM0_BASE_URL` | URL memory service | Không (fail-open) |

**"Fail-open"** nghĩa là: nếu service đó chết hoặc không cấu hình, hệ thống vẫn chạy bình thường — chỉ thiếu dữ liệu từ service đó thôi.

---

## 5. Xử lý sự cố thường gặp

### API trả lỗi 500

```bash
# Xem log chi tiết
tail -100 /tmp/server.log | grep -i error
```

### Langfuse không lên

```bash
cd deploy/langfuse
docker compose ps          # Xem container nào unhealthy
docker compose logs langfuse-web | tail -50   # Xem lỗi cụ thể
```

### Request chậm bất thường

1. Mở http://localhost:3010 → Tracing → Traces
2. Click vào request chậm
3. Xem sơ đồ thời gian — bước nào tô vàng/đỏ là thủ phạm

### Muốn sửa prompt (kịch bản AI)

1. Mở http://localhost:3010 → Prompts
2. Click vào prompt muốn sửa (ví dụ: `context_style_guideline_prompt`)
3. Tạo version mới → Gắn label `production` → Save
4. Service tự động dùng bản mới, không cần restart

---

## 6. Kiến trúc nhanh cho dev

```
app/
├── api/                    # HTTP endpoints
│   └── v3/                 # API version 3 (chính)
├── pipeline/
│   └── lesson_pipeline.py  # Điều phối toàn bộ flow
├── domains/
│   ├── vision_extract/     # Đọc ảnh + soạn bài (gọi OpenAI)
│   ├── lesson_generator/   # Build prompt, transform output
│   ├── profile/            # Lấy thông tin bé
│   └── memory/             # Lấy trí nhớ về bé
└── core/
    ├── config.py           # Đọc .env
    └── tracing.py          # Kết nối Langfuse

deploy/
└── langfuse/
    ├── docker-compose.yml  # Stack Langfuse
    └── .env                # Secrets (đã gitignore)
```

---

## 7. Lệnh hữu ích

```bash
# Restart API sau khi sửa code
pkill -f "uvicorn app.main:app" && sleep 1 && \
  source venv/bin/activate && \
  uvicorn app.main:app --host 0.0.0.0 --port 30001 --reload

# Xem tất cả container Docker đang chạy
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Xem chi phí OpenAI gần nhất trên Langfuse
# Mở http://localhost:3010 → Dashboard → Cost

# Export prompts từ Langfuse (backup)
# Mở http://localhost:3010 → Prompts → ... menu → Export
```
