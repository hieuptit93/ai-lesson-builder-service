# AI Lesson Builder Service

🎓 AI service for generating educational lessons from images.

## Features

- **Vision Analysis**: Analyze images to extract educational content
- **Exercise Detection**: Automatically detect exercise types (fill-in-blank, multiple choice, etc.)
- **Template Matching**: Map content to appropriate lesson templates
- **Multi-Agent Pipeline**: Vision → Curriculum → Psychology → Safety → Final
- **SSE Streaming**: Real-time progress updates during generation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/lessons/generate` | POST | Direct generation from image + prompt (SSE) |
| `/v3/lessons/generate` | POST | Suggest lessons from images (JSON) |
| `/v3/lessons/generate_artifact` | POST | Generate full lessons from suggestions (SSE) |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation (Swagger UI) |

## Quick Start

### 1. Setup Environment

```bash
# Clone and enter directory
cd ai-lesson-builder-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment file
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configure API Keys

Edit `.env`:

```env
# Use OpenAI
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o

# Or use Anthropic
# AI_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-your-key-here
# ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### 3. Run the Service

```bash
# Development mode (with auto-reload)
python run.py

# Or with uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate lesson from image (streaming)
curl -X POST http://localhost:8000/v1/lessons/generate \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "test-profile-123",
    "image_urls": ["https://example.com/worksheet.jpg"],
    "custom_prompt": "Học từ vựng về động vật",
    "optional_parent_config": {"language": "bi"},
    "stream": true
  }'
```

## Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## API Examples

### 1. Direct Generation (v1)

```bash
curl -X POST 'http://localhost:8000/v1/lessons/generate' \
  -H 'Content-Type: application/json' \
  -d '{
    "profile_id": "019caf62-1112-7a89-9c37-155d0ce06646",
    "image_urls": [
      "https://smedia.stepup.edu.vn/pika_robot/photo_to_lesson/img/bodaypart.jpg"
    ],
    "custom_prompt": "Học tên các bộ phận cơ thể người bằng tiếng Anh",
    "optional_parent_config": {"language": "bi"},
    "stream": true
  }'
```

### 2. Suggest Lessons (v3)

```bash
curl -X POST 'http://localhost:8000/v3/lessons/generate' \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "unique-request-123",
    "profile_id": "019db5b5-3667-7009-a4db-c36b48280c32",
    "image_urls": [
      "https://example.com/pdf-page-1.png",
      "https://example.com/pdf-page-2.png"
    ],
    "optional_parent_config": {"language": "vi"}
  }'
```

### 3. Generate Artifacts (v3)

```bash
curl -X POST 'http://localhost:8000/v3/lessons/generate_artifact' \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "unique-request-123",
    "profile_id": "019db5b5-3667-7009-a4db-c36b48280c32",
    "stream": true,
    "lessons": [
      {
        "title": "Lesson 1: Fill in the Blanks with Pronouns",
        "agent_mode": "learn_agent",
        "template_id": "ptl_learn_exercise_solver_v1",
        "exercise_subtypes": ["grammar_fill_blank"],
        "content": "cat and horse………. Mary………… Tom…………",
        "option": "Điền ngữ pháp"
      }
    ]
  }'
```

## SSE Stream Format

```
data: {"phase":"started","message":"Bắt đầu phân tích..."}\n\n
data: {"phase":"thinking","expert":"vision_analyst","content":"..."}\n\n
data: {"phase":"thinking","expert":"curriculum_designer","content":"..."}\n\n
data: {"phase":"thinking","expert":"child_psychologist","content":"..."}\n\n
data: {"phase":"thinking","expert":"safety_reviewer","content":"..."}\n\n
data: {"phase":"thinking","expert":"final_editor","content":"..."}\n\n
data: {"phase":"complete","data":{"lessons":[...]}}\n\n
data: [DONE]\n\n
```

## Template Registry

| Template ID | Name | Category |
|-------------|------|----------|
| `ptl_learn_vocab_flashcard_v1` | Từ vựng | learn |
| `ptl_learn_phonics_pronunciation_v1` | Phát âm | learn |
| `ptl_learn_exercise_solver_v1` | Giải bài tập | learn |
| `ptl_learn_sentence_pattern_practice_v1` | Luyện mẫu câu | learn |
| `ptl_learn_reading_comprehension_v1` | Đọc hiểu | learn |
| `ptl_talk_roleplay_v1` | Nhập vai hội thoại | talk |
| `ptl_talk_speaking_presentation_v1` | Thuyết trình | talk |
| `ptl_talk_storytelling_v1` | Kể chuyện sáng tạo | talk |

## Exercise Subtypes

| Subtype | Name (VI) |
|---------|-----------|
| `grammar_fill_blank` | Điền ngữ pháp |
| `multiple_choice` | Trắc nghiệm |
| `spelling` | Chính tả |
| `vocabulary_matching` | Nối từ vựng |
| `sentence_ordering` | Sắp xếp câu |
| `short_answer` | Trả lời ngắn |
| `reading_question` | Đọc hiểu |
| `classification` | Phân loại |

## Project Structure

```
ai-lesson-builder-service/
├── app/
│   ├── config/
│   │   ├── settings.py      # Environment config
│   │   └── templates.py     # Template registry
│   ├── models/
│   │   ├── request.py       # Request DTOs
│   │   └── response.py      # Response DTOs
│   ├── routes/
│   │   ├── v1_generate.py   # /v1/lessons/generate
│   │   ├── v3_generate.py   # /v3/lessons/generate
│   │   └── v3_artifact.py   # /v3/lessons/generate_artifact
│   ├── services/
│   │   ├── ai_client.py     # OpenAI/Anthropic client
│   │   ├── vision.py        # Image analysis
│   │   ├── template_matcher.py  # Template selection
│   │   ├── lesson_generator.py  # Lesson generation
│   │   └── sse_streamer.py  # SSE streaming
│   └── main.py              # FastAPI app
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Development

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Format code
pip install black isort
black app/
isort app/
```

## License

MIT
