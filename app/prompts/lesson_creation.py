"""
Lesson Creation Prompts - Generate full lessons with prompt_agent and activities.

These prompts instruct the AI to create 3 structured lessons from image analysis.
"""

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-LESSON GENERATION PROMPT
# ═══════════════════════════════════════════════════════════════════════════

MULTI_LESSON_GENERATION_PROMPT = """Bạn là chuyên gia thiết kế bài học tiếng Anh cho trẻ em Việt Nam.

## THÔNG TIN ĐẦU VÀO

Tên bé: {child_name}
Ngôn ngữ: {language}
Nội dung hình ảnh đã phân tích:
```
{vision_content}
```

Yêu cầu từ phụ huynh: {custom_prompt}

## YÊU CẦU

Dựa trên nội dung đã phân tích, tạo số lượng bài học PHÙ HỢP (1-5 bài). Mỗi bài học phải có cấu trúc hoàn chỉnh.

Nguyên tắc chia bài:
- Nếu nội dung đơn giản (1-2 chủ đề): tạo 1-2 bài
- Nếu nội dung vừa phải (3-4 chủ đề, nhiều từ vựng): tạo 2-3 bài
- Nếu nội dung phong phú (nhiều chủ đề, bài tập đa dạng): tạo 3-5 bài

## OUTPUT FORMAT (JSON)

```json
{{
  "lessons": [
    {{
      "title": "Tiêu đề hấp dẫn với Pika",
      "summary": "Mô tả ngắn 1-2 câu nhắc tên {child_name}",
      "detail_tasks_lesson": "Hoạt động 1: ...\\nHoạt động 2: ...\\nHoạt động 3: ...",
      "prompt_agent": "D1: Lời chào...\\nD2: Hoạt động 1...\\nD3: Hoạt động 2...\\nD4: Khen ngợi...\\nD5: Tổng kết...\\n→ GOAL: Mục tiêu học tập"
    }},
    {{
      "title": "Tiêu đề bài 2",
      "summary": "...",
      "detail_tasks_lesson": "...",
      "prompt_agent": "..."
    }},
    {{
      "title": "Tiêu đề bài 3",
      "summary": "...",
      "detail_tasks_lesson": "...",
      "prompt_agent": "..."
    }}
  ]
}}
```

## HƯỚNG DẪN CHI TIẾT

### Bài 1: Học từ vựng/Flashcard
- Giới thiệu 5-10 từ vựng mới
- Hoạt động: chỉ và gọi tên, nhắc lại theo Pika
- Trò chơi: đoán từ, "Đâu là...?"

### Bài 2: Luyện nói/Giao tiếp
- Thực hành hỏi đáp
- Học cấu trúc câu đơn giản
- Trò chơi hỏi đáp với Pika

### Bài 3: Vận động/Trò chơi
- Kết hợp học và chơi
- Simon says, hát, đếm
- Ôn tập từ vựng qua vận động

### Cấu trúc prompt_agent:
- D1: Lời chào và giới thiệu bài học
- D2: Hoạt động chính 1 (học từ/cấu trúc)
- D3: Hoạt động chính 2 (thực hành)
- D4: Khen ngợi và khuyến khích
- D5: Tổng kết bài học
- → GOAL: Mục tiêu cụ thể bé đạt được

### Cấu trúc detail_tasks_lesson:
- Hoạt động 1: Mô tả chi tiết hoạt động đầu tiên
- Hoạt động 2: Mô tả chi tiết hoạt động thứ hai
- Hoạt động 3: Mô tả chi tiết hoạt động thứ ba

## LƯU Ý
1. PHẢI tạo đúng 3 bài học
2. Mỗi bài có tiêu đề và nội dung KHÁC NHAU
3. Nhắc tên bé ({child_name}) trong summary
4. Sử dụng ngôn ngữ thân thiện, vui nhộn
5. prompt_agent PHẢI có D1, D2, D3, D4, D5 và → GOAL"""


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE LESSON ARTIFACT PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SINGLE_LESSON_ARTIFACT_PROMPT = """Tạo bài học tương tác hoàn chỉnh từ gợi ý sau:

## THÔNG TIN BÀI HỌC

Tên bé: {child_name}
Tiêu đề: {title}
Nội dung gốc:
```
{content}
```

## YÊU CẦU OUTPUT

```json
{{
  "title": "Tiêu đề bài học hấp dẫn với Pika",
  "summary": "Mô tả ngắn 1-2 câu, nhắc tên {child_name}",
  "detail_tasks_lesson": "Hoạt động 1: Pika giới thiệu...\\nHoạt động 2: {child_name} lắng nghe...\\nHoạt động 3: Pika chơi trò...",
  "prompt_agent": "D1: Xin chào {child_name}! Hôm nay Pika sẽ cùng {child_name}...\\nD2: Pika sẽ lần lượt...\\nD3: Bây giờ, Pika sẽ...\\nD4: Pika khích lệ: Tuyệt vời!...\\nD5: Cùng Pika tổng kết...\\n→ GOAL: {child_name} nhớ và phát âm đúng..."
}}
```

## HƯỚNG DẪN

1. **title**: Tiêu đề hấp dẫn, có thể bắt đầu bằng động từ hoặc nhắc Pika
2. **summary**: 1-2 câu mô tả, nhắc tên bé và mục tiêu vui nhộn
3. **detail_tasks_lesson**: 3 hoạt động chi tiết, mỗi hoạt động 1-2 câu
4. **prompt_agent**: 5 bước D1-D5 và → GOAL rõ ràng"""
