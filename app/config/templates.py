"""
Template and Exercise Subtype Registry.

These define the available lesson templates and exercise types
that the AI can detect and generate.
"""

# ═══════════════════════════════════════════════════════════════════════════
# TEMPLATE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

TEMPLATE_REGISTRY = {
    # LEARN TEMPLATES
    "ptl_learn_vocab_flashcard_v1": {
        "name": "Từ vựng",
        "name_en": "Vocabulary Flashcard",
        "category": "learn",
        "description": "Học từ vựng qua flashcard",
        "compatible_subtypes": ["vocabulary_matching", "spelling", "multiple_choice"],
    },
    "ptl_learn_phonics_pronunciation_v1": {
        "name": "Phát âm",
        "name_en": "Phonics & Pronunciation",
        "category": "learn",
        "description": "Luyện phát âm phonics",
        "compatible_subtypes": ["spelling"],
    },
    "ptl_learn_exercise_solver_v1": {
        "name": "Giải bài tập",
        "name_en": "Exercise Solver",
        "category": "learn",
        "description": "Giải các dạng bài tập",
        "compatible_subtypes": [
            "grammar_fill_blank",
            "multiple_choice",
            "spelling",
            "vocabulary_matching",
            "sentence_ordering",
            "short_answer",
            "reading_question",
            "classification",
        ],
    },
    "ptl_learn_sentence_pattern_practice_v1": {
        "name": "Luyện mẫu câu",
        "name_en": "Sentence Pattern Practice",
        "category": "learn",
        "description": "Luyện tập cấu trúc câu",
        "compatible_subtypes": ["grammar_fill_blank", "sentence_ordering", "short_answer"],
    },
    "ptl_learn_reading_comprehension_v1": {
        "name": "Đọc hiểu",
        "name_en": "Reading Comprehension",
        "category": "learn",
        "description": "Đọc và trả lời câu hỏi",
        "compatible_subtypes": ["reading_question", "multiple_choice", "short_answer"],
    },
    # TALK TEMPLATES
    "ptl_talk_roleplay_v1": {
        "name": "Nhập vai hội thoại",
        "name_en": "Roleplay Conversation",
        "category": "talk",
        "description": "Thực hành hội thoại theo tình huống",
        "compatible_subtypes": [],
    },
    "ptl_talk_speaking_presentation_v1": {
        "name": "Thuyết trình",
        "name_en": "Speaking Presentation",
        "category": "talk",
        "description": "Luyện thuyết trình",
        "compatible_subtypes": [],
    },
    "ptl_talk_storytelling_v1": {
        "name": "Kể chuyện sáng tạo",
        "name_en": "Creative Storytelling",
        "category": "talk",
        "description": "Sáng tạo và kể chuyện",
        "compatible_subtypes": [],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# EXERCISE SUBTYPES
# ═══════════════════════════════════════════════════════════════════════════

EXERCISE_SUBTYPES = {
    "grammar_fill_blank": {
        "name": "Điền ngữ pháp",
        "name_en": "Grammar Fill in the Blank",
        "description": "Điền từ/cụm từ vào chỗ trống",
        "keywords": ["fill in", "điền vào", "complete", "...", "___", "………", "______"],
    },
    "multiple_choice": {
        "name": "Trắc nghiệm",
        "name_en": "Multiple Choice",
        "description": "Chọn đáp án đúng",
        "keywords": ["choose", "select", "chọn", "A)", "B)", "a.", "b.", "circle"],
    },
    "spelling": {
        "name": "Chính tả",
        "name_en": "Spelling",
        "description": "Viết đúng chính tả",
        "keywords": ["spell", "write", "viết", "correct spelling"],
    },
    "vocabulary_matching": {
        "name": "Nối từ vựng",
        "name_en": "Vocabulary Matching",
        "description": "Nối từ với nghĩa/hình ảnh",
        "keywords": ["match", "nối", "connect", "pair", "draw a line"],
    },
    "sentence_ordering": {
        "name": "Sắp xếp câu",
        "name_en": "Sentence Ordering",
        "description": "Sắp xếp từ thành câu hoàn chỉnh",
        "keywords": ["order", "arrange", "sắp xếp", "reorder", "put in order"],
    },
    "short_answer": {
        "name": "Trả lời ngắn",
        "name_en": "Short Answer",
        "description": "Viết câu trả lời ngắn",
        "keywords": ["answer", "trả lời", "write", "respond"],
    },
    "reading_question": {
        "name": "Câu hỏi đọc hiểu",
        "name_en": "Reading Question",
        "description": "Trả lời câu hỏi về bài đọc",
        "keywords": ["read", "đọc", "passage", "text", "đoạn văn", "comprehension"],
    },
    "classification": {
        "name": "Phân loại",
        "name_en": "Classification",
        "description": "Phân loại theo nhóm",
        "keywords": ["classify", "group", "phân loại", "sort", "category", "categorize"],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# OPTION DESCRIPTIONS (Vietnamese)
# ═══════════════════════════════════════════════════════════════════════════

OPTION_DESCRIPTIONS = {
    "grammar_fill_blank": "Điền ngữ pháp",
    "multiple_choice": "Trắc nghiệm",
    "spelling": "Chính tả",
    "vocabulary_matching": "Nối từ vựng",
    "sentence_ordering": "Sắp xếp câu",
    "short_answer": "Trả lời ngắn",
    "reading_question": "Đọc hiểu",
    "classification": "Phân loại",
}
