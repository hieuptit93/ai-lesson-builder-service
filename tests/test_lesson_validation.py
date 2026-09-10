"""Tests for cross-field lesson validation."""

from app.utils.lesson_validation import (
    ERROR,
    WARNING,
    count_activities,
    count_d_steps,
    extract_taught_words,
    find_checkpoint_answer_leaks,
    find_inline_answer_dlines,
    is_math_lesson,
    is_vocabulary_lesson,
    summarize_reports,
    validate_lesson_consistency,
    validate_lessons,
)


def _codes(report, severity=None):
    issues = report.issues if severity is None else [i for i in report.issues if i.severity == severity]
    return {i.code for i in issues}


# A lesson that satisfies every rule - the baseline for "no issues".
GOOD_LESSON = {
    "title": "Lesson 1: Fill in the Blanks with Pronouns",
    "summary": "Bé sẽ học cách điền đại từ vào chỗ trống qua các câu đơn giản.",
    "detail_tasks_lesson": (
        "Hoạt động 1: Pika giới thiệu các đại từ HE, SHE, IT\n"
        "Hoạt động 2: Bé điền đại từ phù hợp vào chỗ trống\n"
        "Hoạt động 3: Pika kiểm tra và sửa lỗi cùng bé"
    ),
    "prompt_agent": (
        "D1: Chào bé! Hôm nay Pika sẽ cùng bé học về đại từ.\n"
        "D2: Pika đọc từng câu và bé chọn đại từ phù hợp nhé.\n"
        "D3: Câu 1 dùng HE hay SHE, bé chọn đáp án nào?\n"
        "D4: Câu 2 dùng IT phải không bé?\n"
        "D5: Pika kiểm tra lại các đáp án cùng bé.\n"
        "→ GOAL: Bé hoàn thành bài tập điền đại từ chính xác."
    ),
}


class TestGoodLesson:
    def test_clean_lesson_has_no_issues(self):
        report = validate_lesson_consistency(GOOD_LESSON, 1)
        assert report.is_valid
        assert report.issues == (), f"unexpected issues: {[i.code for i in report.issues]}"

    def test_report_metadata(self):
        report = validate_lesson_consistency(GOOD_LESSON, 7)
        assert report.lesson_index == 7
        assert report.lesson_title.startswith("Lesson 1")


class TestRequiredFields:
    def test_missing_prompt_agent_is_error(self):
        lesson = {**GOOD_LESSON, "prompt_agent": ""}
        report = validate_lesson_consistency(lesson)
        assert not report.is_valid
        assert "missing_field" in _codes(report, ERROR)

    def test_whitespace_only_counts_as_missing(self):
        lesson = {**GOOD_LESSON, "summary": "   "}
        report = validate_lesson_consistency(lesson)
        assert "missing_field" in _codes(report, ERROR)

    def test_empty_lesson_reports_all_fields(self):
        report = validate_lesson_consistency({})
        missing = [i for i in report.errors if i.code == "missing_field"]
        assert len(missing) == 4  # title, summary, detail_tasks_lesson, prompt_agent


class TestPromptAgentStructure:
    def test_no_dsteps_is_error(self):
        lesson = {**GOOD_LESSON, "prompt_agent": "Pika sẽ dạy bé về đại từ. → GOAL: xong bài."}
        report = validate_lesson_consistency(lesson)
        assert "prompt_agent_no_dsteps" in _codes(report, ERROR)

    def test_no_terminator_is_error(self):
        lesson = {**GOOD_LESSON, "prompt_agent": "D1: Chào bé.\nD2: Học đại từ.\nD3: Xong rồi."}
        report = validate_lesson_consistency(lesson)
        assert "prompt_agent_no_terminator" in _codes(report, ERROR)

    def test_arrow_variants_both_accepted(self):
        for arrow in ("→ GOAL: xong", "-> GOAL: xong"):
            lesson = {**GOOD_LESSON, "prompt_agent": f"D1: a\nD2: b\nD3: c\n{arrow}"}
            report = validate_lesson_consistency(lesson)
            assert "prompt_agent_no_terminator" not in _codes(report)

    def test_too_many_dsteps_is_warning_not_error(self):
        steps = "\n".join(f"D{i}: bước {i}" for i in range(1, 12))
        lesson = {**GOOD_LESSON, "prompt_agent": f"{steps}\n→ GOAL: xong"}
        report = validate_lesson_consistency(lesson)
        assert "dstep_count_out_of_range" in _codes(report, WARNING)
        assert report.is_valid  # warnings never invalidate


class TestMathLessons:
    MATH_LESSON = {
        "title": "Lesson 2: Phép chia có dư",
        "summary": "Bé học phép chia có dư với các số nhỏ.",
        "detail_tasks_lesson": (
            "Hoạt động 1: Pika giải thích phép chia\n"
            "Hoạt động 2: Bé cùng Pika tính\n"
            "Hoạt động 3: Pika kiểm tra kết quả"
        ),
        "prompt_agent": (
            "D1: Pika cùng bé học phép chia nhé.\n"
            "D2: Mười hai chia năm được hai dư hai.\n"
            "D3: Pika nhắc lại kết quả cùng bé.\n"
            "→ ANSWER: mười hai chia năm bằng hai dư hai."
        ),
    }

    def test_detects_math_lesson(self):
        assert is_math_lesson(self.MATH_LESSON)
        assert not is_math_lesson(GOOD_LESSON)

    def test_math_with_answer_is_valid(self):
        report = validate_lesson_consistency(self.MATH_LESSON)
        assert report.is_valid, [i.code for i in report.errors]

    def test_math_without_answer_is_error(self):
        lesson = {
            **self.MATH_LESSON,
            "prompt_agent": (
                "D1: Pika cùng bé học phép chia.\n"
                "D2: Bé thử tính nhé.\n"
                "D3: Xong rồi.\n"
                "→ GOAL: bé biết chia."
            ),
        }
        report = validate_lesson_consistency(lesson)
        assert "math_missing_answer" in _codes(report, ERROR)

    def test_vague_math_step_is_warning(self):
        lesson = {
            **self.MATH_LESSON,
            "prompt_agent": (
                "D1: Pika cùng bé học phép chia.\n"
                "D2: Hỏi trẻ làm bước tiếp theo.\n"
                "D3: Xong.\n"
                "→ ANSWER: hai dư hai."
            ),
        }
        report = validate_lesson_consistency(lesson)
        assert "vague_math_step" in _codes(report, WARNING)


class TestTaughtWordSync:
    def test_word_missing_from_detail_is_warning(self):
        lesson = {
            **GOOD_LESSON,
            "detail_tasks_lesson": (
                "Hoạt động 1: Pika giới thiệu đại từ\n"
                "Hoạt động 2: Bé điền vào chỗ trống\n"
                "Hoạt động 3: Pika kiểm tra"
            ),
        }
        report = validate_lesson_consistency(lesson)
        assert "taught_word_not_in_detail" in _codes(report, WARNING)
        assert report.is_valid

    def test_words_present_in_detail_pass(self):
        report = validate_lesson_consistency(GOOD_LESSON)
        assert "taught_word_not_in_detail" not in _codes(report)

    def test_vocab_over_five_words_is_warning(self):
        words = ["APPLE", "BANANA", "CHERRY", "DATE", "ELDERBERRY", "FIG", "GRAPE"]
        lesson = {
            "title": "Lesson 3: Từ vựng về trái cây",
            "summary": "Bé học từ vựng về trái cây cùng Pika.",
            "detail_tasks_lesson": (
                f"Hoạt động 1: Pika giới thiệu {', '.join(words)}\n"
                "Hoạt động 2: Bé nhắc lại từng từ\n"
                "Hoạt động 3: Pika kiểm tra"
            ),
            "prompt_agent": (
                f"D1: Pika dạy bé các từ {', '.join(words)}.\n"
                "D2: Bé nhắc lại cùng Pika.\n"
                "D3: Pika kiểm tra lại.\n"
                "→ GOAL: bé nhớ các từ."
            ),
        }
        report = validate_lesson_consistency(lesson)
        assert is_vocabulary_lesson(lesson)
        assert "vocab_word_limit_exceeded" in _codes(report, WARNING)


class TestVoiceOnlyConstraint:
    def test_look_at_picture_is_warning(self):
        lesson = {
            **GOOD_LESSON,
            "prompt_agent": (
                "D1: Bé nhìn vào hình và cho Pika biết đó là gì.\n"
                "D2: Pika giải thích.\n"
                "D3: Xong.\n"
                "→ GOAL: bé hiểu bài."
            ),
        }
        report = validate_lesson_consistency(lesson)
        assert "visual_reference" in _codes(report, WARNING)

    def test_watch_video_is_warning(self):
        lesson = {
            **GOOD_LESSON,
            "detail_tasks_lesson": (
                "Hoạt động 1: Bé xem video về động vật\n"
                "Hoạt động 2: Bé kể lại\n"
                "Hoạt động 3: Pika kiểm tra"
            ),
        }
        report = validate_lesson_consistency(lesson)
        assert "visual_reference" in _codes(report, WARNING)

    def test_clean_lesson_has_no_visual_reference(self):
        report = validate_lesson_consistency(GOOD_LESSON)
        assert "visual_reference" not in _codes(report)


class TestActivityCount:
    def test_two_activities_is_warning(self):
        lesson = {
            **GOOD_LESSON,
            "detail_tasks_lesson": "Hoạt động 1: HE và SHE\nHoạt động 2: IT",
        }
        report = validate_lesson_consistency(lesson)
        assert "activity_count_mismatch" in _codes(report, WARNING)


class TestExtractionHelpers:
    def test_extract_quoted_words(self):
        assert extract_taught_words('Pika dạy từ "apple" cho bé') == {"apple"}

    def test_extract_allcaps_words(self):
        assert extract_taught_words("Học HE, SHE, IT hôm nay") == {"he", "she", "it"}

    def test_extract_slash_options(self):
        assert extract_taught_words("Chọn (He/She/It) nhé") == {"he", "she", "it"}

    def test_format_markers_excluded(self):
        # GOAL / PIKA are format noise, not vocabulary
        assert extract_taught_words("→ GOAL: PIKA xong bài") == set()

    def test_parenthetical_gloss_not_treated_as_options(self):
        # No slash - a Vietnamese gloss, not an option list
        assert "quả táo" not in extract_taught_words("apple (quả táo)")

    def test_vietnamese_text_yields_no_false_words(self):
        text = "Pika sẽ cùng bé ôn lại cách dùng đại từ thông qua bài tập điền từ."
        assert extract_taught_words(text) == set()

    def test_count_d_steps(self):
        assert count_d_steps("D1: a\nD2: b\nD3: c") == 3
        assert count_d_steps("no steps here") == 0

    def test_count_activities(self):
        assert count_activities("Hoạt động 1: a\nHoạt động 2: b") == 2
        assert count_activities("") == 0


def _spelling_ckp(question: str, hint: str, letter: str = "d", word: str = "door") -> dict:
    return {
        "type": "cta",
        "name": f"Item: {word}",
        "question": question,
        "response_guide": (
            f"• Lần 1 — Case 1 (says the letter '{letter}' or the word '{word}'): Đúng rồi! <next_ckp/>\n"
            f"• Lần 1 — Case 2 (incorrect or no answer): {hint}\n"
            f"• Lần 2 — Case 3 (any): Chữ cái đầu là <eng>{letter}</eng>, ghép lại thành <eng>{word}</eng>. <next_ckp/>"
        ),
    }


class TestCheckpointAnswerLeaks:
    def test_clean_vietnamese_checkpoint_has_no_leak(self):
        ckp = _spelling_ckp(
            "Hình tiếp theo: chấm chấm <eng>o o r</eng>. Chữ cái đầu là gì?",
            "Gợi ý: đây là vật mình mở ra để vào phòng. Thử lại nhé!",
        )
        assert find_checkpoint_answer_leaks([ckp]) == []

    def test_bilingual_echo_naming_the_object_is_a_hint_leak(self):
        ckp = _spelling_ckp(
            "Nhìn hình một cái cửa: chấm chấm <eng>o o r</eng>. Chữ cái đầu là gì?",
            "Gợi ý: đây là vật để mở ra vào phòng. Thử lại nhé! <eng>This is a door. Try again!</eng>",
        )
        assert find_checkpoint_answer_leaks([ckp]) == [
            {"checkpoint": "Item: door", "field": "hint", "word": "door"}
        ]

    def test_english_question_naming_the_object_is_a_question_leak(self):
        ckp = _spelling_ckp(
            "<eng>Look at the picture of a door: _oor. What is the first letter?</eng>",
            "<eng>Here's a hint: something you open to go into a room. Try again!</eng>",
        )
        leaks = find_checkpoint_answer_leaks([ckp])
        assert [(l["field"], l["word"]) for l in leaks] == [("question", "door")]

    def test_spelling_the_whole_word_letter_by_letter_is_a_leak(self):
        ckp = _spelling_ckp(
            "Hình con voi: chấm chấm <eng>l e p h a n t</eng>. Nếu đánh vần, sẽ là "
            "<eng>e, l, e, p, h, a, n, t</eng>. Chữ cái đầu là gì?",
            "Gợi ý: con vật to có vòi dài. Thử lại nhé!",
            letter="e",
            word="elephant",
        )
        assert find_checkpoint_answer_leaks([ckp]) == [
            {"checkpoint": "Item: elephant", "field": "question", "word": "elephant"}
        ]

    def test_reading_only_the_printed_letters_is_not_a_leak(self):
        ckp = _spelling_ckp(
            "Hình con voi: chấm chấm <eng>l e p h a n t</eng>. Chữ cái đầu là gì?",
            "Gợi ý: con vật to có vòi dài. <eng>Try again!</eng>",
            letter="e",
            word="elephant",
        )
        assert find_checkpoint_answer_leaks([ckp]) == []

    def test_given_letters_and_single_letter_answer_are_not_leaks(self):
        ckp = _spelling_ckp(
            "Chấm chấm <eng>o o r</eng>. Chữ cái đầu là chữ gì?",
            "Gợi ý: chữ này cũng mở đầu từ <eng>duck</eng>. Thử lại nhé!",
        )
        assert find_checkpoint_answer_leaks([ckp]) == []

    def test_multiple_choice_question_listing_options_is_not_flagged(self):
        ckp = {
            "type": "cta",
            "name": "Exercise 1",
            "question": "<eng>_______ she like apples?</eng> Chọn: <eng>Do, Does, Did,</eng> hay <eng>Are?</eng>",
            "response_guide": (
                "• Lần 1 — Case 1 (says 'Does'): Đúng rồi! <next_ckp/>\n"
                "• Lần 1 — Case 2 (incorrect): Gợi ý: chủ ngữ 'she' là ngôi thứ ba số ít. Thử lại nhé!\n"
                "• Lần 2 — Case 3 (any): Đáp án là 'Does'. <next_ckp/>"
            ),
        }
        assert find_checkpoint_answer_leaks([ckp]) == []

    def test_narratives_and_missing_guides_are_skipped(self):
        ckps = [
            {"type": "narrative", "name": "Intro", "question": "door door door", "response_guide": None},
            {"type": "cta", "name": "No guide", "question": "door", "response_guide": None},
        ]
        assert find_checkpoint_answer_leaks(ckps) == []


HIDDEN_KEY = "[Đáp án — chỉ để Pika kiểm tra, chỉ đọc sau khi đã nói hết Gợi ý 1 và Gợi ý 2 mà bé vẫn sai: d → door]"


class TestInlineAnswerDlines:
    def _plan(self, prompt_agent: str) -> dict:
        return {"lessons": [{"lesson_id": "lesson_001", "prompt_agent": prompt_agent}]}

    def test_compliant_hidden_key_format_passes(self):
        plan = self._plan(
            "D1: Chào Bo!\n"
            f"D2: Mục 1 — hình cái cửa — chấm chấm o o r. Bo đoán chữ cái đầu là gì? Gợi ý 1: vật mình mở ra để vào phòng. Gợi ý 2: từ này có bốn chữ cái. {HIDDEN_KEY}\n"
            "→ GOAL: Bo điền đúng."
        )
        assert find_inline_answer_dlines(plan) == []

    def test_answer_right_after_question_is_flagged(self):
        plan = self._plan("D3: Pika hỏi: 'Từ _oy là gì? Đáp án: boy.'\nD4: Tiếp tục: 'Từ _oor là gì? Đáp án: door.'")
        assert [(p["dline"], p["problem"]) for p in find_inline_answer_dlines(plan)] == [
            ("D3", "inline_answer"),
            ("D4", "inline_answer"),
        ]

    def test_parenthesised_letter_is_flagged(self):
        plan = self._plan("D3: Bo đoán và điền chữ cái phù hợp nhé: _oy (b), _oor (d).")
        assert find_inline_answer_dlines(plan)[0]["problem"] == "inline_answer"

    def test_hidden_key_without_scripted_hints_is_flagged(self):
        plan = self._plan(f"D2: Mục 1 — hình cái cửa — _oor. Bo đoán chữ cái đầu là gì? {HIDDEN_KEY}")
        assert find_inline_answer_dlines(plan) == [
            {"lesson_id": "lesson_001", "dline": "D2", "problem": "missing_hints"}
        ]

    def test_non_item_dlines_are_ignored(self):
        plan = self._plan("D1: Chào Bo! Hôm nay mình luyện điền chữ cái đầu.\nD7: Pika khen Bo.\n→ GOAL: xong.")
        assert find_inline_answer_dlines(plan) == []


class TestBatchValidation:
    def test_validate_lessons_indexes_from_one(self):
        reports = validate_lessons([GOOD_LESSON, GOOD_LESSON])
        assert [r.lesson_index for r in reports] == [1, 2]

    def test_non_dict_entries_skipped(self):
        reports = validate_lessons([GOOD_LESSON, "not a dict", None])
        assert len(reports) == 1

    def test_summary_all_valid(self):
        summary = summarize_reports(validate_lessons([GOOD_LESSON, GOOD_LESSON]))
        assert summary["all_valid"] is True
        assert summary["lessons_checked"] == 2
        assert summary["lessons_with_errors"] == 0

    def test_summary_flags_invalid_lesson(self):
        bad = {**GOOD_LESSON, "prompt_agent": ""}
        summary = summarize_reports(validate_lessons([GOOD_LESSON, bad]))
        assert summary["all_valid"] is False
        assert summary["invalid_lesson_indexes"] == [2]
        assert "missing_field" in summary["error_codes"]

    def test_summary_empty_input(self):
        summary = summarize_reports([])
        assert summary["all_valid"] is True
        assert summary["lessons_checked"] == 0
