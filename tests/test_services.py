"""
Service unit tests.
"""
import pytest
from app.services.template_matcher import (
    select_template,
    get_option_description,
    validate_template_subtype_compatibility,
)
from app.services.vision import detect_exercise_subtypes


class TestTemplateMatching:
    """Test template matching logic."""

    def test_select_reading_template(self):
        """Reading question should select reading comprehension template."""
        template = select_template(["reading_question"], "Read the passage...")
        assert template == "ptl_learn_reading_comprehension_v1"

    def test_select_vocabulary_template(self):
        """Vocabulary matching should select vocab flashcard template."""
        template = select_template(["vocabulary_matching"], "Match the words...")
        assert template == "ptl_learn_vocab_flashcard_v1"

    def test_select_default_template(self):
        """Grammar fill blank should select exercise solver template."""
        template = select_template(["grammar_fill_blank"], "Fill in the blanks...")
        assert template == "ptl_learn_exercise_solver_v1"

    def test_option_description(self):
        """Test option description generation."""
        assert get_option_description(["grammar_fill_blank"]) == "Điền ngữ pháp"
        assert get_option_description(["multiple_choice"]) == "Trắc nghiệm"
        assert get_option_description([]) == "Bài tập tổng hợp"


class TestExerciseDetection:
    """Test exercise subtype detection."""

    def test_detect_fill_blank(self):
        """Detect fill-in-blank patterns."""
        content = "Fill in: cat and horse………. Mary…………"
        subtypes = detect_exercise_subtypes(content)
        assert "grammar_fill_blank" in subtypes

    def test_detect_multiple_choice(self):
        """Detect multiple choice patterns."""
        content = "Choose the correct answer: A) cat B) dog C) bird"
        subtypes = detect_exercise_subtypes(content)
        assert "multiple_choice" in subtypes

    def test_detect_matching(self):
        """Detect matching patterns."""
        content = "Match the words with their meanings"
        subtypes = detect_exercise_subtypes(content)
        assert "vocabulary_matching" in subtypes

    def test_default_short_answer(self):
        """Default to short answer when nothing detected."""
        content = "Some random text without patterns"
        subtypes = detect_exercise_subtypes(content)
        assert "short_answer" in subtypes


class TestTemplateCompatibility:
    """Test template-subtype compatibility."""

    def test_exercise_solver_compatibility(self):
        """Exercise solver should be compatible with most subtypes."""
        assert validate_template_subtype_compatibility(
            "ptl_learn_exercise_solver_v1",
            ["grammar_fill_blank"]
        )
        assert validate_template_subtype_compatibility(
            "ptl_learn_exercise_solver_v1",
            ["multiple_choice"]
        )

    def test_reading_compatibility(self):
        """Reading template should be compatible with reading questions."""
        assert validate_template_subtype_compatibility(
            "ptl_learn_reading_comprehension_v1",
            ["reading_question"]
        )

    def test_talk_templates_allow_all(self):
        """Talk templates have no subtype restrictions."""
        assert validate_template_subtype_compatibility(
            "ptl_talk_roleplay_v1",
            ["any_subtype"]
        )
