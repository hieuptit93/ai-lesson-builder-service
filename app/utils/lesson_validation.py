"""Cross-field consistency validation for generated lessons.

Pure Python - no LLM call, no network. Runs in ~1ms per lesson, so it is safe
to call inline before returning a response to the client.

The generation prompt (VISION_EXTRACTION_PROMPT_V3) asks the model to keep
`summary`, `detail_tasks_lesson` and `prompt_agent` describing the SAME
activities and target items. Structured Outputs enforces the *shape* of the
JSON but cannot enforce that agreement - that is what this module checks.

Two severities, deliberately separated:

- ERROR   structural and unambiguous (missing field, no D-steps, no terminator,
          math lesson with no verified answer). Safe to act on: block caching,
          trigger a retry, alert.
- WARNING soft cross-field signals that carry real false-positive risk because
          the text is Vietnamese with inline English. Useful in aggregate for
          prompt tuning; not safe to fail a request on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

ERROR = "error"
WARNING = "warning"

_REQUIRED_FIELDS = ("title", "summary", "detail_tasks_lesson", "prompt_agent")

# prompt_agent must be D-lines ending in a "→ GOAL" (or "→ ANSWER" for math).
_D_STEP_RE = re.compile(r"(?:^|\n)\s*D(\d+)\s*:", re.MULTILINE)
_TERMINATOR_RE = re.compile(r"(?:→|->)\s*(GOAL|ANSWER)\b", re.IGNORECASE)
_ANSWER_RE = re.compile(r"(?:→|->)\s*ANSWER\b", re.IGNORECASE)

# detail_tasks_lesson is specified as exactly 3 "Hoạt động N:" blocks.
_ACTIVITY_RE = re.compile(r"Hoạt\s*động\s*\d+\s*:", re.IGNORECASE)
_EXPECTED_ACTIVITIES = 3

# Spec says 5-7 D-lines; we widen to 3-8 so ordinary variation stays quiet.
_MIN_D_STEPS = 3
_MAX_D_STEPS = 8

# Vocabulary lessons: at most 5 taught words (prompt rule).
_MAX_VOCAB_WORDS = 5

# --- taught-word extraction -------------------------------------------------
# Vietnamese and English share many short ASCII tokens ("in", "an", "co", "la"),
# so a blanket "every ASCII word" sweep produces noise. We extract only words
# presented as *taught items*, which the prompt format marks unambiguously:
#   quoted ("apple"), ALL-CAPS (HE, SHE, IT), or slash options ((He/She/It)).
_QUOTED_RE = re.compile(r'["“‘\']([A-Za-z][A-Za-z\'\-]{1,})["”’\']')
_ALLCAPS_RE = re.compile(r"\b([A-Z][A-Z\'\-]{1,})\b")
_SLASH_OPTIONS_RE = re.compile(r"\(([A-Za-z][A-Za-z\s/\'\-]{2,})\)")

# ALL-CAPS tokens that are format markers, not taught vocabulary.
_CAPS_NOISE = {
    "GOAL", "ANSWER", "PIKA", "OK", "TODO", "JSON", "A", "B", "C", "D", "E",
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
}

# Voice-only constraint: the agent teaches by conversation and must never tell
# a child to look at an image, screen or video.
_VISUAL_REFERENCE_PATTERNS = (
    r"nhìn\s+vào\s+(hình|ảnh|tranh|màn\s*hình)",
    r"(xem|quan\s*sát)\s+(hình|ảnh|tranh|video|màn\s*hình)",
    r"nhìn\s+(hình|ảnh|tranh|màn\s*hình)",
    r"chỉ\s+vào\s+(hình|ảnh|tranh)",
    r"look\s+at\s+the\s+(picture|image|screen)",
)
_VISUAL_REFERENCE_RE = re.compile("|".join(_VISUAL_REFERENCE_PATTERNS), re.IGNORECASE)

# Math lessons must pre-solve every step; these phrases delegate the work.
_VAGUE_MATH_PATTERNS = (
    r"hỏi\s+trẻ\s+làm\s+bước\s+tiếp",
    r"tiếp\s+tục\s+từng\s+số",
    r"cho\s+trẻ\s+tự\s+tính",
    r"tính\s+tiếp\s+các\s+phép",
)
_VAGUE_MATH_RE = re.compile("|".join(_VAGUE_MATH_PATTERNS), re.IGNORECASE)

_MATH_KEYWORDS = (
    "toán", "phép tính", "phép cộng", "phép trừ", "phép nhân", "phép chia",
    "cộng", "trừ", "nhân", "chia", "số dư", "math", "addition", "subtraction",
    "multiplication", "division",
)
_VOCAB_KEYWORDS = ("từ vựng", "vocabulary", "từ mới")


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a lesson."""

    code: str
    severity: str
    message: str
    lesson_field: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """All issues found in a single lesson."""

    lesson_index: int
    lesson_title: str
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == WARNING)

    @property
    def is_valid(self) -> bool:
        """True when nothing structural is wrong (warnings do not count)."""
        return not self.errors

    def as_log_dict(self) -> dict:
        """Flat dict for structlog / metric emission."""
        return {
            "lesson_index": self.lesson_index,
            "lesson_title": self.lesson_title[:80],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "error_codes": [i.code for i in self.errors],
            "warning_codes": [i.code for i in self.warnings],
        }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
def extract_taught_words(text: str) -> set[str]:
    """Return lower-cased words the text presents as taught vocabulary.

    Only high-signal presentations count (quoted, ALL-CAPS, slash options), so
    the result is precise rather than exhaustive. Returns an empty set when the
    text teaches no explicitly marked words - callers then skip the sync check.
    """
    if not text:
        return set()

    words: set[str] = set()

    for match in _QUOTED_RE.findall(text):
        words.add(match.lower())

    for match in _ALLCAPS_RE.findall(text):
        if match not in _CAPS_NOISE and len(match) >= 2:
            words.add(match.lower())

    for group in _SLASH_OPTIONS_RE.findall(text):
        if "/" not in group:
            continue  # a parenthetical gloss, not an option list
        for part in group.split("/"):
            token = part.strip()
            if token.isalpha() and len(token) >= 2:
                words.add(token.lower())

    return words


def count_d_steps(prompt_agent: str) -> int:
    """Number of D-lines (D1:, D2:, ...) in prompt_agent."""
    return len(_D_STEP_RE.findall(prompt_agent or ""))


def count_activities(detail_tasks: str) -> int:
    """Number of 'Hoạt động N:' blocks in detail_tasks_lesson."""
    return len(_ACTIVITY_RE.findall(detail_tasks or ""))


def _text_blob(lesson: dict) -> str:
    parts = (
        lesson.get("title", ""),
        lesson.get("summary", ""),
        lesson.get("detail_tasks_lesson", ""),
    )
    return " ".join(p for p in parts if p).lower()


def is_math_lesson(lesson: dict) -> bool:
    """Heuristic: does this lesson teach arithmetic?

    Rules:
    - talk_agent lessons are NEVER math lessons (they're conversation-based)
    - Must have math keywords in content
    - Avoid false positives from words like "chia sẻ" (share) containing "chia" (divide)
    """
    # talk_agent is for conversation/dialogue - never math drill
    agent_mode = str(lesson.get("agent_mode", "") or "").lower()
    if agent_mode == "talk_agent":
        return False

    # template_id can also indicate non-math
    template_id = str(lesson.get("template_id", "") or "").lower()
    if "talk" in template_id or "conversation" in template_id or "speaking" in template_id:
        return False

    blob = _text_blob(lesson)

    # Require stronger signal: keyword + number patterns or explicit math context
    has_math_keyword = any(keyword in blob for keyword in _MATH_KEYWORDS)
    if not has_math_keyword:
        return False

    # Filter out false positives: "chia sẻ", "chia tay", etc.
    # If the word "chia" appears but followed by common non-math words, not a math lesson
    if "chia sẻ" in blob or "chia tay" in blob:
        # Check if there's ALSO real math context (numbers, equations)
        import re
        has_numbers = bool(re.search(r"\d\s*[+\-×÷=]\s*\d|\d\s*(cộng|trừ|nhân|chia)\s*\d", blob))
        if not has_numbers:
            return False

    return True


def is_vocabulary_lesson(lesson: dict) -> bool:
    """Heuristic: is this a vocabulary lesson (max 5 words rule applies)?"""
    blob = _text_blob(lesson)
    return any(keyword in blob for keyword in _VOCAB_KEYWORDS)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_lesson_consistency(lesson: dict, lesson_index: int = 0) -> ValidationReport:
    """Check one lesson's cross-field consistency and structure.

    Never raises - a malformed lesson comes back as a report full of errors.
    """
    issues: list[ValidationIssue] = []
    title = str(lesson.get("title", "") or "")

    # --- required fields (ERROR) -------------------------------------------
    for name in _REQUIRED_FIELDS:
        value = lesson.get(name)
        if not value or not str(value).strip():
            issues.append(
                ValidationIssue(
                    code="missing_field",
                    severity=ERROR,
                    message=f"Field '{name}' is missing or empty",
                    lesson_field=name,
                )
            )

    prompt_agent = str(lesson.get("prompt_agent", "") or "")
    detail_tasks = str(lesson.get("detail_tasks_lesson", "") or "")

    # --- prompt_agent structure (ERROR) ------------------------------------
    d_step_count = count_d_steps(prompt_agent)
    if prompt_agent and d_step_count == 0:
        issues.append(
            ValidationIssue(
                code="prompt_agent_no_dsteps",
                severity=ERROR,
                message="prompt_agent contains no D-steps (expected 'D1:', 'D2:', ...)",
                lesson_field="prompt_agent",
            )
        )

    if prompt_agent and not _TERMINATOR_RE.search(prompt_agent):
        issues.append(
            ValidationIssue(
                code="prompt_agent_no_terminator",
                severity=ERROR,
                message="prompt_agent has no '→ GOAL' (or '→ ANSWER') closing line",
                lesson_field="prompt_agent",
            )
        )

    # --- math lessons must carry a verified answer (ERROR) -----------------
    if prompt_agent and is_math_lesson(lesson) and not _ANSWER_RE.search(prompt_agent):
        issues.append(
            ValidationIssue(
                code="math_missing_answer",
                severity=ERROR,
                message="Math lesson has no '→ ANSWER' line with the verified result",
                lesson_field="prompt_agent",
            )
        )

    # --- D-step count (WARNING) --------------------------------------------
    if d_step_count and not (_MIN_D_STEPS <= d_step_count <= _MAX_D_STEPS):
        issues.append(
            ValidationIssue(
                code="dstep_count_out_of_range",
                severity=WARNING,
                message=(
                    f"prompt_agent has {d_step_count} D-steps, "
                    f"expected {_MIN_D_STEPS}-{_MAX_D_STEPS}"
                ),
                lesson_field="prompt_agent",
            )
        )

    # --- activity count (WARNING) ------------------------------------------
    activity_count = count_activities(detail_tasks)
    if detail_tasks and activity_count != _EXPECTED_ACTIVITIES:
        issues.append(
            ValidationIssue(
                code="activity_count_mismatch",
                severity=WARNING,
                message=(
                    f"detail_tasks_lesson lists {activity_count} activities, "
                    f"expected {_EXPECTED_ACTIVITIES}"
                ),
                lesson_field="detail_tasks_lesson",
            )
        )

    # --- taught-word sync (WARNING) ----------------------------------------
    # Any word prompt_agent teaches must also be named in detail_tasks_lesson.
    prompt_words = extract_taught_words(prompt_agent)
    if prompt_words:
        detail_blob = detail_tasks.lower()
        missing = sorted(w for w in prompt_words if w not in detail_blob)
        if missing:
            issues.append(
                ValidationIssue(
                    code="taught_word_not_in_detail",
                    severity=WARNING,
                    message=(
                        "Words taught in prompt_agent are absent from "
                        f"detail_tasks_lesson: {', '.join(missing[:8])}"
                    ),
                    lesson_field="detail_tasks_lesson",
                )
            )

        if is_vocabulary_lesson(lesson) and len(prompt_words) > _MAX_VOCAB_WORDS:
            issues.append(
                ValidationIssue(
                    code="vocab_word_limit_exceeded",
                    severity=WARNING,
                    message=(
                        f"Vocabulary lesson teaches {len(prompt_words)} words, "
                        f"max is {_MAX_VOCAB_WORDS}"
                    ),
                    lesson_field="prompt_agent",
                )
            )

    # --- voice-only constraint (WARNING) -----------------------------------
    for name, text in (("prompt_agent", prompt_agent), ("detail_tasks_lesson", detail_tasks)):
        match = _VISUAL_REFERENCE_RE.search(text)
        if match:
            issues.append(
                ValidationIssue(
                    code="visual_reference",
                    severity=WARNING,
                    message=(
                        f"{name} tells the child to look at something "
                        f"('{match.group(0).strip()}') - the agent is voice-only"
                    ),
                    lesson_field=name,
                )
            )

    # --- vague math steps (WARNING) ----------------------------------------
    if is_math_lesson(lesson):
        match = _VAGUE_MATH_RE.search(prompt_agent)
        if match:
            issues.append(
                ValidationIssue(
                    code="vague_math_step",
                    severity=WARNING,
                    message=(
                        f"Math prompt_agent delegates calculation "
                        f"('{match.group(0).strip()}') instead of pre-solving it"
                    ),
                    lesson_field="prompt_agent",
                )
            )

    return ValidationReport(
        lesson_index=lesson_index,
        lesson_title=title,
        issues=tuple(issues),
    )


def validate_lessons(lessons: Iterable[dict]) -> list[ValidationReport]:
    """Validate every lesson in a plan. Returns one report per lesson."""
    return [
        validate_lesson_consistency(lesson, index)
        for index, lesson in enumerate(lessons, start=1)
        if isinstance(lesson, dict)
    ]


def summarize_reports(reports: list[ValidationReport]) -> dict:
    """Aggregate reports into one dict for logging and response metadata."""
    error_codes: list[str] = []
    warning_codes: list[str] = []
    for report in reports:
        error_codes.extend(i.code for i in report.errors)
        warning_codes.extend(i.code for i in report.warnings)

    invalid = [r.lesson_index for r in reports if not r.is_valid]
    return {
        "lessons_checked": len(reports),
        "lessons_with_errors": len(invalid),
        "invalid_lesson_indexes": invalid,
        "error_count": len(error_codes),
        "warning_count": len(warning_codes),
        "error_codes": sorted(set(error_codes)),
        "warning_codes": sorted(set(warning_codes)),
        "all_valid": not error_codes,
    }
