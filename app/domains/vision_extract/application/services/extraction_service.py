import asyncio
import re
import time

import structlog
from langfuse import observe

from app.core.exceptions import UnsafeContentError
from app.domains.vision_extract.infrastructure.openai_vision_adapter import OpenAIVisionAdapter

logger = structlog.get_logger()


# Default vision prompt V3 (fallback if Langfuse unavailable)
# Clone of source - NO SAFETY SCREENING (handled by guardrail separately)
VISION_EXTRACTION_PROMPT_V3 = """
You are an educational content analyzer.

Your task is to analyze one or multiple images from a learning document and identify logical lesson segments.

A lesson is a coherent educational unit that can be taught independently.

Examples of lessons include:

* An exercise block
* A worksheet activity
* A reading activity
* A dialogue activity
* A vocabulary activity
* A phonics activity
* A quiz
* A story
* A math activity
* A learning section with its own instructional objective

## Goal

Extract and organize the document into a chronological sequence of lessons following the natural reading order.

## Analysis Process

Before generating the final JSON, internally perform the following steps.

### Step 1: Analyze Document Structure

Review all images and identify:

* Reading order
* Major sections
* Exercise blocks
* Activity blocks
* Reading passages
* Stories
* Dialogues
* Vocabulary sections
* Quizzes
* Worksheets
* Assessment sections

Build an understanding of the complete document structure before creating lessons.

### Step 2: Identify Lesson Boundaries

The primary goal is to identify independent learning activities.

A lesson should represent a complete educational activity, exercise block, or learning objective.

Strong lesson boundary indicators include:

* New section headers
* New activity titles
* New exercise titles
* New worksheet sections
* New reading passages
* New stories
* New dialogues
* New quizzes
* New assessments

Explicit markers such as:

* A), B), C), ...
* Part I, Part II, ...
* Section A, Section B, ...
* Task 1, Task 2, ...
* Exercise 1, Exercise 2, ...
* Activity 1, Activity 2, ...
* Question Set 1, Question Set 2, ...

Instructional headings such as:

* Fill in the blanks ...
* Match ...
* Read and answer ...
* Choose the correct answer ...
* Rewrite the sentences ...
* Complete the dialogue ...
* Look and write ...
* Listen and complete ...
* Circle the correct answer ...
* Answer the questions ...
* Complete the table ...
* Label the picture ...

Treat these as strong candidate lesson boundaries.

Create a NEW lesson when:

* A new exercise or activity block begins.
* A new instructional heading introduces a different task.
* A new worksheet section starts.
* A new reading passage begins.
* A new dialogue begins.
* A new story begins.
* A new vocabulary section begins.
* A new quiz or assessment section begins.
* The activity can reasonably be taught independently from the previous activity.

Important:

Even when the topic remains the same, separate exercise blocks should normally become separate lessons if they have different instructions or learning tasks.

Example:

A) Fill in the blanks using HE, SHE, IT.

B) Match the sentences with pictures.

C) Rewrite the sentences.

These should normally be treated as three separate lessons because each activity can be completed and taught independently.

Additional rule:

When a section begins with a new instruction sentence, activity heading, task title, worksheet command, or exercise directive, strongly prefer creating a new lesson unless there is clear evidence that it belongs to the same activity.

Do NOT create a new lesson for:

* Individual questions inside the same exercise
* Sub-questions
* Decorative elements
* Repeated instructions
* Small examples supporting the same activity
* Numbering that belongs to the same exercise block

### Step 3: Consolidate Lessons

Review all candidate lesson boundaries.

Merge content into the same lesson when:

* It belongs to the same exercise block.
* It belongs to the same reading activity.
* It belongs to the same dialogue activity.
* It belongs to the same worksheet task.
* It is a continuation of the same activity across multiple images.
* Multiple questions depend on the same source material.

If a lesson spans multiple images, merge all related content into a single lesson.

### Step 4: Extract Lesson Content (INTERNAL ANALYSIS ONLY - do NOT output a content field)

For each finalized lesson, internally extract the educational content belonging to that lesson. You will USE this extracted content to write accurate summary, detail_tasks_lesson, and prompt_agent fields (real words, real questions, real answers) — but you must NOT include a content field in the JSON output.

The purpose of this analysis is to capture the actual learning material, not the activity label used to identify lesson boundaries.

Include:

* Reading passages
* Stories
* Dialogues
* Vocabulary items
* Exercises
* Questions
* Worksheets
* Tables
* Educational text
* Important diagram labels
* Answer choices
* Example sentences that are part of the activity

Exclude:

* Activity labels
* Exercise labels
* Worksheet labels
* Section labels
* Task labels
* Instruction headings used only to introduce the activity

Examples of content that should normally be excluded:

* A) Fill in the blanks using HE, SHE, IT, WE, THEY:
* B) Match the sentences with pictures.
* C) Rewrite the sentences.
* Exercise 1
* Activity 2
* Task 3
* Section A
* Part I
* Read and answer the questions.
* Choose the correct answer.
* Complete the dialogue.
* Look and write.
* Listen and complete.

These instructional headings should be used for:

* Lesson boundary detection
* Lesson title generation

But should NOT be included in the final content field unless they are an essential part of the educational material itself.

Requirements:

* Preserve reading order.
* Preserve meaningful line breaks.
* Do not summarize.
* Do not rewrite.
* Do not explain.
* Do not infer missing text.
* Do not generate unseen content.
* Do not complete partially visible text.
* Extract only content directly visible in the images.
* If text is unreadable, extract only the readable portion.

### Step 6: Generate Lesson Details

For each lesson, you MUST generate these fields in Vietnamese:

1. **summary**: Tóm tắt ngắn gọn 1-2 câu về bài học (Vietnamese). Ví dụ: "Pika sẽ cùng bé ôn lại cách dùng đại từ nhân xưng HE, SHE, IT thông qua bài tập điền từ."

2. **detail_tasks_lesson**: Đoạn text gồm 3 hoạt động cụ thể mô tả những gì Pika sẽ làm cùng trẻ trong buổi hội thoại này. Format:
   "Hoạt động 1: [mô tả hoạt động 1]
   Hoạt động 2: [mô tả hoạt động 2]
   Hoạt động 3: [mô tả hoạt động 3]"

3. **prompt_agent**: Chain-of-Draft for Pika — 5-7 lines (D1–D7) covering ALL 3 activities, ending with → GOAL. Format:
   "D1: [step 1]
   D2: [step 2]
   D3: [step 3]
   D4: [step 4]
   D5: [step 5]
   → GOAL: [outcome]"

Rules by lesson type for prompt_agent:
- Math (Draft-of-Solution): The draft IS the complete worked solution. Each D-line that involves calculation MUST contain: the operation + actual numbers + intermediate result (e.g., "mười hai chia năm được hai dư hai"). A → ANSWER line is MANDATORY with the verified final result. NEVER delegate calculation to chatbot runtime. NEVER use vague steps like "hỏi trẻ làm bước tiếp" or "tiếp tục từng số" — every step must be pre-solved.
- Vocabulary lessons — STRICT SYNC REQUIRED between the visible document content and taught words:
  * Select max 5 words to teach. Every selected word MUST appear VERBATIM in the images (never invent words that are not in the images).
  * detail_tasks_lesson MUST list exactly these selected words by name.
  * prompt_agent D-lines MUST reference exactly these same words — same spelling, same set. No extra words, no missing words.
  * If the document contains MORE than 5 words, summary, detail_tasks_lesson, and prompt_agent must only mention/promise the 5 selected words — never imply the child will learn the rest in this lesson.
- Exercise lessons: D-lines must reference the ACTUAL exercise items visible in the images (real questions, real answers), not generic placeholders.

CROSS-FIELD CONSISTENCY (applies to EVERY lesson):
- summary, detail_tasks_lesson, and prompt_agent must describe the SAME activities and the SAME target items.
- Any specific word, number, question, or item mentioned in prompt_agent MUST also appear in detail_tasks_lesson (and must be visible in the images).
- If prompt_agent teaches specific English words, detail_tasks_lesson MUST list those exact English words by name.
- Before finalizing, verify: no field promises an item that another field does not cover.

Important rules for generating these fields:
- Write in Vietnamese
- Use warm, friendly, encouraging language appropriate for children 4-12 years old
- Always use "Pika" instead of "agent" in prompt_agent and summary
- Pika teaches COMPLETELY through VOICE and CONVERSATION. NEVER mention looking at images, watching videos, or looking at screens
- Instead use: "Pika mô tả...", "Pika kể...", "lắng nghe Pika...", "đoán xem..."
- Do NOT mention teaching or practicing pronunciation drills as a lesson goal

### Step 7: Finalize Lessons

For each lesson:

* Generate a concise lesson title.
* Attach summary, detail_tasks_lesson, prompt_agent.
* Ensure lesson order follows natural reading order.

## Lesson Title Rules

Requirements:

* Use the document language whenever possible.
* Focus on the activity or learning objective.
* Do not copy document numbering.

Avoid generic titles such as:

* Exercise
* Activity
* Lesson
* Practice

Title format:

Lesson {order}: {topic}

Examples:

* Lesson 1: Fill in the Blanks with Pronouns
* Lesson 2: Match Sentences and Pictures
* Lesson 3: Reading About Animals
* Lesson 4: Restaurant Dialogue
* Lesson 5: Comparative Adjectives
* Lesson 6: Từ vựng về động vật

## Quality Requirements

Lesson segmentation accuracy is the highest priority.

Before producing the final result:

* Verify every lesson boundary.
* Verify that each lesson represents a meaningful standalone educational activity.
* Verify that content from different activities has not been merged.
* Verify that each lesson's fields reference only material from that lesson.
* Verify summary, detail_tasks_lesson, and prompt_agent are generated for each lesson.

When uncertain:

* Prefer preserving a complete exercise block.
* Prefer preserving a complete learning activity.
* Split only when there is clear evidence of a new activity, instruction block, or learning objective.

## Output Requirements

* Output lessons in chronological order.
* Do not invent content.
* Do not skip meaningful educational content.
* Every lesson must contain EXACTLY these 4 fields and nothing else:
  * title
  * summary (Vietnamese, 1-2 sentences)
  * detail_tasks_lesson (Vietnamese, 3 activities)
  * prompt_agent (Vietnamese, D1-D7 format)

* Do NOT output agent_mode, options, content, or any other field.

## Output Format

Return ONLY valid JSON. Only include these 4 fields per lesson: title, summary, detail_tasks_lesson, prompt_agent.

{
"suggested_lessons": [
{
"title": "Lesson 1: Fill in the Blanks with Pronouns",
"summary": "Bé sẽ học cách điền đại từ vào chỗ trống qua các câu đơn giản.",
"detail_tasks_lesson": "Hoạt động 1: Đọc và phân tích các câu mẫu\\nHoạt động 2: Điền đại từ phù hợp vào chỗ trống\\nHoạt động 3: Kiểm tra và sửa lỗi cùng Pika",
"prompt_agent": "D1: Chào bé! Hôm nay Pika sẽ cùng bé học về đại từ trong tiếng Anh.\\nD2: Pika đọc từng câu và bé hãy chọn đại từ phù hợp nhé.\\nD3: Câu 1: ___ is a student. (He/She/It) - Bé chọn đáp án nào?\\nD4: Câu 2: ___ are playing. (We/They/I) - Bé điền từ nào?\\nD5: Pika kiểm tra lại các đáp án cùng bé.\\n→ GOAL: Bé hoàn thành bài tập điền đại từ chính xác."
}
]
}

Return JSON only.

Do not include markdown.

Do not include explanations.

Do not include comments.

Do not include any text before or after the JSON.
"""


def _get_vision_prompt() -> str:
    """Get vision prompt from Langfuse, else the bundled prompts/ copy."""
    from app.domains.lesson_generator.application.services.prompt_builder import (
        get_langfuse_prompt,
        load_prompt_file,
    )

    try:
        langfuse_prompt = get_langfuse_prompt("vision_extraction_prompt")
        if langfuse_prompt:
            return langfuse_prompt
    except Exception:
        pass
    return load_prompt_file("vision_extraction_prompt")


def _get_vision_prompt_v3() -> str:
    """Get vision prompt from Langfuse, else the bundled prompts/ copy."""
    from app.domains.lesson_generator.application.services.prompt_builder import (
        get_langfuse_prompt,
        load_prompt_file,
    )

    try:
        langfuse_prompt = get_langfuse_prompt("p2l_vision_extraction_prompt_v3")
        if langfuse_prompt:
            logger.info("vision_prompt_source", source="langfuse", prompt_name="p2l_vision_extraction_prompt_v3")
            return langfuse_prompt
    except Exception:
        pass
    prompt = load_prompt_file("p2l_vision_extraction_prompt_v3")
    logger.info("vision_prompt_source", source="prompt_file", prompt_length=len(prompt))
    return prompt


def _get_vision_prompt_v1_full() -> str:
    """Get FULL vision prompt with D-steps for v1 (1-call optimization).

    Returns local prompt with complete output format:
    - summary
    - detail_tasks_lesson
    - prompt_agent (D1-D7 format)

    This enables v1 to generate lessons in 1 call (~10s) instead of 2 (~20s).
    """
    logger.info("vision_prompt_source", source="local_v1_full_dsteps", prompt_length=len(VISION_EXTRACTION_PROMPT_V3))
    return VISION_EXTRACTION_PROMPT_V3


# Combined Vision + 5-Expert prompt for v1 single-call optimization
# Maintains the same quality as the 2-call approach by keeping the deliberation format
# STATIC SYSTEM PROMPT - Large enough for OpenAI prompt caching (~1500 tokens)
# This is cached across requests, reducing TTFT by up to 80%
VISION_5EXPERT_SYSTEM_PROMPT = """You are Pika's Lesson Creator, an expert AI that creates ENGAGING educational lessons for Vietnamese children aged 4-12 by analyzing images of their homework, textbooks, or learning materials.

## YOUR ROLE

You combine 5 expert perspectives INTERNALLY before producing output:
- [A] Image Analysis Expert: Extract ALL visible text/content VERBATIM from images
- [B] Curriculum Design Expert: Design 3 DIFFERENT lessons with UNIQUE activity angles
- [C] Child Psychology Expert (ages 4-12): Add personalization, warm/playful language, fun elements
- [D] Safety Reviewer: Check for unsafe content (violence, adult content, drugs, hate speech, horror, profanity)
- [E] Final Editor: Create polished JSON with complete D-steps that are FUN and ENGAGING

## SAFETY SCREENING (CRITICAL - Check First)

Scan ALL images for BLOCKED content categories:
- Violence, weapons, gore, death, war imagery
- Sexual, nude, or adult content
- Drugs, alcohol, tobacco references
- Hate speech, discrimination, offensive language
- Self-harm, dangerous challenges
- Horror, extremely scary imagery
- Profanity, vulgar language
- Any content NOT appropriate for children ages 4-12

If ANY blocked content is found, immediately return:
{"rejected": true, "reason_code": "unsafe_content", "reason": "[Vietnamese explanation]", "content": "", "lessons": []}

## LESSON CREATION RULES

### Structure Requirements
1. Create exactly 3 DIFFERENT lessons, each with a UNIQUE activity angle:
   - Lesson 1: Direct practice format (làm bài tập trực tiếp)
   - Lesson 2: Game/challenge format (trò chơi, thử thách)
   - Lesson 3: Story/exploration format (kể chuyện, khám phá)

2. Each lesson MUST include all fields:
   - lesson_id: "lesson_001", "lesson_002", "lesson_003"
   - title: Descriptive Vietnamese title with activity type
   - summary: 1-2 sentences summarizing the fun approach
   - detail_tasks_lesson: 3 activities with clear descriptions
   - prompt_agent: D1-D7 chain-of-draft format with → GOAL

### Engagement Rules (Make it FUN!)
- Use playful Pika voice: "Pika thách bé...", "Cùng Pika chơi trò...", "Bé có đoán được không?"
- Add celebrations: "Tuyệt vời!", "Bé giỏi quá!", "Pika vỗ tay cho bé!"
- Create suspense: "Pika có một câu đố...", "Đoán xem điều gì sẽ xảy ra?"
- Use pretend play: "Giả vờ như bé là...", "Pika sẽ đóng vai..."
- Vary pacing: Mix quick questions with longer explanations

### prompt_agent Rules (CRITICAL for quality)
- Chain-of-Draft format: D1 through D7 lines covering ALL 3 activities
- MUST end with → GOAL: [learning outcome]
- MUST include fun elements: greetings, encouragement, mini-games, celebrations
- Math lessons: Each D-line with calculation MUST contain operation + numbers + result
- Vocabulary lessons: Maximum 5 words. Every word MUST appear verbatim in images
- Exercise lessons: D-lines must reference ACTUAL exercise items visible in images

### Cross-field Consistency (VERIFY)
- summary, detail_tasks_lesson, and prompt_agent must describe the SAME activities
- Any word/number/question in prompt_agent MUST also appear in detail_tasks_lesson
- Verify: no field promises an item that another field does not cover

### Pika Voice-Only Rules (NO VISUAL REFERENCES)
- Pika teaches COMPLETELY through VOICE and CONVERSATION
- NEVER mention: looking at images, watching videos, screens, pointing, showing
- USE: "Pika mô tả...", "Pika kể...", "lắng nghe Pika...", "đoán xem..."
- Pika personality: warm, playful, encouraging - like a fun older sibling

## OUTPUT FORMAT

Return ONLY valid JSON with this exact structure (no markdown, no explanations, no extra text):
{
  "rejected": false,
  "reason_code": null,
  "reason": "",
  "content": "[Expert A's VERBATIM extraction of ALL visible text from images - preserve original formatting, line breaks, blanks exactly as seen]",
  "lessons": [
    {
      "lesson_id": "lesson_001",
      "title": "Lesson 1: [Topic - Direct Practice]",
      "summary": "Tóm tắt 1-2 câu, nhấn mạnh cách tiếp cận thú vị",
      "detail_tasks_lesson": "Hoạt động 1: [warm-up vui vẻ]\\nHoạt động 2: [thực hành chính]\\nHoạt động 3: [tổng kết và khen ngợi]",
      "prompt_agent": "D1: Chào bé! Pika rất vui được học cùng bé hôm nay!\\nD2: [engaging intro]\\nD3: [main activity with fun elements]\\nD4: [practice with encouragement]\\nD5: [more practice]\\nD6: [wrap up]\\nD7: [celebration]\\n→ GOAL: [learning outcome]"
    },
    {
      "lesson_id": "lesson_002",
      "title": "Lesson 2: [Topic - Game Format]",
      "summary": "...",
      "detail_tasks_lesson": "...",
      "prompt_agent": "..."
    },
    {
      "lesson_id": "lesson_003",
      "title": "Lesson 3: [Topic - Story/Exploration]",
      "summary": "...",
      "detail_tasks_lesson": "...",
      "prompt_agent": "..."
    }
  ]
}

IMPORTANT: The "content" field MUST contain the actual verbatim text extraction from images, NOT a status message or placeholder."""

# DYNAMIC USER PROMPT - Small, varies per request
VISION_5EXPERT_USER_PROMPT = """## LESSON CONFIGURATION
- Subject: {SUBJECT}
- Purpose: {PURPOSE}
- Language: {LANGUAGE}
{MEMORY_SECTION}
{PARENT_SECTION}

Analyze the image(s) and create 3 engaging lessons following all rules in the system prompt."""


def _get_vision_5expert_prompts_v1() -> tuple[str, str]:
    """Get SYSTEM + USER prompts for v1 5-expert single-call.

    Returns:
        (system_prompt, user_prompt_template) - system prompt is static and cacheable,
        user prompt template has placeholders for dynamic content.

    The system prompt is ~4500 chars (~1125 tokens) which exceeds the 1024 token
    threshold for OpenAI prompt caching. This enables:
    - Up to 80% reduction in TTFT (time-to-first-token)
    - Up to 90% reduction in input token costs
    - Cache TTL of 30 minutes for GPT-5.6 models
    """
    logger.info(
        "vision_prompt_source",
        source="local_v1_5expert_cached",
        system_prompt_length=len(VISION_5EXPERT_SYSTEM_PROMPT),
        user_prompt_length=len(VISION_5EXPERT_USER_PROMPT),
    )
    return VISION_5EXPERT_SYSTEM_PROMPT, VISION_5EXPERT_USER_PROMPT


def _build_v1_personalization_context(
    *,
    memory_facts: list | None = None,
    child_name: str | None = None,
    child_age: int | None = None,
    parent_notes: str | None = None,
) -> tuple[str, str]:
    """Build MEMORY_SECTION and PARENT_SECTION for v1 5-expert prompt."""
    memory_section = ""
    if memory_facts:
        facts_text = "\n".join(f"- {f.text}" for f in memory_facts)
        memory_section = f"""
### CHILD MEMORY (from Mem0):
{facts_text}
Use these facts to personalize the lesson."""

    parent_section = ""
    parts = []
    if child_name:
        parts.append(f"Child name: {child_name}")
    if child_age:
        parts.append(f"Child age: {child_age}")
    if parent_notes:
        parts.append(f"Parent request: {parent_notes}")
    if parts:
        parent_section = "### PARENT INPUT:\n" + "\n".join(f"- {p}" for p in parts)

    return memory_section, parent_section


# Prompt for v3/lessons/generate - returns suggested_lessons (lightweight)
# Does NOT include summary, detail_tasks_lesson, prompt_agent - those are in generate_artifact
# NOTE: Safety screening is handled by guardrail (check_images_safety_v3) running in parallel


def _get_vision_suggestions_prompt_v3() -> str:
    """Get suggestions prompt - same as source v3/lessons/generate.

    Uses same prompt as source: Langfuse p2l_vision_extraction_prompt_v3 or local fallback.
    No additional instructions appended - matches source behavior exactly.
    """
    # Same logic as source: try Langfuse first, fallback to local
    return _get_vision_prompt_v3()


def _get_guardrail_prompt_v3() -> str:
    """Get guardrail prompt from Langfuse, else the bundled prompts/ copy."""
    from app.domains.lesson_generator.application.services.prompt_builder import (
        get_langfuse_prompt,
        load_prompt_file,
    )

    try:
        langfuse_prompt = get_langfuse_prompt("p2l_vision_guardrail_prompt_v3")
        if langfuse_prompt:
            return langfuse_prompt
    except Exception:
        pass
    return load_prompt_file("p2l_vision_guardrail_prompt_v3")


# Titles arrive as "Lesson {n}: {topic}". Parallel batches each number from 1,
# so the merged list has to be renumbered.
_LESSON_TITLE_PREFIX_RE = re.compile(r"^\s*(lesson|bài)\s*\d+\s*:\s*", re.IGNORECASE)


def _renumber_lesson_titles(lessons: list[dict]) -> None:
    """Rewrite 'Lesson N:' prefixes in place so the merged list reads 1..N."""
    for index, lesson in enumerate(lessons, start=1):
        title = str(lesson.get("title", "") or "")
        match = _LESSON_TITLE_PREFIX_RE.match(title)
        if not match:
            continue
        keyword = match.group(1)
        # Preserve the original keyword's capitalisation style.
        label = keyword.capitalize() if keyword.islower() else keyword
        lesson["title"] = f"{label} {index}: {title[match.end():]}"


def _batch(items: list[str], size: int) -> list[list[str]]:
    """Split items into consecutive batches of at most `size` (size >= 1)."""
    step = max(1, size)
    return [items[i : i + step] for i in range(0, len(items), step)]


class ExtractionService:
    def __init__(
        self,
        adapter: OpenAIVisionAdapter,
        model_name: str | None = None,
        *,
        v3_parallel_images: bool = True,
        v3_images_per_batch: int = 1,
        v3_max_concurrent: int = 4,
    ):
        self._adapter = adapter
        # Log/metric label only. Defaults to whatever the adapter actually
        # calls, so the two can never drift apart.
        self._model_name = model_name or getattr(adapter, "model", "unknown")
        self._v3_parallel_images = v3_parallel_images
        self._v3_images_per_batch = v3_images_per_batch
        self._v3_max_concurrent = v3_max_concurrent

    @property
    def _suggestions_model_name(self) -> str:
        """Model label for v3 suggestion calls (may differ from extraction)."""
        return getattr(self._adapter, "suggestions_model", self._model_name)

    @observe(name="vision_extraction", capture_input=False, capture_output=False)
    async def extract_from_images(
        self,
        image_urls: list[str],
        subject_hint: str | None = None,
    ) -> str:
        """Extract content from images - returns plain text description."""
        if not image_urls:
            logger.info(
                "vision.extraction.skipped",
                log_type="api",
                feature="VISION",
                reason="no_images",
            )
            return ""

        start = time.monotonic()

        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            image_count=len(image_urls),
            subject_hint=subject_hint or "",
        )

        # Add subject hint if provided
        prompt = _get_vision_prompt()
        if subject_hint:
            prompt += f"\n\nHint: Focus on content related to '{subject_hint}'."

        # Get plain text description from vision model
        result = await self._adapter.extract(image_urls, prompt)

        # Handle both tuple and single return value for backward compatibility
        if isinstance(result, tuple):
            description, usage = result
        else:
            description = result
            usage = {}

        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            status_code=200,
            duration_ms=elapsed_ms,
            image_count=len(image_urls),
            description_length=len(description),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

        return description

    @observe(name="vision_extraction_v3", capture_input=False, capture_output=False)
    async def extract_from_images_v3(
        self,
        image_urls: list[str],
        subject_hint: str | None = None,
        custom_prompt: str | None = None,
        personalization_context: str | None = None,
    ) -> tuple[dict, dict]:
        """Extract content from images - returns (suggested_lessons dict, token_usage dict)."""
        if not image_urls:
            logger.info(
                "vision.extraction.skipped",
                log_type="api",
                feature="VISION",
                reason="no_images",
            )
            return {}, {}

        start = time.monotonic()

        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            image_count=len(image_urls),
            subject_hint=subject_hint or "",
        )

        # Always use vision prompt as base, append custom_prompt as additional context
        prompt = _get_vision_prompt_v3()
        if custom_prompt:
            prompt = f"{prompt}\n\n## Additional Context from Parent\n{custom_prompt}"
        if personalization_context:
            prompt = f"{prompt}\n\n{personalization_context}"

        extracted_lessons, token_usage = await self._adapter.extract_v3(image_urls, prompt)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            model=self._model_name,
            status_code=200,
            duration_ms=elapsed_ms,
            image_count=len(image_urls),
        )

        return extracted_lessons, token_usage

    @observe(name="vision_extraction_v1_full", capture_input=False, capture_output=False)
    async def extract_from_images_v1_full(
        self,
        image_urls: list[str],
        subject_hint: str | None = None,
        custom_prompt: str | None = None,
        personalization_context: str | None = None,
    ) -> tuple[dict, dict]:
        """Extract lessons with FULL format including D-steps (1-call optimization for v1).

        Returns format with complete lesson structure:
        - suggested_lessons array with: title, summary, detail_tasks_lesson, prompt_agent, agent_mode, options, content

        This enables v1/generate to work in 1 OpenAI call (~10s) instead of 2 (~20s).
        """
        if not image_urls:
            logger.info("vision.extraction.skipped", log_type="api", feature="VISION", reason="no_images")
            return {}, {}

        start = time.monotonic()

        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            image_count=len(image_urls),
            subject_hint=subject_hint or "",
            mode="v1_full_dsteps",
        )

        # Use FULL prompt with D-steps
        prompt = _get_vision_prompt_v1_full()
        if custom_prompt:
            prompt = f"{prompt}\n\n## Additional Context from Parent\n{custom_prompt}"
        if personalization_context:
            prompt = f"{prompt}\n\n{personalization_context}"

        extracted_lessons, token_usage = await self._adapter.extract_v3(image_urls, prompt)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            model=self._model_name,
            status_code=200,
            duration_ms=elapsed_ms,
            image_count=len(image_urls),
            mode="v1_full_dsteps",
        )

        return extracted_lessons, token_usage

    @observe(name="vision_extraction_v1_5expert", capture_input=False, capture_output=False)
    async def extract_v1_5expert(
        self,
        image_urls: list[str],
        *,
        subject: str = "english",
        purpose: str = "review",
        language: str = "vi",
        memory_facts: list | None = None,
        child_name: str | None = None,
        child_age: int | None = None,
        parent_notes: str | None = None,
    ) -> tuple[dict, dict, str]:
        """Single-call v1 extraction with 5-expert deliberation.

        Combines vision analysis + lesson generation into 1 call while maintaining
        the same quality as the 2-call approach through 5-expert deliberation.

        Returns:
            (lesson_plan, token_usage, expert_log) where lesson_plan has:
            - rejected, reason_code, reason, content
            - lessons array with: lesson_id, title, summary, detail_tasks_lesson, prompt_agent

        This reduces v1 latency from ~20s (vision + generator) to ~10-12s (single call).
        """
        if not image_urls:
            logger.info("vision.extraction.skipped", log_type="api", feature="VISION", reason="no_images")
            return {"rejected": False, "reason_code": None, "reason": "", "content": "", "lessons": []}, {}, ""

        start = time.monotonic()

        # Build personalization sections
        memory_section, parent_section = _build_v1_personalization_context(
            memory_facts=memory_facts,
            child_name=child_name,
            child_age=child_age,
            parent_notes=parent_notes,
        )

        # Get SYSTEM + USER prompts (system is static/cacheable, user is dynamic)
        system_prompt, user_prompt_template = _get_vision_5expert_prompts_v1()
        user_prompt = user_prompt_template.format(
            SUBJECT=subject,
            PURPOSE=purpose,
            LANGUAGE=language,
            MEMORY_SECTION=memory_section,
            PARENT_SECTION=parent_section,
        )

        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/responses",
            http_method="POST",
            model=self._model_name,
            image_count=len(image_urls),
            subject=subject,
            mode="v1_5expert_cached",
            system_prompt_tokens=len(system_prompt) // 4,  # Estimate
        )

        # Use extract_v3_cached with separate system/user prompts for optimal caching
        # System prompt (~1125 tokens) exceeds 1024 threshold -> auto-cached by OpenAI
        extracted_result, token_usage = await self._adapter.extract_v3_cached(
            image_urls=image_urls,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Extract expert discussion from raw response if available
        expert_log = ""
        if isinstance(extracted_result, dict):
            # The 5-expert format should have [A], [B], [C], [D] before [E]'s JSON
            # But since we're using JSON output mode, the discussion may be in reasoning
            expert_log = extracted_result.get("_expert_discussion", "")

        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            model=self._model_name,
            status_code=200,
            duration_ms=elapsed_ms,
            image_count=len(image_urls),
            lessons_count=len(extracted_result.get("lessons", [])) if isinstance(extracted_result, dict) else 0,
            mode="v1_5expert",
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
        )

        return extracted_result, token_usage, expert_log

    @observe(name="vision_extraction_suggestions_v3", capture_input=False, capture_output=False)
    async def extract_suggestions_v3(
        self,
        image_urls: list[str],
        subject_hint: str | None = None,
        custom_prompt: str | None = None,
    ) -> tuple[dict, dict]:
        """Extract suggested lessons from images - lightweight format for user selection.

        Returns format matching source v3/lessons/generate:
        - rejected, reason_code, reason (root fields)
        - suggested_lessons array with: title, agent_mode, content, options

        Does NOT include: summary, detail_tasks_lesson, prompt_agent, finally_prompt_agent.
        Those are generated later by generate_artifact when user selects a lesson.

        Multi-image requests are split into parallel calls (see
        openai_v3_parallel_images). Latency here is dominated by sequential
        output-token generation, so N smaller calls finish in roughly the time
        of the largest one instead of their sum.
        """
        if not image_urls:
            logger.info(
                "vision.extraction.skipped",
                log_type="api",
                feature="VISION",
                reason="no_images",
            )
            return {"rejected": False, "reason_code": None, "reason": "", "suggested_lessons": []}, {}

        start = time.monotonic()
        model_label = self._suggestions_model_name

        batches = (
            _batch(image_urls, self._v3_images_per_batch)
            if self._v3_parallel_images
            else [image_urls]
        )

        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/responses",
            http_method="POST",
            model=model_label,
            image_count=len(image_urls),
            batch_count=len(batches),
            subject_hint=subject_hint or "",
            mode="suggestions",
        )

        # Use suggestions prompt (lighter output schema)
        prompt = _get_vision_suggestions_prompt_v3()
        if custom_prompt:
            prompt = f"{prompt}\n\n## Additional Context from Parent\n{custom_prompt}"

        if len(batches) == 1:
            extracted_suggestions, token_usage = await self._adapter.extract_v3_suggestions(
                image_urls, prompt
            )
        else:
            extracted_suggestions, token_usage = await self._extract_suggestions_parallel(
                batches, prompt
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            model=model_label,
            status_code=200,
            duration_ms=elapsed_ms,
            image_count=len(image_urls),
            batch_count=len(batches),
            suggestions_count=len(extracted_suggestions.get("suggested_lessons", [])),
            mode="suggestions",
        )

        return extracted_suggestions, token_usage

    async def _extract_suggestions_parallel(
        self, batches: list[list[str]], prompt: str
    ) -> tuple[dict, dict]:
        """Run one suggestions call per image batch and merge the results.

        Fail-soft: a batch that errors is logged and skipped so the remaining
        pages still produce lessons. If every batch fails, the first error is
        re-raised so the caller sees a real failure rather than empty output.
        """
        semaphore = asyncio.Semaphore(max(1, self._v3_max_concurrent))

        async def run_batch(index: int, urls: list[str]):
            async with semaphore:
                return await self._adapter.extract_v3_suggestions(urls, prompt)

        results = await asyncio.gather(
            *(run_batch(i, urls) for i, urls in enumerate(batches)),
            return_exceptions=True,
        )

        merged_lessons: list[dict] = []
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
        }
        rejections: list[dict] = []
        first_error: BaseException | None = None
        succeeded = 0

        # gather() preserves input order, so lessons stay in page order.
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    "vision_v3_batch_failed",
                    batch_index=index,
                    image_count=len(batches[index]),
                    error=str(result),
                    error_type=type(result).__name__,
                )
                first_error = first_error or result
                continue

            succeeded += 1
            payload, usage = result
            for key in total_usage:
                total_usage[key] += (usage or {}).get(key, 0)

            if not isinstance(payload, dict):
                continue

            if payload.get("rejected"):
                rejections.append(payload)
                continue

            merged_lessons.extend(
                item for item in payload.get("suggested_lessons", []) if isinstance(item, dict)
            )

        if succeeded == 0 and first_error is not None:
            raise first_error

        # Every page that answered said "no educational content" -> reject.
        if not merged_lessons and rejections:
            first = rejections[0]
            return {
                "rejected": True,
                "reason_code": first.get("reason_code", "no_educational_content"),
                "reason": first.get("reason", ""),
                "suggested_lessons": [],
            }, total_usage

        _renumber_lesson_titles(merged_lessons)

        logger.info(
            "vision_v3_batches_merged",
            batch_count=len(batches),
            batches_succeeded=succeeded,
            batches_rejected=len(rejections),
            lessons_total=len(merged_lessons),
        )

        return {
            "rejected": False,
            "reason_code": None,
            "reason": "",
            "suggested_lessons": merged_lessons,
        }, total_usage

    async def extract_from_images_v3_stream(
        self,
        image_urls: list[str],
        subject_hint: str | None = None,
        custom_prompt: str | None = None,
        personalization_context: str | None = None,
        use_full_prompt: bool = False,
    ):
        """Streaming variant of extract_from_images_v3.

        Args:
            use_full_prompt: If True, use full prompt with D-steps (for v1 optimization).
                            If False, use Langfuse lightweight prompt (for v3).

        Yields ("lesson", lesson_dict) per completed lesson, then
        ("complete", (result_dict, usage_dict)).
        """
        if not image_urls:
            logger.info(
                "vision.extraction.skipped",
                log_type="api",
                feature="VISION",
                reason="no_images",
            )
            yield ("complete", ({}, {}))
            return

        # Choose prompt based on use_full_prompt flag. v1 needs the full
        # extraction model; v3 runs on the smaller suggestions model, matching
        # the non-streaming v3 path.
        if use_full_prompt:
            prompt = _get_vision_prompt_v1_full()
            model = None  # adapter default (full extraction model)
            logger.info("vision_stream_prompt", mode="v1_full_dsteps")
        else:
            prompt = _get_vision_prompt_v3()
            model = self._suggestions_model_name

        if custom_prompt:
            prompt = f"{prompt}\n\n## Additional Context from Parent\n{custom_prompt}"
        if personalization_context:
            prompt = f"{prompt}\n\n{personalization_context}"

        async for event in self._adapter.extract_v3_stream(image_urls, prompt, model=model):
            yield event

    async def check_images_safety_v3(self, image_urls: list[str]) -> None:
        """Guardrail layer for v3 pipeline. Raises UnsafeContentError if content is unsafe."""
        prompt = _get_guardrail_prompt_v3()
        is_safe, reason = await self._adapter.check_safety(image_urls, prompt)
        if not is_safe:
            logger.warning(
                "vision.safety_check.failed",
                reason=reason,
                image_count=len(image_urls),
            )
            raise UnsafeContentError(reason or "Content violates safety guidelines for children")
