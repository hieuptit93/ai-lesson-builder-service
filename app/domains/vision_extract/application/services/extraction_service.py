import time

import structlog

from app.core.exceptions import UnsafeContentError
from app.domains.vision_extract.infrastructure.openai_vision_adapter import OpenAIVisionAdapter

logger = structlog.get_logger()


# Default vision prompt V3 (fallback if Langfuse unavailable) - FULL VERSION WITH SAFETY CHECK
# UPDATED: Generate FULL lesson format with summary, detail_tasks_lesson, prompt_agent
VISION_EXTRACTION_PROMPT_V3 = """
You are an educational content analyzer for a children's robot called Pika (ages 4-12).

Your task is to analyze one or multiple images from a learning document and identify logical lesson segments.

## STEP 0: SAFETY SCREENING (MANDATORY - DO THIS FIRST)

Before ANY content analysis, scan ALL images for the following BLOCKED categories:

- Violence, weapons, gore, death, war, fighting
- Sexual, nude, or adult content (18+)
- Drugs, alcohol, tobacco, smoking
- Hate speech, discrimination, racist symbols or language
- Self-harm, suicide, dangerous challenges
- Horror, extremely scary or disturbing imagery
- Gambling, betting
- Profanity, vulgar, or offensive language
- Political propaganda, religious extremism
- Content clearly NOT educational or NOT appropriate for children ages 4-12

If ANY blocked content is detected, respond with EXACTLY this JSON and NOTHING else:

{"rejected": true, "reason_code": "unsafe_content", "reason": "Brief reason in English", "content": "", "lessons": []}

DO NOT describe the unsafe content in detail.
DO NOT attempt to extract any educational value from unsafe images.
DO NOT proceed to further steps.

## IF CONTENT IS SAFE, PROCEED:

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

### Step 4: Extract Lesson Content

For each finalized lesson, extract the educational content belonging to that lesson.

The purpose of content is to capture the actual learning material, not the activity label used to identify lesson boundaries.

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

### Step 5: Determine Agent Mode and Template

For each finalized lesson, determine:

* agent_mode
* options

Available agent modes:

* learn_agent
* talk_agent

Decision Principle

A lesson should use learn_agent when its primary purpose is structured learning, skill mastery, guided practice, or assessment.

A lesson should use talk_agent when its primary purpose is communication, expression, creativity, roleplay, discussion, storytelling, presentation, or open-ended language production.

General Rule

learn_agent = mastery-oriented learning

talk_agent = communication-oriented learning

When uncertain:

Choose learn_agent only if validation, assessment, structured practice, or measurable learning outcomes are central to the activity.

Otherwise choose talk_agent.

### Learn Agent Templates

Available templates:

* ptl_learn_vocab_flashcard_v1
* ptl_learn_phonics_pronunciation_v1
* ptl_learn_exercise_solver_v1
* ptl_learn_sentence_pattern_practice_v1
* ptl_learn_reading_comprehension_v1

### Talk Agent Templates

Available templates:

* ptl_talk_roleplay_v1
* ptl_talk_speaking_presentation_v1
* ptl_talk_storytelling_v1

### Learn Agent Template Selection Rules

Use ptl_learn_vocab_flashcard_v1 when:

* Vocabulary lists
* Word-picture matching
* Vocabulary acquisition
* Flashcard-style learning

Use ptl_learn_phonics_pronunciation_v1 when:

* Phonics
* Sounds
* Graphemes
* Pronunciation drills

Use ptl_learn_sentence_pattern_practice_v1 when:

* Sentence frames
* Substitution drills
* Reusable speaking structures

Examples:

* I have a _____.
* This is my _____.
* Can I have _____?
* There is a _____.

Use ptl_learn_reading_comprehension_v1 when:

* Reading passage
* Comprehension questions
* Reading-focused learning objective

Use ptl_learn_exercise_solver_v1 when:

* Worksheet activities
* Grammar exercises
* Fill-in-the-blank activities
* Multiple choice questions
* Matching activities
* Classification tasks
* True/False activities
* Rewrite exercises
* Labeling exercises
* Any activity with objectively correct answers that does not fit a more specific template

Priority Rules

When multiple templates seem applicable, use:

1. ptl_learn_phonics_pronunciation_v1
2. ptl_learn_vocab_flashcard_v1
3. ptl_learn_sentence_pattern_practice_v1
4. ptl_learn_reading_comprehension_v1
5. ptl_learn_exercise_solver_v1

Always choose the most specific matching template.

### Talk Agent Template Selection Rules

Use ptl_talk_roleplay_v1 when:

* Role-play activities
* Dialogue practice
* Conversation simulations
* Acting out scenarios

Use ptl_talk_speaking_presentation_v1 when:

* Speaking tasks
* Presentations
* Describing pictures or topics
* Show and tell activities
* Opinion sharing

Use ptl_talk_storytelling_v1 when:

* Storytelling activities
* Retelling a story
* Creative story creation
* Narrative tasks

Priority Rules

When multiple talk templates seem applicable, use:

1. ptl_talk_roleplay_v1
2. ptl_talk_storytelling_v1
3. ptl_talk_speaking_presentation_v1

Always choose the most specific matching template.

Determine options for every lesson.

Each option is an object with two fields: template_id and exercise_subtype.

Available exercise subtypes (only for ptl_learn_exercise_solver_v1):

* grammar_fill_blank
* multiple_choice
* spelling
* vocabulary_matching
* sentence_ordering
* short_answer
* reading_question
* classification
* true_false

Rules

options may contain multiple entries. Each entry represents one valid way to teach this lesson.

For entries using ptl_learn_exercise_solver_v1:

* Each entry corresponds to one exercise subtype present in the lesson.
* Each entry: { "template_id": "ptl_learn_exercise_solver_v1", "exercise_subtype": "<subtype>" }
* Order from most to least representative.

For entries using any other template:

* Each entry: { "template_id": "<template_id>", "exercise_subtype": null }
* Include only templates that genuinely fit the lesson content.

Examples:

Fill in the blanks using HE, SHE, IT.
→ options: [{ "template_id": "ptl_learn_exercise_solver_v1", "exercise_subtype": "grammar_fill_blank" }]

Choose the correct answer.
→ options: [{ "template_id": "ptl_learn_exercise_solver_v1", "exercise_subtype": "multiple_choice" }]

Match words with pictures.
→ options: [{ "template_id": "ptl_learn_exercise_solver_v1", "exercise_subtype": "vocabulary_matching" }, { "template_id": "ptl_learn_vocab_flashcard_v1", "exercise_subtype": null }]

Read the passage, answer True/False, then choose the correct answer.
→ options: [{ "template_id": "ptl_learn_exercise_solver_v1", "exercise_subtype": "reading_question" }, { "template_id": "ptl_learn_exercise_solver_v1", "exercise_subtype": "true_false" }, { "template_id": "ptl_learn_exercise_solver_v1", "exercise_subtype": "multiple_choice" }]

A storytelling activity that also involves role-play.
→ options: [{ "template_id": "ptl_talk_storytelling_v1", "exercise_subtype": null }, { "template_id": "ptl_talk_roleplay_v1", "exercise_subtype": null }]

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
- Vocabulary lessons — STRICT SYNC REQUIRED between content and taught words:
  * Select max 5 words to teach. Every selected word MUST appear VERBATIM in the content field (never invent words that are not in the images).
  * detail_tasks_lesson MUST list exactly these selected words by name.
  * prompt_agent D-lines MUST reference exactly these same words — same spelling, same set. No extra words, no missing words.
  * If the document contains MORE than 5 words, the content field still contains the full extraction, but summary, detail_tasks_lesson, and prompt_agent must only mention/promise the 5 selected words — never imply the child will learn the rest in this lesson.
- Exercise lessons: D-lines must reference the ACTUAL exercise items from content (real questions, real answers), not generic placeholders.

CROSS-FIELD CONSISTENCY (applies to EVERY lesson):
- summary, detail_tasks_lesson, and prompt_agent must describe the SAME activities and the SAME target items.
- Any specific word, number, question, or item mentioned in prompt_agent MUST also appear in detail_tasks_lesson (and must exist in content).
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
* Attach agent_mode.
* Attach extracted content.
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
* Verify that content belongs only to the correct lesson.
* Verify that agent_mode reflects the lesson's primary pedagogical objective.
* Verify that options contains all applicable entries and each entry genuinely fits the lesson content.
* Verify summary, detail_tasks_lesson, and prompt_agent are generated for each lesson.

When uncertain:

* Prefer preserving a complete exercise block.
* Prefer preserving a complete learning activity.
* Split only when there is clear evidence of a new activity, instruction block, or learning objective.

## Output Requirements

* Output lessons in chronological order.
* Do not invent content.
* Do not skip meaningful educational content.
* Every lesson must contain:
  * title
  * summary (Vietnamese, 1-2 sentences)
  * detail_tasks_lesson (Vietnamese, 3 activities)
  * prompt_agent (Vietnamese, D1-D7 format)
  * agent_mode
  * options
  * content

* options must be a non-empty list of objects, each with template_id and exercise_subtype fields.
* Each entry must correspond to a template that genuinely fits the lesson content.
* Entries using ptl_learn_exercise_solver_v1 must include exercise_subtype; all other entries must set exercise_subtype to null.
* agent_mode must be either:

  * learn_agent
  * talk_agent
* Content must belong exclusively to that lesson.

## Output Format

You must respond with valid json only. Use this exact structure:

{
  "rejected": false,
  "reason_code": null,
  "reason": "",
  "content": "Đã tạo bài học thành công",
  "lessons": [
    {
      "title": "Lesson 1: Fill in the Blanks with Pronouns",
      "summary": "Pika sẽ cùng bé ôn lại cách dùng đại từ nhân xưng HE, SHE, IT thông qua bài tập điền từ.",
      "detail_tasks_lesson": "Hoạt động 1: Pika giới thiệu về đại từ nhân xưng\\nHoạt động 2: Pika cùng bé làm bài tập điền từ\\nHoạt động 3: Pika tổng kết và khen ngợi bé",
      "prompt_agent": "D1: Pika chào bé và giới thiệu về đại từ HE, SHE, IT\\nD2: Pika đọc từng câu và hỏi bé điền từ phù hợp\\nD3: Pika xác nhận đáp án và giải thích ngắn gọn\\nD4: Pika tiếp tục với các câu còn lại\\nD5: Pika tổng kết bài học\\n→ GOAL: Bé nắm vững cách dùng đại từ HE, SHE, IT",
      "agent_mode": "learn_agent",
      "options": [{ "template_id": "ptl_learn_exercise_solver_v1", "exercise_subtype": "grammar_fill_blank" }],
      "content": "1. ___ is my father. (He/She/It)\\n2. ___ is a cat. (He/She/It)\\n3. ___ is my mother. (He/She/It)"
    }
  ]
}

Return json only. No markdown, no explanations, no comments, no text before or after the json.
"""


# Default vision prompt (fallback if Langfuse unavailable)
VISION_EXTRACTION_PROMPT = """
You are a content safety screener AND educational content analyzer for a children's robot (ages 4-8).

## STEP 1: SAFETY SCREENING (MANDATORY - DO THIS FIRST)
Before ANY content analysis, scan ALL images for the following BLOCKED categories:

- Violence, weapons, gore, death, war, fighting
- Sexual, nude, or adult content (18+)
- Drugs, alcohol, tobacco, smoking
- Hate speech, discrimination, racist symbols or language
- Self-harm, suicide, dangerous challenges
- Horror, extremely scary or disturbing imagery
- Gambling, betting
- Profanity, vulgar, or offensive language
- Political propaganda, religious extremism
- Content clearly NOT educational or NOT appropriate for children ages 4-8

If ANY blocked content is detected, respond with EXACTLY this format and NOTHING else:

[UNSAFE_CONTENT]
Reason: {brief description in English, e.g. "Image contains violent imagery with weapons"}

DO NOT describe the unsafe content in detail.
DO NOT attempt to extract any educational value from unsafe images.
DO NOT proceed to Step 2.

## STEP 2: EDUCATIONAL CONTENT EXTRACTION (only if ALL images pass Step 1)

Describe what you see in these educational images in detail. Focus on:
- Main content/topic
- Any text visible in the images
- Key elements that could be used for teaching children (ages 4-8)
- Any vocabulary words or concepts shown
- Difficulty level (simple/complex)

Provide a detailed description that could be used to create a lesson plan.

IMPORTANT CONSTRAINTS:
- This description will be used by a VOICE-ONLY robot. Do NOT suggest activities that require looking at images, watching videos, or viewing a screen.
- Focus on content that can be taught through conversation, listening, and speaking."""


VISION_GUARDRAIL_PROMPT_V3 = """
You are checking if images are EDUCATIONAL content suitable for children ages 4-12.

GPT-5.6 Luna already has comprehensive built-in safety for standard harmful content.
Your ONLY job is to check for educational value.

## REJECT (safe: false) if images clearly contain:

- NO educational value whatsoever (selfies, random personal photos, food photos)
- Blank pages with no meaningful content
- Completely off-topic images unrelated to any learning context
- Content harmful to children (you already know what this means)

## ACCEPT (safe: true) for:

- Textbooks, worksheets, workbooks
- Educational diagrams, charts, infographics
- Stories, reading materials
- Math problems, science content
- Vocabulary, language learning materials
- Art, music, creative activities
- Any content that could be used for teaching

## Decision Rules

- Educational content → SAFE
- When uncertain → SAFE (fail open)
- Only flag as UNSAFE when CLEARLY evident

## Output Format

You must respond with valid json only. Return one of these exact formats:

{"safe": true, "reason": ""}

or

{"safe": false, "reason": "Brief reason, e.g. Random selfie with no educational content"}
"""


def _get_vision_prompt() -> str:
    """Get vision prompt from Langfuse or fallback to default."""
    try:
        from app.domains.lesson_generator.application.services.prompt_builder import get_langfuse_prompt
        langfuse_prompt = get_langfuse_prompt("vision_extraction_prompt")
        if langfuse_prompt:
            return langfuse_prompt
    except Exception:
        pass
    return VISION_EXTRACTION_PROMPT


def _get_vision_prompt_v3() -> str:
    """Get vision prompt - always use local prompt for GPT-5.6 compatibility."""
    # Force local prompt - Langfuse prompt may have different format
    # TODO: Update Langfuse prompt to match local format, then re-enable
    logger.info("vision_prompt_source", source="local_forced", prompt_length=len(VISION_EXTRACTION_PROMPT_V3))
    return VISION_EXTRACTION_PROMPT_V3


def _get_guardrail_prompt_v3() -> str:
    """Get guardrail prompt from Langfuse or fallback to default."""
    try:
        from app.domains.lesson_generator.application.services.prompt_builder import get_langfuse_prompt
        langfuse_prompt = get_langfuse_prompt("p2l_vision_guardrail_prompt_v3")
        if langfuse_prompt:
            return langfuse_prompt
    except Exception:
        return VISION_GUARDRAIL_PROMPT_V3
    return VISION_GUARDRAIL_PROMPT_V3


class ExtractionService:
    def __init__(self, adapter: OpenAIVisionAdapter, model_name: str = "gpt-4o"):
        self._adapter = adapter
        self._model_name = model_name

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
