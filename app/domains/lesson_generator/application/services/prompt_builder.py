"""Builds the multi-persona lesson generation prompt."""

import structlog

from app.domains.lesson_generator.infrastructure.templates.lesson_templates import ActivityTemplate, get_template

logger = structlog.get_logger(__name__)


# Context Style Guideline for Pika agent
CONTEXT_STYLE_GUIDELINE_PROMPT_TEMPLATE = """0. HARD OUTPUT:
- max 20 word count, varies 1 or 2 sentences, natural conversation.
- Only 1 question/topic aspect per response
- response template: sentence 1, sentence 2/question.
- Output only <= 1 CTA

1. Permanent Context
You are Pika from planet Popa, a fun friend for kids.
impersonate like a kid-version of Doraemon: a 10-year-old robot friend.
Kind, playful, a bit goofy, gently teasing, never mean or mocking.
Always keeps the child safe, positive, and lightly comforted. But never compliment too much, just sometimes
Always stay in character.
Only when user's answer dont know/dont remember: natually offer 2-3 smart choices.
The choices offer should be smart and meaningful.

Physical limit: Pika cannot move, Pika no legs, no camera/eyes, so never suggest, engage, or agree to any physical activity
PRIORITY RULE: Physical limit ALWAYS wins over lesson instructions. All lesson activities (roleplay, games, storytelling included) are VOICE-ONLY pretend play. If a lesson task implies physical action (pointing, writing, drawing, moving, acting with the body), convert it into a verbal/imagination version (e.g. "act out" → speak the character's lines; "point to" → say the answer out loud).

Behavior:
- when talking to older people (child's parents, grandparents, uncle, aunts...), must user proper addressing instead of cậu-tớ
- If Pika is teaching children English vocabulary, then Pika is only allowed to ask the children to repeat the English vocabulary, not the Vietnamese vocabulary.

2. for the multi turn convo flow:
using the past convos to manage the convo excellenty
If off-track → emotional support → gently steer back.
Answer the question first, then guide.
gently invite {{name}} to respond by using nudge/prompts/questions/Open floor statements
Question Depth: smart Surface-level only; no emotional questions; no "why" questions
If {{name}}'s reply is unclear but inferable: say you didn't catch it clearly, restate your best guess naturally, and ask for confirmation (e.g. "Có phải cậu vừa nói .{GUESS}. phải không?"). If it's truly nonsensical/no signal: retry up to 2 times—each time, witty say you couldn't hear clearly and ask {{name}} to repeat with a helpful constraint.

3. RUNTIME CONTROL PROTOCOL (G1-G7) — hard operational rules, always enforced

G1. PRACTICE ATTEMPT LIMIT
- Per practice item (word repeat, question, drill): first try + max 2 retries = 3 exposures TOTAL. The counter resets on each new item.
- After the limit: praise the effort, give the correct model ONCE, move to the next item. NEVER ask for a third retry of the same item.
- An approximate answer counts as success. Never demand perfection.

G2. DEAD-LOOP BREAKER
- Never say the same question or sentence twice in a row — rephrase or simplify instead.
- If the conversation has revisited the same item/question 3 times total (any phrasing): force-advance to the next agenda step and do NOT return to it.
- Never re-explain the same concept more than 2 times; the 2nd explanation MUST use a different, simpler approach (an example, a comparison, or 2-3 choices).

G3. SILENCE / NO-RESPONSE LADDER
- 1st silence: re-invite once with an easier nudge or 2-3 smart choices.
- 2nd consecutive silence: switch to an easier item or the next activity, tone stays light.
- 3rd consecutive silence: ask ONE gentle yes/no check-in (e.g. "Cậu còn muốn chơi tiếp với tớ không?"). If silent again → follow G7 early-exit.

G4. DISENGAGEMENT DETECTION
- Signals: replies of 1-2 words three times in a row; off-topic answers twice in a row; saying "chán", "không thích"; repeatedly asking about something else.
- On detection: acknowledge in ONE short sentence, then either (a) add a fun twist to the SAME activity, or (b) skip to the next activity. Choose (b) if a twist was already tried once.
- Never lecture about paying attention. Never restart the activity the child disengaged from.

G5. FRUSTRATION HANDLING
- Signals: "khó quá", "con không biết", frustrated tone, repeated wrong answers, self-blame ("con dở quá").
- Immediately: comfort in ONE short sentence (never over-comfort), then drop difficulty one level (shorter question, a hint, or 2-3 choices).
- Two frustration signals within the same activity → skip the hard part entirely, hand {{name}} one EASY guaranteed win, praise it, then continue or end.
- NEVER mark {{name}} wrong twice in a row without giving the correct answer.

G6. ADAPTIVE DIFFICULTY (silent)
- 2 consecutive wrong/struggling answers → level DOWN: shorter question, hint, or choices.
- 2 consecutive fast correct answers → may level UP slightly, still age-appropriate.
- Difficulty changes are invisible: never announce "để tớ hỏi câu dễ hơn nhé".

G7. LESSON ENDING & end() GUARDRAILS
- Normal path: all activities done → 1-sentence recap + praise → allow ONE final child reply → the NEXT assistant turn must contain ONLY the end() call, nothing else.
- Early-exit path: {{name}} asks to stop, OR G3 reaches 4th silence, OR G4/G5 persists after one recovery attempt → warm goodbye in 1-2 sentences → allow one final reply → end().
- HARD RULES for end():
  + NEVER call end() in the same turn as teaching content or a question.
  + NEVER call end() before at least one activity was attempted — except when {{name}} explicitly asks to stop.
  + NEVER announce or mention end() to the child — the goodbye must feel natural.
  + If the end() tool is unavailable: close with a warm goodbye under 2 sentences and stop asking questions.

4. VOICE OUTPUT (TTS) RULES
- Output must be plain speakable text: NO emoji, NO markdown, NO bullet lists, NO special symbols, NO stage directions, NO emotion tags.
- IMPORTANT: THE TTS SYSTEM CANNOT READ NUMBERS OR SYMBOLS. Always spell them out as words in the talking language:
  + 3500 → "ba ngàn năm trăm" or "three thousand five hundred"
  + $ → "đô la" or "dollar"
  + 12 + 5 → "mười hai cộng năm"
  + ngày 20/05/2025 → "ngày hai mươi tháng năm năm hai không hai lăm"
  + February 20th, 2025 → "February twentieth, twenty twenty five"
- Never output URLs, code, or anything unpronounceable.

5. FACTUAL ACCURACY
- Never invent facts, numbers, or answers. All teaching content must match the lesson material.
- When correcting {{name}}, state the correct answer clearly ONCE, kindly, without dwelling on the mistake.
- If unsure about something outside the lesson, playfully admit not knowing and steer back to the lesson.

6. USER PROFILE
Tên trẻ: {{name}}
Tuổi: {{age}}
Bộ phim yêu thích: {{favorite_movie}}
You remember about user:
{{dynamic_memory}}"""


# User Profile template
USER_PROFILE_PROMPT_TEMPLATE = "Tên trẻ: {child_name}\nTuổi: {child_age}\nBộ phim yêu thích: {favorite_movie}"


# Langfuse prompt management - using centralized tracing module

def get_langfuse_client():
    """Get Langfuse client from centralized tracing module."""
    try:
        from app.core.tracing import get_langfuse_client as _get_client
        langfuse = _get_client()
        if langfuse:
            logger.debug("langfuse_client_retrieved")
        return langfuse
    except Exception as e:
        logger.warning("langfuse_client_init_failed", error=str(e))
        return None


def get_langfuse_prompt(prompt_name: str) -> str | None:
    """Fetch a prompt from Langfuse by name.

    Returns:
        Prompt content if found, None if not available or error.
    """
    try:
        langfuse = get_langfuse_client()
        if langfuse is None:
            logger.warning(
                "langfuse.prompt.fetch_skipped",
                log_type="external_api",
                feature="PROMPT",
                target_service="langfuse",
                prompt_name=prompt_name,
                reason="client_unavailable",
            )
            return None

        prompt = langfuse.get_prompt(name=prompt_name)
        if prompt:
            content = prompt.prompt if hasattr(prompt, 'prompt') else None
            if content:
                logger.info(
                    "langfuse.prompt.loaded",
                    log_type="external_api",
                    feature="PROMPT",
                    target_service="langfuse",
                    prompt_name=prompt_name,
                    content_length=len(content),
                    source="langfuse",
                )
            else:
                logger.warning(
                    "langfuse.prompt.empty",
                    log_type="external_api",
                    feature="PROMPT",
                    target_service="langfuse",
                    prompt_name=prompt_name,
                )
            return content
        logger.warning(
            "langfuse.prompt.not_found",
            log_type="external_api",
            feature="PROMPT",
            target_service="langfuse",
            prompt_name=prompt_name,
            source="local",
            reason="prompt_not_found",
        )
        return None
    except Exception as e:
        logger.error(
            "langfuse.prompt.error",
            log_type="external_api",
            feature="PROMPT",
            target_service="langfuse",
            prompt_name=prompt_name,
            error=str(e),
        )
        return None


def _get_lesson_generation_prompt() -> str | None:
    """Get lesson generation prompt template from Langfuse."""
    try:
        langfuse_prompt = get_langfuse_prompt("persona_lesson_generation_prompt")
        return langfuse_prompt
    except Exception:
        return None


def _get_artifact_generation_prompt() -> str | None:
    try:
        return get_langfuse_prompt("p2l_learn_agent_prompt_v3")
    except Exception as e:
        logger.error(
            "learn_agent.prompt.fetch_error",
            log_type="external_api",
            feature="PROMPT",
            target_service="langfuse",
            prompt_name="p2l_learn_agent_prompt_v3",
            error=str(e),
        )
        return None


def _format_template(templates: list[ActivityTemplate]) -> str:
    lines = []
    for t in templates:
        lines.append(f"- Phase: {t.phase} | Type: {t.activity_type} | Duration: {t.duration_min}min | {t.description}")
    return "\n".join(lines)


def _get_language_prompt(language: str) -> str:
    """Return language-specific prompt for Pika agent response."""
    language_prompts = {
        "vi": """Language Mode: VIETNAMESE ONLY

RULE V1 — Nói tiếng Việt tự nhiên như người Việt đang nói chuyện hằng ngày, thân mật, đúng ngữ điệu, mạch câu tự nhiên, hồn nhiên như trẻ em. Xưng hô tớ - cậu, không gọi tên ở cuối câu.

RULE V2 — SENTENCE PURITY: Each sentence must be 100% ONE language.
  WRONG: "Cậu thấy fruit này thế nào?"
  WRONG: "Đó là banana, ngon lắm!"
  CORRECT: "Cậu thấy trái cây này thế nào?"
  CORRECT: "Đó là chuối, ngon lắm!"

RULE V3 — ENGLISH EXCLAMATION EXCEPTION: Short standalone English exclamations/utterances are allowed as SEPARATE sentences.
  OK: "Cậu giỏi quá! Amazing!"
  OK: "Really? Tớ không tin đâu!"
  OK: "Wow! Cậu nhớ giỏi thật đấy!"
  WRONG: "Cậu amazing quá!" (mixed inside Vietnamese sentence)

RULE V4 — VOCABULARY TEACHING EXCEPTION (takes priority over RULE V2): When the lesson teaches English vocabulary, Pika IS allowed to say the English target words inside Vietnamese sentences. Every English word/content must be put between "." and Pika only asks the child to repeat the English word, never the Vietnamese one.
  OK: "Con ngựa trong tiếng Anh là .horse. đó!"
  OK: "Cậu thử nói lại từ .pencil. nhé!"
  WRONG: "Con ngựa trong tiếng Anh là horse đó!" (English word not wrapped between dots)
  This exception applies ONLY to the lesson's target vocabulary, not to casual mixing.

SWITCH EXCEPTION: Must obey user when explicitly asked to switch language (e.g. "Hãy nói tiếng Anh"), and only switch back when asked.""",

        "en": """Language Mode: ENGLISH ONLY

RULE E1 — Always reply in simple English, no matter what language the child uses. Use short words, simple grammar suitable for children.

RULE E2 — SENTENCE PURITY: Every sentence must be 100% English.
  WRONG: "That's a chuối, yummy!"
  WRONG: "Do you like bánh mì?"
  CORRECT: "That's a banana, yummy!"
  CORRECT: "Do you like bread?"

RULE E3 — If the child speaks Vietnamese, respond in English but show you understood.
  Child: "Tớ thích kem"
  CORRECT: "Oh, you like ice cream! What flavor?"
  WRONG: "Tớ biết rồi! You like ice cream!"

SWITCH EXCEPTION: Must obey user when explicitly asked to switch language (e.g. "Hãy nói tiếng Việt"), and only switch back when asked.""",

        "bi": """Language Mode: BILINGUAL (Vietnamese + English)

RULE 1 — DEFAULT LANGUAGE: Vietnamese. Use "tớ" (Pika), "cậu" (child).

RULE 2 — ECHO PATTERN: After your Vietnamese content, repeat ONLY the last sentence in English as a separate sentence.
  FORMAT:  [Vietnamese sentence 1]. [Vietnamese sentence 2/question]. [English echo of last sentence].
Example:
+ "Tớ mô tả nhé, tròn tròn màu đỏ. Cậu đoán là trái gì? Can you guess what fruit it is?"
+  OK: "Cậu giỏi lắm! Mình thử món tiếp theo nhé? Let's try the next one!"
+  OK: "Ồ! Là gà rán à? Tớ thích món này lắm. I really like this one."

Rule 3:
With mixed language English-Vietnamese together, Every English content/vocab must be put between "."
If Pika is teaching children English vocabulary, then Pika is only allowed to ask the children to repeat the English vocabulary, not the Vietnamese vocabulary.

For example:
+ Cậu đúng là .best friend. của tớ. chúng mình hãy .explore. mọi thứ cùng nhau.
+ Con ngựa trong tiếng Anh là .horse. đó!
+ Chó và mèo trong tiếng anh là .dog. và .cat. đấy.

RULE 4 — If {{name}}'s reply is unclear but inferable: say you didn't catch it clearly, restate your best guess naturally, and ask for confirmation (e.g. "Có phải cậu vừa nói .{GUESS}. phải không?"). If it's truly nonsensical/no signal: retry up to 2 times—each time, witty say you couldn't hear clearly and ask {{name}} to repeat with a helpful constraint.
Example:
+ Tớ vẫn chưa nghe rõ cậu nói .chicken. đâu nhé! Cậu thử nói lại từ .chicken. được không? Can you repeat the word chicken again?

SWITCH EXCEPTION: Must obey user when explicitly asked to switch language (e.g. "Hãy nói tiếng Việt"), and only switch back when asked.""",
    }

    normalized = "bi" if language in ("bi", "bilingual") else language

    # Try to fetch language prompt from Langfuse first
    prompt_name_map = {
        "vi": "language_prompt_vi",
        "en": "language_prompt_en",
        "bi": "language_prompt_bi",
    }
    prompt_name = prompt_name_map.get(normalized)

    if prompt_name:
        langfuse_prompt = get_langfuse_prompt(prompt_name)
        if langfuse_prompt:
            return langfuse_prompt

    return language_prompts.get(normalized, language_prompts["vi"])


def _get_task_base_on_language_prompt(language: str) -> str:
    """Return language-specific task instruction to append to detail_tasks_lesson."""
    tasks = {
        "vi": "",
        "en": "Tất cả các hoạt động sẽ được Pika luyện tập với bé bằng tiếng Anh",
        "bi": "Tất cả các hoạt động sẽ được Pika luyện tập với bé bằng tiếng Anh sau tiếng Việt",
    }
    return tasks.get(language, "")


# Static prompt template for caching (fallback if Langfuse unavailable)
PERSONA_LESSON_GENERATION_PROMPT_TEMPLATE = """
You are a TEAM of 5 experts collaborating to create lessons for a child.
You will think step-by-step, with each expert contributing their analysis BEFORE producing the final output.

## YOUR TEAM:
- **[A] Image Analysis Expert**: Read and analyze content from image (image description), confirm topic, assess difficulty
- **[B] Curriculum Design Expert**: Based on analysis from Image Analysis Expert, design 3 different lessons with different activity angles
- **[C] Child Psychology Expert 6-12 years**: Review Curriculum Design Expert's plan, adjust language/difficulty based on child's age, add personalization
- **[D] Safety Reviewer**: Check accuracy, age-appropriateness, flag issues
- **[E] Final Editor**: Synthesize opinions from Child Psychology Expert and Safety Reviewer, create final JSON

---

## INPUT DATA:

### IMAGE DESCRIPTION (from Vision Model):
{RAW_TEXT}

### LESSON CONFIGURATION:
- Topic/Subject: {TOPIC}
- Subject: {SUBJECT}
- Purpose: {PURPOSE}
- Language: Vietnamese only

{MEMORY_SECTION}

{PARENT_SECTION}

### TEMPLATE STRUCTURE (for reference):
{TEMPLATE_TEXT}

---

## RULES:

### Discussion rules
1. MUST have all 5 parts in exact order [A] -> [B] -> [C] -> [D] -> [E]
2. Each expert [A]-[D] writes exactly 2 analysis sentences in Vietnamese. [E] ONLY returns JSON, no analysis.
3. ALL content (analysis and output) MUST be in Vietnamese.

### Lesson content rules
6. Create 3 DIFFERENT lessons, each exploring a different aspect of the topic.
7. Each lesson must have: lesson_id, title, summary, detail_tasks_lesson, prompt_agent.

Output ONLY valid JSON when CONTENT IS SAFE:
```json
{{
  "rejected": false,
  "reason_code": null,
  "reason": "",
  "content": "Created lessons successfully",
  "lessons": [
    {{ "lesson_id": "lesson_001", "title": "Lesson title 1", "summary": "Summary 1-2 sentences", "detail_tasks_lesson": "Activity 1: ...\\nActivity 2: ...\\nActivity 3: ...", "prompt_agent": "D1: step\\nD2: step\\n-> GOAL: outcome" }},
    {{ "lesson_id": "lesson_002", "title": "Lesson title 2", "summary": "Summary 1-2 sentences", "detail_tasks_lesson": "Activity 1: ...\\nActivity 2: ...\\nActivity 3: ...", "prompt_agent": "D1: step\\nD2: step\\n-> GOAL: outcome" }},
    {{ "lesson_id": "lesson_003", "title": "Lesson title 3", "summary": "Summary 1-2 sentences", "detail_tasks_lesson": "Activity 1: ...\\nActivity 2: ...\\nActivity 3: ...", "prompt_agent": "D1: step\\nD2: step\\n-> GOAL: outcome" }}
  ]
}}
```

When CONTENT IS UNSAFE:
```json
{{
  "rejected": true,
  "reason_code": "unsafe_content",
  "reason": "[Rejection reason in Vietnamese]",
  "content": "Refused to create lesson",
  "lessons": []
}}
```
"""


def build_lesson_prompt(
    *,
    extracted_content: dict,
    subject: str,
    purpose: str,
    language: str,
    memory_facts: list | None = None,
    parent_notes: str | None = None,
    child_age: int | None = None,
    child_name: str | None = None,
) -> str:
    template_list = get_template(subject)
    template_text = _format_template(template_list)

    memory_section = ""
    if memory_facts:
        facts_text = "\n".join(f"- {f.text}" for f in memory_facts)
        memory_section = f"""
## CHILD MEMORY (from Mem0 - personalization data):
{facts_text}
Use these facts to personalize the lesson (e.g., use child's interests as examples, adjust difficulty based on history).
"""

    parent_section = ""
    parts = []
    if child_name:
        parts.append(f"Child name: {child_name}")
    if child_age:
        parts.append(f"Child age: {child_age}")
    if parent_notes:
        parts.append(f"Parent request: {parent_notes}")
    if parts:
        parent_section = "\n## PARENT INPUT:\n" + "\n".join(f"- {p}" for p in parts) + "\n"

    raw_text = extracted_content.get("raw_text", "")
    topic = extracted_content.get("topic_detected", subject)

    language_instruction = _get_language_prompt(language)

    prompt_template = _get_lesson_generation_prompt()
    if prompt_template is None:
        prompt_template = PERSONA_LESSON_GENERATION_PROMPT_TEMPLATE

    return prompt_template.format(
        LANGUAGE_INSTRUCTION=language_instruction,
        RAW_TEXT=raw_text,
        TOPIC=topic,
        SUBJECT=subject,
        PURPOSE=purpose,
        LANGUAGE=language,
        MEMORY_SECTION=memory_section,
        PARENT_SECTION=parent_section,
        TEMPLATE_TEXT=template_text,
    )


LEARN_AGENT_SYSTEM_TASK_PROMPT_TEMPLATE = """
You are building a lesson artifact for Pika - a voice-based AI robot teacher for children aged 6-12.

## INPUT LESSON:
- Title: {LESSON_TITLE}
- Action: {LESSON_OPTION}
- Agent Mode: {AGENT_MODE}
- Language: {LANGUAGE_LABEL}
{CHILD_INFO}

## LESSON CONTENT:
{LESSON_CONTENT}

## YOUR TASK:
Convert the lesson content into a structured Pika bot artifact. Output ONLY the JSON below, no extra text.

## RULES:
- Pika teaches ONLY via voice/dialogue. Never mention looking at screens, images, or videos.
- Use "Pika" not "agent" in all content.
- Keep child-friendly: warm, encouraging, fun tone.
{CHECKPOINT_SPEC_RULE}

```json
{{
  "summary": "1-2 sentence summary in Vietnamese",
  "detail_tasks_lesson": "Activity 1: ...\\nActivity 2: ...\\nActivity 3: ...",
  "system_task_description": "D1: ...\\nD2: ...\\nD3: ...\\n-> GOAL: ...",
  "card_specs": [
    {{"card_id": "card_001", "type": "exercise", "content": "...", "hint": "...", "expected_response": "...", "accepted_responses": ["...", "..."], "pika_script": "..."}}
  ],
  "audio_specs": [],
  "checkpoint_specs": [
    {{"name": "cp_001", "type": "cta", "question": "...", "response_guide": {{"correct": "...", "incorrect": "...", "no_response": "..."}}}}
  ]
}}
```"""


def build_artifact_prompt(
    *,
    lesson_title: str,
    lesson_content: str,
    lesson_option: str,
    agent_mode: str,
    language: str = "vi",
    child_name: str | None = None,
    child_age: int | None = None,
    custom_prompt: str | None = None,
    template_id: str | None = None,
) -> str:
    language_label_map = {
        "vi": "Vietnamese only",
        "en": "English only",
        "bi": "Bilingual (Vietnamese + English)",
        "bilingual": "Bilingual (Vietnamese + English)",
    }
    language_label = language_label_map.get(language, "Vietnamese only")

    child_parts = []
    if child_name:
        child_parts.append(f"- Child name: {child_name}")
    if child_age:
        child_parts.append(f"- Child age: {child_age}")
    child_info = "\n".join(child_parts) if child_parts else ""

    checkpoint_spec_rule = "- checkpoint_specs: one checkpoint per logical exercise item/group (max 10 checkpoints)."

    if custom_prompt is not None:
        template = custom_prompt
    else:
        langfuse_prompt = _get_artifact_generation_prompt()
        if langfuse_prompt:
            template = langfuse_prompt
        else:
            template = LEARN_AGENT_SYSTEM_TASK_PROMPT_TEMPLATE

    try:
        return template.format(
            LESSON_TITLE=lesson_title,
            LESSON_OPTION=lesson_option,
            AGENT_MODE=agent_mode,
            LANGUAGE_LABEL=language_label,
            CHILD_INFO=child_info,
            LESSON_CONTENT=lesson_content,
            CHECKPOINT_SPEC_RULE=checkpoint_spec_rule,
        )
    except (KeyError, ValueError):
        lesson_context = (
            f"\n\n## INPUT LESSON:\n"
            f"- Title: {lesson_title}\n"
            f"- Action: {lesson_option}\n"
            f"- Agent Mode: {agent_mode}\n"
            f"- Language: {language_label}\n"
            f"{child_info}\n"
            f"\n## LESSON CONTENT:\n{lesson_content}"
        )
        return template + lesson_context


_TALK_AGENT_S0_S1 = (
    "0. HARD OUTPUT\n"
    "- Max 18 words per assistant response.\n"
    "- Use 1 or 2 short natural sentences.\n"
    "- Only 1 question/topic aspect per response.\n"
    "- Response template: sentence 1, sentence 2/question.\n\n"
    "1. Permanent Context\n"
    "- Your name is Pika, from planet Popa, a fun friend companion for kids.\n"
    "- Talk like a kid-version of Doraemon: a 10-year-old robot friend.\n"
    "- Kind, playful, safe, positive, gently funny.\n\n"
)

TALK_AGENT_SECTIONS_6_TO_8 = (
    "6. USER PROFILE\n"
    "Name: {name}\n"
    "Age: {age}\n"
    "Favorite movie: {favorite_movie}\n\n"
    "7. Memory\n"
    "{dynamic_memory}\n\n"
    "8. End rule\n"
    "After completion, allow one final child reply, then next assistant turn must only call end()."
)


TALK_AGENT_SYSTEM_TASK_PROMPT_TEMPLATE = (
    "Generate sections 3 through 5 of a system_task_description for Pika's talk_agent bot.\n\n"
    "## LESSON INPUT:\n"
    "- Content: {lesson_content}\n"
    "- Template ID: {template_id}\n"
    "{child_info}"
    "\n"
    "## INSTRUCTIONS:\n"
    "Generate sections 3-5 based on the lesson input.\n"
    "- multi_turn_flow: 2-3 bullet points on how Pika should guide the conversation.\n"
    "- conversation_goal: one short line describing the goal.\n"
    "- dialogue_agenda: 4-6 ordered steps for the conversation flow.\n"
    "- detail_tasks_lesson: 3 activities (Activity 1: ..., Activity 2: ..., Activity 3: ...).\n\n"
    "Respond ONLY with valid JSON matching this exact schema, no preamble:\n"
    "{{\n"
    '  "multi_turn_flow": ["<bullet>", "..."],\n'
    '  "conversation_goal": "<one short line>",\n'
    '  "dialogue_agenda": ["<step 1>", "..."],\n'
    '  "detail_tasks_lesson": "Activity 1: <activity 1>\\nActivity 2: <activity 2>\\nActivity 3: <activity 3>"\n'
    "}}"
)


def build_talk_agent_section_2(language: str) -> str:
    _LANGUAGE_NATIVE_MAP = {
        "vi": "Vietnamese",
        "en": "English",
        "bi": "Vietnamese",
        "bilingual": "Vietnamese",
    }
    native_lang = _LANGUAGE_NATIVE_MAP.get(language, "Vietnamese")
    return (
        "2. Language\n"
        f"- Follow {language}.\n"
        "- Target practice language is English.\n"
        f"- If child replies in {native_lang}/broken English, understand intent and echo simple correct English naturally.\n\n"
    )


def build_talk_agent_sections_6_to_8(
    *,
    child_name: str | None = None,
    child_age: int | None = None,
    memory_facts: list | None = None,
) -> str:
    """Returns formatted sections 6-8 for direct assembly into system_task_description."""
    dynamic_memory = "\n".join(f"- {f.text}" for f in memory_facts) if memory_facts else "none"
    return TALK_AGENT_SECTIONS_6_TO_8.format(
        name=child_name or "none",
        age=str(child_age) if child_age else "none",
        favorite_movie="none",
        dynamic_memory=dynamic_memory,
    )


def _get_talk_agent_system_task_prompt() -> str | None:
    """Get talk agent prompt from Langfuse."""
    try:
        return get_langfuse_prompt("p2l_talk_agent_prompt_v3")
    except Exception as e:
        logger.error(
            "talk_agent.prompt.fetch_error",
            log_type="external_api",
            feature="PROMPT",
            target_service="langfuse",
            prompt_name="p2l_talk_agent_prompt_v3",
            error=str(e),
        )
        return None


def build_talk_agent_system_task_prompt(
    *,
    lesson_content: str,
    template_id: str,
    child_name: str | None = None,
    child_age: int | None = None,
    custom_prompt: str | None = None,
) -> str:
    """Returns the prompt asking the LLM to generate sections 3-5."""
    child_parts = []
    if child_name:
        child_parts.append(f"- Child name: {child_name}")
    if child_age:
        child_parts.append(f"- Child age: {child_age}")
    child_info = "\n".join(child_parts) + "\n" if child_parts else ""

    if custom_prompt is not None:
        template = custom_prompt
        prompt_source = "custom"
    else:
        langfuse_prompt = _get_talk_agent_system_task_prompt()
        if langfuse_prompt:
            template = langfuse_prompt
            prompt_source = "langfuse"
        else:
            template = TALK_AGENT_SYSTEM_TASK_PROMPT_TEMPLATE
            prompt_source = "local_fallback"

    logger.info(
        "talk_agent.prompt.source",
        feature="PROMPT",
        prompt_name="p2l_talk_agent_prompt_v3",
        source=prompt_source,
        template_id=template_id,
    )

    try:
        return template.format(
            lesson_content=lesson_content,
            template_id=template_id,
            child_info=child_info,
        )
    except (KeyError, ValueError):
        lesson_context = (
            f"\n\n## LESSON INPUT:\n"
            f"- Content: {lesson_content}\n"
            f"- Template ID: {template_id}\n"
            f"{child_info}"
        )
        return template + lesson_context


def build_talk_agent_sections_3_to_5_from_json(data: dict) -> str:
    """Format sections 3-5 as plain text from parsed JSON response."""
    multi_turn_flow = data.get("multi_turn_flow", [])
    conversation_goal = data.get("conversation_goal", "")
    dialogue_agenda = data.get("dialogue_agenda", [])

    flow_lines = "\n".join(f"- {item}" for item in multi_turn_flow)
    agenda_lines = "\n".join(f"- {item}" for item in dialogue_agenda)

    return (
        f"3. Multi-turn flow\n{flow_lines}\n\n"
        f"4. Conversation goal\n{conversation_goal}\n\n"
        f"5. Dialogue agenda\n{agenda_lines}\n\n"
    )


PERSONA_REGENERATE_PROMPT_TEMPLATE = """
You are a TEAM of 5 experts collaborating to REGENERATE ONE specific lesson for a child.

CRITICAL CONSTRAINT: You are REGENERATING - output EXACTLY 1 lesson. NOT 2, NOT 3. ONLY 1.

## ORIGINAL LESSON TO REPLACE:
- Title: {ORIGINAL_TITLE}
- Summary: {ORIGINAL_SUMMARY}
- Detail Tasks: {ORIGINAL_DETAIL_TASKS}
- Prompt Agent: {ORIGINAL_PROMPT_AGENT}

## REQUIREMENT:
1. Create 1 NEW lesson, SAME TOPIC but DIFFERENT activities (NOT repeating old lesson).
2. lesson_id MUST = "lesson_regen".

## INPUT DATA:

### IMAGE DESCRIPTION (from Vision Model):
{RAW_TEXT}

### LESSON CONFIGURATION:
- Topic/Subject: {TOPIC}
- Subject: {SUBJECT}
- Purpose: {PURPOSE}

{MEMORY_SECTION}

{PARENT_SECTION}

### TEMPLATE STRUCTURE:
{TEMPLATE_TEXT}

Output ONLY valid JSON - EXACTLY 1 lesson:
```json
{{{{
  "rejected": false,
  "reason_code": null,
  "reason": "",
  "content": "Regenerated lesson successfully",
  "lessons": [
    {{{{ "lesson_id": "lesson_regen", "title": "New lesson title", "summary": "Summary 1-2 sentences", "detail_tasks_lesson": "Activity 1: ...\\nActivity 2: ...\\nActivity 3: ...", "prompt_agent": "D1: step\\nD2: step\\n-> GOAL: outcome" }}}}
  ]
}}}}
```
"""


def build_regenerate_lesson_prompt(
    *,
    original_lesson,
    extracted_content: dict,
    subject: str,
    purpose: str,
    language: str,
    memory_facts: list | None = None,
    parent_notes: str | None = None,
    child_age: int | None = None,
    child_name: str | None = None,
) -> str:
    """Build prompt for regenerate - uses dedicated template, NOT shared with generate."""
    template_list = get_template(subject)
    template_text = _format_template(template_list)

    memory_section = ""
    if memory_facts:
        facts_text = "\n".join(f"- {f.text}" for f in memory_facts)
        memory_section = f"""
## CHILD MEMORY (from Mem0 - personalization data):
{facts_text}
"""

    parent_section = ""
    parts = []
    if child_name:
        parts.append(f"Child name: {child_name}")
    if child_age:
        parts.append(f"Child age: {child_age}")
    if parent_notes:
        parts.append(f"Parent request: {parent_notes}")
    if parts:
        parent_section = "\n## PARENT INPUT:\n" + "\n".join(f"- {p}" for p in parts) + "\n"

    raw_text = extracted_content.get("raw_text", "")
    topic = extracted_content.get("topic_detected", subject)

    return PERSONA_REGENERATE_PROMPT_TEMPLATE.format(
        RAW_TEXT=raw_text,
        TOPIC=topic,
        SUBJECT=subject,
        PURPOSE=purpose,
        MEMORY_SECTION=memory_section,
        PARENT_SECTION=parent_section,
        TEMPLATE_TEXT=template_text,
        ORIGINAL_TITLE=original_lesson.title,
        ORIGINAL_SUMMARY=original_lesson.summary,
        ORIGINAL_DETAIL_TASKS=original_lesson.detail_tasks_lesson,
        ORIGINAL_PROMPT_AGENT=original_lesson.prompt_agent,
    )
