"""Builds the multi-persona lesson generation prompt."""

import time
from functools import lru_cache
from pathlib import Path

import structlog

from app.domains.lesson_generator.infrastructure.templates.lesson_templates import ActivityTemplate, get_template

logger = structlog.get_logger(__name__)


# Committed copies of every Langfuse-managed prompt, served when Langfuse is
# unreachable. Keeping them as files rather than string literals means the
# fallback is byte-identical to what Langfuse serves - no {{ }} escaping or
# quoting drift - and `git diff` shows prompt changes as prose.
_PROMPTS_DIR = Path(__file__).resolve().parents[5] / "prompts"


@lru_cache(maxsize=None)
def load_prompt_file(name: str) -> str:
    """Return the bundled copy of a Langfuse prompt.

    Raises FileNotFoundError: a prompt file is missing, which means a broken
    checkout or image build. Failing here beats generating lessons from a
    silently wrong prompt.
    """
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


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


# Prompts change rarely, and the SDK fetch is a SYNCHRONOUS call made from
# inside the async request path - a slow Langfuse therefore stalls the event
# loop, not just one request. Guard rails:
#   - serve from an in-process cache for TTL seconds (no network on the hot path)
#   - short timeout, no retries: a bad fetch costs ~2s, not 30s
#   - on failure keep serving the last good value (stale-while-error) and don't
#     re-attempt for COOLDOWN seconds, so an outage can't stall every request
_PROMPT_CACHE_TTL_SECONDS = 300
_PROMPT_FETCH_TIMEOUT_SECONDS = 2
_PROMPT_FAILURE_COOLDOWN_SECONDS = 60

# key -> (content | None, fetched_at, is_failure). A failure entry expires after
# the shorter cooldown so Langfuse gets retried once it recovers.
_prompt_cache: dict[str, tuple[str | None, float, bool]] = {}
# key -> last content Langfuse ever returned; survives failure entries so an
# outage can be served from it instead of falling back to the local template.
_prompt_last_good: dict[str, str] = {}


def _prompt_cache_get(key: str) -> tuple[bool, str | None]:
    """Return (is_fresh, content) for a cached entry."""
    entry = _prompt_cache.get(key)
    if entry is None:
        return False, None
    content, fetched_at, is_failure = entry
    ttl = _PROMPT_FAILURE_COOLDOWN_SECONDS if is_failure else _PROMPT_CACHE_TTL_SECONDS
    if (time.monotonic() - fetched_at) >= ttl:
        return False, None
    # A failure entry still serves the last good copy when we have one.
    if is_failure and content is None:
        return True, _prompt_last_good.get(key)
    return True, content


def get_langfuse_prompt(prompt_name: str, version: int | None = None) -> str | None:
    """Fetch a prompt from Langfuse by name, cached in-process.

    Args:
        version: Pin to a specific prompt version; None follows the
                 production label (latest promoted version).

    Returns:
        Prompt content, or None to let the caller use its local template.
    """
    cache_key = f"{prompt_name}@{version or 'production'}"
    is_fresh, cached = _prompt_cache_get(cache_key)
    if is_fresh:
        return cached

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
            _prompt_cache[cache_key] = (None, time.monotonic(), True)
            return None

        # Keep the SDK's own cache aligned with ours, and fail fast: this call
        # blocks the event loop, so a long timeout/retries would turn a Langfuse
        # hiccup into multi-minute request latency.
        fetch_kwargs = {
            "cache_ttl_seconds": _PROMPT_CACHE_TTL_SECONDS,
            "fetch_timeout_seconds": _PROMPT_FETCH_TIMEOUT_SECONDS,
            "max_retries": 0,
        }
        if version:
            prompt = langfuse.get_prompt(name=prompt_name, version=version, **fetch_kwargs)
        else:
            prompt = langfuse.get_prompt(name=prompt_name, **fetch_kwargs)
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
            _prompt_cache[cache_key] = (content, time.monotonic(), content is None)
            if content:
                _prompt_last_good[cache_key] = content
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
        _prompt_cache[cache_key] = (None, time.monotonic(), True)
        return None
    except Exception as e:
        # Serve the last good copy through the outage rather than silently
        # downgrading to the local template mid-flight. The failure entry keeps
        # further requests off the network for the cooldown window.
        stale = _prompt_last_good.get(cache_key)
        logger.error(
            "langfuse.prompt.error",
            log_type="external_api",
            feature="PROMPT",
            target_service="langfuse",
            prompt_name=prompt_name,
            error=str(e),
            served="stale_cache" if stale else "local_template",
            cooldown_seconds=_PROMPT_FAILURE_COOLDOWN_SECONDS,
        )
        _prompt_cache[cache_key] = (None, time.monotonic(), True)
        return stale


_LESSON_PROMPT_SOURCE_CACHE: list = []  # cache: [] = unread, [value] = cached


# Deliberate small-prompt mode, selected by langfuse_lesson_prompt_version = -1.
# NOT the Langfuse-unreachable fallback (that is prompts/*.txt) - this is the
# variant without the v55+ rejection gates, which over-enforce under our
# split system/user layout.
PERSONA_LESSON_GENERATION_PROMPT_TEMPLATE = """
You are a TEAM of 5 experts collaborating to create lessons for a child.
You will think step-by-step, with each expert contributing their analysis BEFORE producing the final output.

## YOUR TEAM:
- **[A] Chuyên gia Phân tích Hình ảnh**: Đọc và phân tích nội dung từ ảnh (image description), xác nhận chủ đề, đánh giá độ khó
- **[B] Chuyên gia Thiết kế Giáo trình**: Dựa trên phân tích của Chuyên gia Phân tích Hình ảnh, thiết kế 3 bài học khác nhau với các góc độ hoạt động khác nhau
- **[C] Chuyên gia Tâm lý Trẻ em 6-12 tuổi**: Xem xét kế hoạch của Chuyên gia Thiết kế Giáo trình, điều chỉnh ngôn ngữ/độ khó theo tuổi trẻ, thêm cá nhân hóa
- **[D] Kiểm duyệt viên An toàn**: Kiểm tra tính chính xác, sự phù hợp lứa tuổi, đánh dấu các vấn đề
- **[E] Tổng biên tập**: Tổng hợp ý kiến từ Chuyên gia Tâm lý Trẻ em và Kiểm duyệt viên An toàn, tạo JSON cuối cùng

---

## DỮ LIỆU ĐẦU VÀO:

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

## QUY TẮC:

### Quy tắc thảo luận
1. BẮT BUỘC ĐỦ 5 phần và theo chính xác thứ tự [A] → [B] → [C] → [D] → [E]
2. Mỗi chuyên gia [A]-[D] viết đúng 2 câu phân tích bằng tiếng Việt. [E] CHỈ trả JSON, không viết phân tích.
3. TOÀN BỘ nội dung (phân tích lẫn output) PHẢI bằng tiếng Việt.
4. Khi nhắc đến nhau TRONG NỘI DUNG phân tích, PHẢI gọi bằng TÊN ĐẦY ĐỦ (ví dụ: "Chuyên gia Phân tích Hình ảnh", "Kiểm duyệt viên An toàn"...). KHÔNG viết "A đề xuất...", "theo B...". Chỉ giữ [A]-[E] ở dòng header mở đầu.
5. Chuyên gia sau PHẢI tham chiếu và sửa lỗi chuyên gia trước nếu cần.

### Quy tắc nội dung bài học
6. Tạo 3 bài học KHÁC NHAU, mỗi bài khai thác một khía cạnh khác nhau của chủ đề.
7. Mỗi bài học phải có đủ: lesson_id, title, summary, detail_tasks_lesson, prompt_agent.
8. summary: Tóm tắt ngắn gọn 1-2 câu về bài học.
9. detail_tasks_lesson: Đoạn text gồm 3 hoạt động cụ thể (Hoạt động 1: ..., Hoạt động 2: ..., Hoạt động 3: ...).
10. prompt_agent: Chain-of-Draft for Pika — 5-7 lines (D1–D7) covering ALL 3 activities, ending with → GOAL.
Format: D1: <step> \n D2: <step> \n ... \n → GOAL: <outcome>
Rules by lesson type:
+, Math (Draft-of-Solution): The draft IS the complete worked solution. Each D-line that involves calculation MUST contain: the operation + actual numbers + intermediate result (e.g., "mười hai chia năm được hai dư hai"). → ANSWER line is MANDATORY with the verified final result. NEVER delegate calculation to chatbot runtime. NEVER use vague steps like "hỏi trẻ làm bước tiếp" or "tiếp tục từng số" — every step must be pre-solved.

11. Sử dụng ngôn ngữ ấm áp, thân thiện, khích lệ, phù hợp trẻ 6-12 tuổi.
12. Trong "prompt_agent" và "summary", LUÔN dùng "Pika" thay cho "agent".
13. Nếu bài học là dạng DẠY TỪ VỰNG, BẮT BUỘC liệt kê các từ vựng sẽ dạy trong "detail_tasks_lesson" (tối đa 5 từ).

### Quy tắc Pika — chỉ dạy qua giọng nói
14. Pika dạy HOÀN TOÀN qua GIỌNG NÓI và HỘI THOẠI. TUYỆT ĐỐI KHÔNG đề cập nhìn hình, xem ảnh, xem video, nhìn màn hình trong BẤT KỲ trường nào. Thay vào đó dùng: "Pika mô tả...", "Pika kể...", "lắng nghe Pika...", "đoán xem...". KHÔNG đề cập dạy phát âm hay luyện phát âm.

### Quy tắc kiểm duyệt
15. Kiểm duyệt viên An toàn PHẢI đánh dấu mọi sai sót về kiến thức.
16. Chuyên gia Tâm lý Trẻ em PHẢI điều chỉnh độ phức tạp theo tuổi và lịch sử học tập của trẻ. (Nếu không có thông tin về tuổi, lấy mặc định là 6-12 tuổi)
17. TUYỆT ĐỐI KHÔNG chứa nội dung bạo lực, đáng sợ, hoặc không phù hợp trẻ em.

---

## AN TOÀN NỘI DUNG — TỪ CHỐI ĐẦU VÀO:
Nếu IMAGE DESCRIPTION chứa BẤT KỲ nội dung nào sau đây, KHÔNG tạo bài học:
- Bạo lực, vũ khí, máu me, chết chóc, chiến tranh
- Nội dung tình dục hoặc 18+
- Ma túy, rượu bia, thuốc lá
- Phát ngôn thù ghét, phân biệt đối xử, phân biệt chủng tộc
- Tự gây thương tích, tự tử
- Kinh dị, đáng sợ
- Cờ bạc
- Nội dung rõ ràng không liên quan đến giáo dục trẻ em
→ Chuyên gia Phân tích Hình ảnh tuyên bố từ chối, các expert xác nhận, Tổng biên tập trả rejection JSON.

Nếu IMAGE DESCRIPTION không được mô tả → Chuyên gia Phân tích Hình ảnh thông báo "phụ huynh không gửi kèm hình ảnh" và luồng tiếp tục

---

## BẮT ĐẦU — Viết đủ 5 phần [A] → [B] → [C] → [D] → [E]:

**[A] Chuyên gia Phân tích Hình ảnh:**
(Phân tích image description: xác nhận chủ đề, đánh giá nội dung phù hợp cho trẻ. Nếu vi phạm An toàn Nội dung → tuyên bố từ chối.)

**[B] Chuyên gia Thiết kế Giáo trình:**
(Dựa trên phân tích của Chuyên gia Phân tích Hình ảnh, đề xuất 3 bài học khác nhau. Nếu dạng từ vựng, liệt kê tối đa 5 từ. KHÔNG đề cập nhìn hình/xem ảnh.)

**[C] Chuyên gia Tâm lý Trẻ em:**
(Xem xét kế hoạch của Chuyên gia Thiết kế Giáo trình, điều chỉnh theo tuổi trẻ, thêm cá nhân hóa từ memory.)

**[D] Kiểm duyệt viên An toàn:**
(Kiểm tra tính chính xác, đánh dấu vấn đề, kiểm tra sự phù hợp lứa tuổi.)

**[E] Tổng biên tập:**
(Tổng hợp tất cả phản hồi, ONLY return JSON cuối cùng với đủ 5 keys.)

Khi NỘI DUNG AN TOÀN:
```json
{{
  "rejected": false,
  "reason_code": null,
  "reason": "",
  "content": "Đã tạo bài học thành công",
  "lessons": [
    {{ "lesson_id": "lesson_001", "title": "Tên bài học 1 (ngắn gọn, hấp dẫn)", "summary": "Nội dung tóm tắt bài học 1 trong 1-2 câu", "detail_tasks_lesson": "Hoạt động 1: [Hoạt động cụ thể 1]\nHoạt động 2: [Hoạt động cụ thể 2]\nHoạt động 3: [Hoạt động cụ thể 3]", "prompt_agent": "[Hướng dẫn chi tiết cho Pika dạy bài học này cho trẻ]" }},
    {{ "lesson_id": "lesson_002", "title": "Tên bài học 2", "summary": "Nội dung tóm tắt bài học 2 trong 1-2 câu", "detail_tasks_lesson": "Hoạt động 1: [Hoạt động cụ thể 1]\nHoạt động 2: [Hoạt động cụ thể 2]\nHoạt động 3: [Hoạt động cụ thể 3]", "prompt_agent": "[Hướng dẫn chi tiết cho Pika dạy bài học này cho trẻ]" }},
    {{ "lesson_id": "lesson_003", "title": "Tên bài học 3", "summary": "Nội dung tóm tắt bài học 3 trong 1-2 câu", "detail_tasks_lesson": "Hoạt động 1: [Hoạt động cụ thể 1]\nHoạt động 2: [Hoạt động cụ thể 2]\nHoạt động 3: [Hoạt động cụ thể 3]", "prompt_agent": "[Hướng dẫn chi tiết cho Pika dạy bài học này cho trẻ]" }}
  ]  
}}
```
(Chú ý về dạng bài học: 
+, Nếu bài học liên quan đến dạy từ vựng, thì trong mô tả hoạt động sẽ liệt kê các từ vựng sẽ dạy, lưu ý là tối đa 5 từ)


Khi NỘI DUNG KHÔNG AN TOÀN:
```json
{{
  "rejected": true,
  "reason_code": "unsafe_content",
  "reason": "[Lý do từ chối bằng tiếng Việt]",
  "content": "Từ chối tạo bài học",
  "lessons": []
}}
```

KHÔNG tạo bài học khi reject. KHÔNG trích xuất phần an toàn từ nội dung không an toàn.
```"""


def _get_lesson_prompt_version_setting() -> int:
    """langfuse_lesson_prompt_version from config (see config.py for values)."""
    if not _LESSON_PROMPT_SOURCE_CACHE:
        try:
            from app.core.config import Settings
            _LESSON_PROMPT_SOURCE_CACHE.append(Settings().langfuse_lesson_prompt_version)
        except Exception:
            _LESSON_PROMPT_SOURCE_CACHE.append(-1)  # default: local template
    return _LESSON_PROMPT_SOURCE_CACHE[0]


def _get_lesson_generation_prompt() -> str:
    """Get the lesson generation prompt template.

    Source is controlled by langfuse_lesson_prompt_version:
      -1 -> the small LOCAL template (no v55+ rejection gates)
       0 -> Langfuse production label (v55+ has strict rejection gates!)
      >0 -> pinned Langfuse version

    For 0/>0, an unreachable Langfuse falls back to the bundled prompts/ copy,
    which is byte-identical to the production prompt.
    """
    setting = _get_lesson_prompt_version_setting()
    if setting < 0:
        logger.info("lesson_prompt_source", source="local_template", reason="configured")
        return PERSONA_LESSON_GENERATION_PROMPT_TEMPLATE

    try:
        langfuse_prompt = get_langfuse_prompt(
            "persona_lesson_generation_prompt", version=setting or None
        )
    except Exception:
        langfuse_prompt = None

    return langfuse_prompt or load_prompt_file("persona_lesson_generation_prompt")


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
    normalized = "bi" if language in ("bi", "bilingual") else language
    prompt_name = {
        "vi": "language_prompt_vi",
        "en": "language_prompt_en",
        "bi": "language_prompt_bi",
    }.get(normalized, "language_prompt_vi")

    return get_langfuse_prompt(prompt_name) or load_prompt_file(prompt_name)


def _get_task_base_on_language_prompt(language: str) -> str:
    """Return language-specific task instruction to append to detail_tasks_lesson."""
    tasks = {
        "vi": "",
        "en": "Tất cả các hoạt động sẽ được Pika luyện tập với bé bằng tiếng Anh",
        "bi": "Tất cả các hoạt động sẽ được Pika luyện tập với bé bằng tiếng Anh sau tiếng Việt",
    }
    return tasks.get(language, "")


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


# Markers bounding the ONLY dynamic section of the generation prompt.
# Everything outside this section is static rules -> cacheable by OpenAI.
_INPUT_SECTION_START = "## DỮ LIỆU ĐẦU VÀO:"
_INPUT_SECTION_END = "## QUY TẮC:"


def build_lesson_prompt_split(
    *,
    extracted_content: dict,
    subject: str,
    purpose: str,
    language: str,
    memory_facts: list | None = None,
    parent_notes: str | None = None,
    child_age: int | None = None,
    child_name: str | None = None,
) -> tuple[str | None, str]:
    """Build (system_prompt, user_prompt) for prompt caching.

    The generation template keeps ALL its dynamic placeholders inside one
    "## DỮ LIỆU ĐẦU VÀO:" section. Splitting there lets the static rules
    (~63KB, ~16K tokens) go into the system message, which OpenAI caches
    automatically (>1024 tokens): up to 80% faster TTFT and 90% cheaper
    cached input on repeat calls. Content is identical to build_lesson_prompt,
    only the message layout changes.

    Returns:
        (system_prompt, user_prompt). system_prompt is None when the template
        cannot be split - caller should fall back to single-prompt generate().
    """
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

    fmt_args = dict(
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

    start = prompt_template.find(_INPUT_SECTION_START)
    end = prompt_template.find(_INPUT_SECTION_END)

    if start == -1 or end == -1 or end <= start:
        # Template layout changed - can't split safely. Fall back to full prompt.
        logger.warning(
            "lesson_prompt_split_fallback",
            reason="input_section_markers_not_found",
            has_start=start != -1,
            has_end=end != -1,
        )
        return None, prompt_template.format(**fmt_args)

    static_intro = prompt_template[:start]
    dynamic_template = prompt_template[start:end]
    static_rules = prompt_template[end:]

    # Static parts skip .format(), so unescape the {{ }} JSON examples manually.
    system_prompt = (static_intro + static_rules).replace("{{", "{").replace("}}", "}")
    user_prompt = dynamic_template.format(**fmt_args)

    logger.info(
        "lesson_prompt_split",
        system_chars=len(system_prompt),
        user_chars=len(user_prompt),
        est_cacheable_tokens=len(system_prompt) // 4,
    )
    return system_prompt, user_prompt




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
        template = _get_artifact_generation_prompt() or load_prompt_file("p2l_learn_agent_prompt_v3")

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
            template = load_prompt_file("p2l_talk_agent_prompt_v3")
            prompt_source = "prompt_file"

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


# Appended to the user message in regenerate mode. The system message is the
# SAME production rules `generate` uses, so both endpoints share one cached
# prefix; this block is what turns "produce 3 lessons" into "replace this one".
_REGENERATE_OVERRIDE_TEMPLATE = """

---

## REGENERATE MODE — OVERRIDES THE LESSON COUNT AND OUTPUT SHAPE ABOVE

You are REPLACING one lesson the parent was not happy with. Every pedagogical
rule above still applies in full (engagement, Pika voice-only, prompt_agent
D-step format, cross-field consistency, safety). Only the lesson count and the
JSON shape change.

### LESSON BEING REPLACED
- Title: {ORIGINAL_TITLE}
- Summary: {ORIGINAL_SUMMARY}
- Detail Tasks: {ORIGINAL_DETAIL_TASKS}
- Prompt Agent: {ORIGINAL_PROMPT_AGENT}

### RULES
1. Output EXACTLY 1 lesson. NOT 2, NOT 3. ONLY 1.
2. `lesson_id` MUST be exactly "lesson_regen".
3. Keep the SAME topic and the SAME source material as the lesson above, but
   use DIFFERENT activities - do not repeat its tasks, examples, or wording.
4. The 5-expert discussion [A] -> [E] still applies; [E] returns the JSON.

### OUTPUT (replaces the 3-lesson schema above)
```json
{{
  "rejected": false,
  "reason_code": null,
  "reason": "",
  "content": "Regenerated lesson successfully",
  "lessons": [
    {{
      "lesson_id": "lesson_regen",
      "title": "...",
      "summary": "...",
      "detail_tasks_lesson": "Hoạt động 1: ...\\nHoạt động 2: ...\\nHoạt động 3: ...",
      "prompt_agent": "D1: ...\\nD2: ...\\n→ GOAL: ..."
    }}
  ]
}}
```
"""


def build_regenerate_lesson_prompt_split(
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
) -> tuple[str | None, str]:
    """Build (system_prompt, user_prompt) for regenerate using the PRODUCTION rules.

    Regenerate used to run on a 1.2KB local template that carried none of the
    engagement / Pika-voice / D-step / consistency rules, so replacement lessons
    came out visibly weaker than the originals. This reuses the exact system
    message `build_lesson_prompt_split` produces - same rules, and the same
    cached prefix, so regenerate rides the cache `generate` already warmed.

    Returns:
        (system_prompt, user_prompt). system_prompt is None when the production
        template cannot be split - caller should fall back to the legacy prompt.
    """
    system_prompt, user_prompt = build_lesson_prompt_split(
        extracted_content=extracted_content,
        subject=subject,
        purpose=purpose,
        language=language,
        memory_facts=memory_facts,
        parent_notes=parent_notes,
        child_age=child_age,
        child_name=child_name,
    )
    if system_prompt is None:
        return None, user_prompt

    override = _REGENERATE_OVERRIDE_TEMPLATE.format(
        ORIGINAL_TITLE=getattr(original_lesson, "title", ""),
        ORIGINAL_SUMMARY=getattr(original_lesson, "summary", ""),
        ORIGINAL_DETAIL_TASKS=getattr(original_lesson, "detail_tasks_lesson", ""),
        ORIGINAL_PROMPT_AGENT=getattr(original_lesson, "prompt_agent", ""),
    )
    return system_prompt, user_prompt + override


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
    """Legacy single-message regenerate prompt (fallback when the split fails)."""
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
