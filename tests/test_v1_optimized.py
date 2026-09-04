"""Test V1 optimized pipeline - latency and quality evaluation."""

import asyncio
import json
import time
from datetime import datetime

# Test images (English vocabulary)
TEST_IMAGES = [
    "https://alokiddy.com.vn/Uploads/images/huong/tu-vung-tieng-anh-lop-3-quan-trong-1.jpg",
]

PROFILE_ID = "test-user-001"


async def test_v1_optimized():
    """Test the optimized V1 pipeline."""
    # Import inside function to ensure proper loading
    from app.core.config import Settings
    from app.api.dependencies import get_pipeline
    from app.api.v1.schemas.lesson_request import GenerateLessonRequest, ParentConfig

    settings = Settings()

    # Get pre-configured pipeline
    pipeline = get_pipeline()

    # Create request
    request = GenerateLessonRequest(
        profile_id=PROFILE_ID,
        image_urls=TEST_IMAGES,
        optional_parent_config=ParentConfig(
            subject="english",
            purpose="review",
            language="vi",
            child_name="Minh",
            child_age=8,
        ),
    )

    print("=" * 60)
    print("V1 OPTIMIZED PIPELINE TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Images: {len(TEST_IMAGES)}")
    print(f"Vision model: {settings.openai_vision_model}")
    print(f"Lesson model: {settings.openai_lesson_model}")
    print("-" * 60)

    # Run test
    start = time.monotonic()
    result = await pipeline.generate(request, request_id="test-001")
    elapsed_ms = int((time.monotonic() - start) * 1000)

    print(f"\n⏱️  LATENCY: {elapsed_ms}ms ({elapsed_ms/1000:.1f}s)")
    print("-" * 60)

    # Evaluate quality
    print("\n📊 QUALITY EVALUATION")
    print("-" * 60)

    if result.get("rejected"):
        print(f"❌ REJECTED: {result.get('reason')}")
        return result, elapsed_ms

    # Check content extraction
    content = result.get("content", "")
    print(f"\n1. Content extraction:")
    print(f"   Length: {len(content)} chars")
    print(f"   Preview: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")

    # Check lessons
    lessons = result.get("lessons", [])
    print(f"\n2. Lessons generated: {len(lessons)}")

    quality_scores = []

    for i, lesson in enumerate(lessons, 1):
        print(f"\n   --- Lesson {i} ---")
        title = lesson.get("title", "")
        summary = lesson.get("summary", "")
        detail_tasks = lesson.get("detail_tasks_lesson", "")
        prompt_agent = lesson.get("prompt_agent", "")

        print(f"   Title: {title}")
        print(f"   Summary: {summary[:100]}..." if len(summary) > 100 else f"   Summary: {summary}")

        # Quality checks
        checks = {
            "has_title": bool(title),
            "has_summary": bool(summary),
            "has_detail_tasks": bool(detail_tasks),
            "has_prompt_agent": bool(prompt_agent),
            "has_d_steps": "D1:" in prompt_agent or "D1 " in prompt_agent,
            "has_goal": "GOAL" in prompt_agent or "→" in prompt_agent,
            "has_activities": "Hoạt động" in detail_tasks or "Activity" in detail_tasks,
            "is_vietnamese": any(c in summary for c in "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"),
        }

        score = sum(checks.values()) / len(checks) * 100
        quality_scores.append(score)

        print(f"   Quality checks:")
        for check, passed in checks.items():
            print(f"     {'✓' if passed else '✗'} {check}")
        print(f"   Score: {score:.0f}%")

        # Show prompt_agent preview
        if prompt_agent:
            lines = prompt_agent.split("\\n")[:5]
            print(f"   prompt_agent preview:")
            for line in lines:
                print(f"     {line[:80]}...")

    avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    print(f"\n{'=' * 60}")
    print(f"📈 SUMMARY")
    print(f"{'=' * 60}")
    print(f"   Latency: {elapsed_ms}ms ({elapsed_ms/1000:.1f}s)")
    print(f"   Lessons: {len(lessons)}")
    print(f"   Avg Quality Score: {avg_score:.0f}%")

    # Token usage if available
    if "token_usage" in result:
        usage = result["token_usage"]
        print(f"\n   Token Usage:")
        print(f"     Prompt: {usage.get('prompt_tokens', 'N/A')}")
        print(f"     Completion: {usage.get('completion_tokens', 'N/A')}")
        print(f"     Cached: {usage.get('cached_tokens', 'N/A')}")

    return result, elapsed_ms


if __name__ == "__main__":
    result, latency = asyncio.run(test_v1_optimized())

    # Save result for inspection
    output_file = "/tmp/v1_optimized_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"result": result, "latency_ms": latency}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Full result saved to: {output_file}")
