#!/usr/bin/env python3
"""
Benchmark script for AI Lesson Builder APIs
Tests: v1/lessons/generate, v3/lessons/generate, v3/lessons/generate_artifact
Cases: 1, 2, 3, 5 images
Metrics: Latency, Token Usage, Cost, Quality
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# ============================================
# Configuration
# ============================================
BASE_URL = "http://localhost:8765"
PROFILE_ID = "benchmark_user_001"

# Test image URLs (safe educational content only)
IMAGE_URLS = [
    "https://alokiddy.com.vn/Uploads/images/huong/tu-vung-tieng-anh-lop-3-quan-trong-1.jpg",
    "https://thanhtay.edu.vn/wp-content/uploads/2023/06/bai-tap-tieng-anh-lop-1-28.jpg",
    "https://ad.pantado.edu.vn/wp-content/uploads/2020/01/mau-cau-chao-hoi-tieng-anh-cho-be_1691806474.png",
    "https://bmyc.vn/wp-content/uploads/2025/08/Tu-vung-tieng-Anh-ve-cac-hoat-dong-hang-ngay.3.jpg",
    "https://tailieugiaovien.com.vn/storage/uploads/images/documents/contents/aa0ff33ea239c5e85dfa13342a266def-preview/1.png",
]

# Pricing per 1M tokens (USD)
PRICING = {
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "cached": 0.50, "output": 8.00},
    "gpt-5.6-terra": {"input": 2.00, "cached": 0.20, "cache_write": 2.50, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached": 0.02, "cache_write": 0.25, "output": 1.20},
}

# Test cases
TEST_CASES = [1, 2, 3, 5]


@dataclass
class BenchmarkResult:
    api: str
    image_count: int
    latency_ms: int
    status: str
    lessons_count: int = 0
    token_usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    model_used: str = ""
    error: str = ""
    raw_response: dict = field(default_factory=dict)


def calculate_cost(model: str, usage: dict) -> float:
    """Calculate cost based on token usage and model pricing."""
    if not usage or model not in PRICING:
        return 0.0

    pricing = PRICING[model]
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    cache_write_tokens = usage.get("cache_write_tokens", 0)

    # Calculate fresh tokens (not cached, not written)
    fresh_tokens = prompt_tokens - cached_tokens - cache_write_tokens

    # Cost calculation
    input_cost = (fresh_tokens / 1_000_000) * pricing["input"]
    cached_cost = (cached_tokens / 1_000_000) * pricing.get("cached", pricing["input"])
    cache_write_cost = (cache_write_tokens / 1_000_000) * pricing.get("cache_write", pricing["input"])
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + cached_cost + cache_write_cost + output_cost


async def test_v1_generate(client: httpx.AsyncClient, image_urls: list[str]) -> BenchmarkResult:
    """Test v1/lessons/generate API."""
    payload = {
        "profile_id": PROFILE_ID,
        "image_urls": image_urls,
        "optional_parent_config": {
            "subject": "english",
            "purpose": "review",
            "language": "vi",
            "child_name": "Bé Minh",
            "child_age": 6,
        },
    }

    start = time.monotonic()
    try:
        response = await client.post("/v1/lessons/generate", json=payload, timeout=120.0)
        latency_ms = int((time.monotonic() - start) * 1000)

        data = response.json()
        result_data = data.get("data", {})
        metadata = result_data.get("metadata", {})

        lessons = result_data.get("lessons", [])
        # v1 returns "usage" not "token_usage"
        token_usage = metadata.get("usage", {}) or metadata.get("token_usage", {})
        model = metadata.get("model_used", "gpt-5.6-terra")
        # Use pre-calculated cost if available
        api_cost = metadata.get("cost_usd", 0.0)

        return BenchmarkResult(
            api="v1/lessons/generate",
            image_count=len(image_urls),
            latency_ms=latency_ms,
            status=data.get("status", "unknown"),
            lessons_count=len(lessons),
            token_usage=token_usage,
            cost_usd=api_cost if api_cost else calculate_cost(model, token_usage),
            model_used=model,
            raw_response=data,
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        return BenchmarkResult(
            api="v1/lessons/generate",
            image_count=len(image_urls),
            latency_ms=latency_ms,
            status="error",
            error=str(e),
        )


async def test_v3_generate(client: httpx.AsyncClient, image_urls: list[str]) -> BenchmarkResult:
    """Test v3/lessons/generate API."""
    payload = {
        "profile_id": PROFILE_ID,
        "image_urls": image_urls,
        "optional_parent_config": {
            "subject": "english",
            "purpose": "review",
            "language": "vi",
            "child_name": "Bé Minh",
            "child_age": 6,
        },
    }

    start = time.monotonic()
    try:
        response = await client.post("/v3/lessons/generate", json=payload, timeout=120.0)
        latency_ms = int((time.monotonic() - start) * 1000)

        data = response.json()
        result_data = data.get("data", {})
        metadata = result_data.get("metadata", {})

        lessons = result_data.get("suggested_lessons", [])
        token_usage = metadata.get("token_usage", {})
        model = "gpt-5.6-terra"  # v3 generate uses terra

        return BenchmarkResult(
            api="v3/lessons/generate",
            image_count=len(image_urls),
            latency_ms=latency_ms,
            status=data.get("status", "unknown"),
            lessons_count=len(lessons),
            token_usage=token_usage,
            cost_usd=calculate_cost(model, token_usage),
            model_used=model,
            raw_response=data,
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        return BenchmarkResult(
            api="v3/lessons/generate",
            image_count=len(image_urls),
            latency_ms=latency_ms,
            status="error",
            error=str(e),
        )


async def test_v3_artifact(
    client: httpx.AsyncClient, v3_generate_response: dict
) -> BenchmarkResult:
    """Test v3/lessons/generate_artifact API using output from v3/generate."""
    result_data = v3_generate_response.get("data", {})
    suggested_lessons = result_data.get("suggested_lessons", [])

    if not suggested_lessons:
        return BenchmarkResult(
            api="v3/lessons/generate_artifact",
            image_count=0,
            latency_ms=0,
            status="skipped",
            error="No suggested lessons from v3/generate",
        )

    # Build artifact request from suggested lessons
    lessons = []
    for lesson in suggested_lessons[:3]:  # Max 3 lessons for artifact
        options = lesson.get("options", [])
        first_option = options[0] if options else {}
        template_id = first_option.get("template_id", "ptl_learn_exercise_solver_v1")

        lesson_data = {
            "title": lesson.get("title", ""),
            "content": lesson.get("content", ""),
            "agent_mode": lesson.get("agent_mode", "learn_agent"),
            "template_id": template_id,
            "option": first_option.get("option", ""),
        }

        # Only add exercise_subtypes if it has a value
        exercise_subtype = first_option.get("exercise_subtype")
        if exercise_subtype:
            lesson_data["exercise_subtypes"] = [exercise_subtype]

        lessons.append(lesson_data)

    payload = {
        "profile_id": PROFILE_ID,
        "lessons": lessons,
    }

    start = time.monotonic()
    try:
        response = await client.post("/v3/lessons/generate_artifact", json=payload, timeout=180.0)
        latency_ms = int((time.monotonic() - start) * 1000)

        data = response.json()
        result_data = data.get("data", {})
        metadata = result_data.get("metadata", {})

        artifact_lessons = result_data.get("lessons", [])
        token_usage = metadata.get("token_usage", {})
        model = "gpt-4.1"  # Artifact uses gpt-4.1

        return BenchmarkResult(
            api="v3/lessons/generate_artifact",
            image_count=len(lessons),
            latency_ms=latency_ms,
            status=data.get("status", "unknown"),
            lessons_count=len(artifact_lessons),
            token_usage=token_usage,
            cost_usd=calculate_cost(model, token_usage),
            model_used=model,
            raw_response=data,
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        return BenchmarkResult(
            api="v3/lessons/generate_artifact",
            image_count=len(lessons),
            latency_ms=latency_ms,
            status="error",
            error=str(e),
        )


def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    status_icon = "✅" if result.status == "success" else "❌"
    print(f"\n{status_icon} {result.api} ({result.image_count} images)")
    print(f"   Latency: {result.latency_ms:,}ms ({result.latency_ms/1000:.1f}s)")
    print(f"   Lessons: {result.lessons_count}")
    print(f"   Model: {result.model_used}")

    if result.token_usage:
        usage = result.token_usage
        print(f"   Tokens: {usage.get('prompt_tokens', 0):,} in / {usage.get('completion_tokens', 0):,} out")
        if usage.get("cached_tokens"):
            print(f"   Cached: {usage.get('cached_tokens', 0):,} tokens")
        if usage.get("cache_write_tokens"):
            print(f"   Cache Write: {usage.get('cache_write_tokens', 0):,} tokens")

    print(f"   Cost: ${result.cost_usd:.4f}")

    if result.error:
        print(f"   Error: {result.error}")


def generate_report(results: list[BenchmarkResult], output_file: str):
    """Generate a comprehensive benchmark report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(results),
            "successful": sum(1 for r in results if r.status == "success"),
            "failed": sum(1 for r in results if r.status != "success"),
        },
        "by_api": {},
        "by_image_count": {},
        "results": [],
    }

    # Organize by API
    for api in ["v1/lessons/generate", "v3/lessons/generate", "v3/lessons/generate_artifact"]:
        api_results = [r for r in results if r.api == api and r.status == "success"]
        if api_results:
            report["by_api"][api] = {
                "avg_latency_ms": sum(r.latency_ms for r in api_results) // len(api_results),
                "total_cost_usd": sum(r.cost_usd for r in api_results),
                "avg_lessons": sum(r.lessons_count for r in api_results) // len(api_results),
            }

    # Organize by image count
    for count in TEST_CASES:
        count_results = [r for r in results if r.image_count == count and r.status == "success"]
        if count_results:
            report["by_image_count"][f"{count}_images"] = {
                "v1_latency_ms": next((r.latency_ms for r in count_results if r.api == "v1/lessons/generate"), None),
                "v3_generate_latency_ms": next((r.latency_ms for r in count_results if r.api == "v3/lessons/generate"), None),
                "v3_artifact_latency_ms": next((r.latency_ms for r in count_results if r.api == "v3/lessons/generate_artifact"), None),
            }

    # All results
    for r in results:
        report["results"].append({
            "api": r.api,
            "image_count": r.image_count,
            "latency_ms": r.latency_ms,
            "status": r.status,
            "lessons_count": r.lessons_count,
            "token_usage": r.token_usage,
            "cost_usd": r.cost_usd,
            "model_used": r.model_used,
            "error": r.error,
        })

    # Save report
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def print_summary_table(results: list[BenchmarkResult]):
    """Print summary comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'API':<30} {'Images':<8} {'Latency':<12} {'Lessons':<10} {'Cost':<10}")
    print("-" * 70)

    for result in sorted(results, key=lambda x: (x.api, x.image_count)):
        if result.status == "success":
            print(
                f"{result.api:<30} {result.image_count:<8} "
                f"{result.latency_ms/1000:.1f}s{'':<7} "
                f"{result.lessons_count:<10} ${result.cost_usd:.4f}"
            )
        else:
            print(f"{result.api:<30} {result.image_count:<8} {'FAILED':<12}")

    # Totals by API
    print("\n" + "-" * 70)
    print("TOTALS BY API:")
    for api in ["v1/lessons/generate", "v3/lessons/generate", "v3/lessons/generate_artifact"]:
        api_results = [r for r in results if r.api == api and r.status == "success"]
        if api_results:
            total_cost = sum(r.cost_usd for r in api_results)
            avg_latency = sum(r.latency_ms for r in api_results) / len(api_results)
            print(f"  {api}: ${total_cost:.4f} total, {avg_latency/1000:.1f}s avg")


async def run_benchmark():
    """Run the complete benchmark suite."""
    print("=" * 80)
    print("AI LESSON BUILDER - API BENCHMARK")
    print("=" * 80)
    print(f"Base URL: {BASE_URL}")
    print(f"Test cases: {TEST_CASES} images")
    print(f"Total image URLs: {len(IMAGE_URLS)}")
    print()

    results: list[BenchmarkResult] = []

    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Check health
        try:
            health = await client.get("/health", timeout=5.0)
            print(f"✅ Service healthy: {health.json()}")
        except Exception as e:
            print(f"❌ Service not available: {e}")
            return

        for image_count in TEST_CASES:
            test_images = IMAGE_URLS[:image_count]
            print(f"\n{'='*60}")
            print(f"Testing with {image_count} image(s)...")
            print(f"{'='*60}")

            # Test v1/lessons/generate
            print("\n[1/3] Testing v1/lessons/generate...")
            v1_result = await test_v1_generate(client, test_images)
            results.append(v1_result)
            print_result(v1_result)

            # Test v3/lessons/generate
            print("\n[2/3] Testing v3/lessons/generate...")
            v3_result = await test_v3_generate(client, test_images)
            results.append(v3_result)
            print_result(v3_result)

            # Test v3/lessons/generate_artifact (using v3/generate output)
            print("\n[3/3] Testing v3/lessons/generate_artifact...")
            if v3_result.status == "success":
                artifact_result = await test_v3_artifact(client, v3_result.raw_response)
                results.append(artifact_result)
                print_result(artifact_result)
            else:
                print("   ⚠️ Skipped: v3/generate failed")

            # Brief pause between test sets
            await asyncio.sleep(1)

    # Print summary
    print_summary_table(results)

    # Generate report
    report_file = Path(__file__).parent / "benchmark_report.json"
    report = generate_report(results, str(report_file))
    print(f"\n📊 Report saved to: {report_file}")

    # Print cost summary
    total_cost = sum(r.cost_usd for r in results if r.status == "success")
    print(f"\n💰 Total benchmark cost: ${total_cost:.4f}")

    return results


if __name__ == "__main__":
    asyncio.run(run_benchmark())
