#!/usr/bin/env python3
"""
Benchmark: Original vs Optimized Service
Compares real API performance between:
- Original: https://robot-parents-lesson-builder.hacknao.edu.vn/
- Optimized: http://localhost:8000/ (or configured endpoint)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

ORIGINAL_BASE_URL = "https://robot-parents-lesson-builder.hacknao.edu.vn"
OPTIMIZED_BASE_URL = "http://localhost:8000"

# Safe educational images for testing
IMAGE_URLS = [
    "https://alokiddy.com.vn/Uploads/images/huong/tu-vung-tieng-anh-lop-3-quan-trong-1.jpg",
    "https://thanhtay.edu.vn/wp-content/uploads/2023/06/bai-tap-tieng-anh-lop-1-28.jpg",
    "https://tailieugiaovien.com.vn/storage/uploads/images/documents/contents/aa0ff33ea239c5e85dfa13342a266def-preview/1.png",
    "https://ad.pantado.edu.vn/wp-content/uploads/2020/01/mau-cau-chao-hoi-tieng-anh-cho-be_1691806474.png",
    "https://bmyc.vn/wp-content/uploads/2025/08/Tu-vung-tieng-Anh-ve-cac-hoat-dong-hang-ngay.3.jpg",
]

# Test cases: number of images to test
IMAGE_TEST_CASES = [1, 2, 3, 5]

# Test cases: number of lessons for artifact API
LESSON_TEST_CASES = [1, 2, 3, 5, 7, 9]

PROFILE_ID = "benchmark-test-user"

# Pricing (per 1M tokens)
PRICING = {
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "cached": 0.50, "output": 8.00},
    "gpt-5.6-terra": {"input": 2.00, "cached": 0.20, "cache_write": 2.50, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached": 0.02, "cache_write": 0.25, "output": 1.20},
}


@dataclass
class BenchmarkResult:
    api: str
    version: str  # "original" or "optimized"
    image_count: int = 0
    lesson_count: int = 0
    latency_ms: int = 0
    status: str = "pending"
    lessons_count: int = 0
    token_usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    model_used: str = ""
    error: str = ""


def estimate_original_cost(api: str, image_count: int = 0, lesson_count: int = 0) -> float:
    """Estimate cost for original service (no actual token tracking)"""
    if api == "v1/lessons/generate":
        # Original uses GPT-4o for vision + GPT-4.1 for generation
        # Estimate: ~5000 input per image for vision, ~2000 output
        # Then ~4000 input for generation, ~2500 output
        vision_input = 5000 * image_count
        vision_output = 1500
        gen_input = 4000 + vision_output  # includes vision output
        gen_output = 2500

        vision_cost = (vision_input * PRICING["gpt-4o"]["input"] / 1_000_000 +
                      vision_output * PRICING["gpt-4o"]["output"] / 1_000_000)
        gen_cost = (gen_input * PRICING["gpt-4.1"]["input"] / 1_000_000 +
                   gen_output * PRICING["gpt-4.1"]["output"] / 1_000_000)
        return vision_cost + gen_cost

    elif api == "v3/lessons/generate":
        # Original uses GPT-4o for vision extraction
        input_tokens = 6000 * image_count
        output_tokens = 200 * image_count
        return (input_tokens * PRICING["gpt-4o"]["input"] / 1_000_000 +
                output_tokens * PRICING["gpt-4o"]["output"] / 1_000_000)

    elif api == "v3/lessons/generate_artifact":
        # Each lesson uses GPT-4.1
        input_per_lesson = 5500
        output_per_lesson = 4000
        total_input = input_per_lesson * lesson_count
        total_output = output_per_lesson * lesson_count
        return (total_input * PRICING["gpt-4.1"]["input"] / 1_000_000 +
                total_output * PRICING["gpt-4.1"]["output"] / 1_000_000)

    return 0.0


async def test_v1_generate(client: httpx.AsyncClient, base_url: str, image_urls: list[str], version: str) -> BenchmarkResult:
    """Test v1/lessons/generate endpoint"""
    result = BenchmarkResult(
        api="v1/lessons/generate",
        version=version,
        image_count=len(image_urls),
    )

    payload = {
        "profile_id": PROFILE_ID,
        "image_urls": image_urls,
        "stream": False,
        "optional_parent_config": {
            "subject": "english",
            "purpose": "review",
            "child_age": 6,
            "child_name": "Benchmark Kid",
        }
    }

    start = time.monotonic()
    try:
        response = await client.post(
            f"{base_url}/v1/lessons/generate",
            json=payload,
            timeout=120.0,
        )
        result.latency_ms = int((time.monotonic() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            result.status = "success"
            result.lessons_count = len(data.get("lessons", []))

            # Extract token usage if available
            metadata = data.get("metadata", {})
            usage = metadata.get("usage") or metadata.get("token_usage", {})
            result.token_usage = usage
            result.model_used = metadata.get("model_used", "unknown")

            # Calculate cost
            if usage and version == "optimized":
                result.cost_usd = metadata.get("cost_usd", 0)
            else:
                result.cost_usd = estimate_original_cost("v1/lessons/generate", len(image_urls))
        else:
            result.status = "error"
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        result.latency_ms = int((time.monotonic() - start) * 1000)
        result.status = "error"
        result.error = str(e)

    return result


async def test_v3_generate(client: httpx.AsyncClient, base_url: str, image_urls: list[str], version: str) -> BenchmarkResult:
    """Test v3/lessons/generate endpoint"""
    result = BenchmarkResult(
        api="v3/lessons/generate",
        version=version,
        image_count=len(image_urls),
    )

    payload = {
        "profile_id": PROFILE_ID,
        "image_urls": image_urls,
        "stream": False,
        "optional_parent_config": {
            "subject": "english",
            "purpose": "review",
            "child_age": 6,
            "child_name": "Benchmark Kid",
        }
    }

    start = time.monotonic()
    try:
        response = await client.post(
            f"{base_url}/v3/lessons/generate",
            json=payload,
            timeout=120.0,
        )
        result.latency_ms = int((time.monotonic() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            result.status = "success"

            # Handle both response formats
            suggested = data.get("suggested_lessons") or data.get("lessons", [])
            result.lessons_count = len(suggested)

            metadata = data.get("metadata", {})
            usage = metadata.get("usage") or metadata.get("token_usage", {})
            result.token_usage = usage
            result.model_used = metadata.get("model_used", "unknown")

            if usage and version == "optimized":
                result.cost_usd = metadata.get("cost_usd", 0)
            else:
                result.cost_usd = estimate_original_cost("v3/lessons/generate", len(image_urls))
        else:
            result.status = "error"
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        result.latency_ms = int((time.monotonic() - start) * 1000)
        result.status = "error"
        result.error = str(e)

    return result


async def test_v3_artifact(client: httpx.AsyncClient, base_url: str, lessons: list[dict], version: str) -> BenchmarkResult:
    """Test v3/lessons/generate_artifact endpoint"""
    result = BenchmarkResult(
        api="v3/lessons/generate_artifact",
        version=version,
        lesson_count=len(lessons),
    )

    payload = {
        "profile_id": PROFILE_ID,
        "lessons": lessons,
        "stream": False,
    }

    start = time.monotonic()
    try:
        response = await client.post(
            f"{base_url}/v3/lessons/generate_artifact",
            json=payload,
            timeout=300.0,  # Longer timeout for many lessons
        )
        result.latency_ms = int((time.monotonic() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            result.status = "success"
            result.lessons_count = len(data.get("lessons", []))

            metadata = data.get("metadata", {})
            usage = metadata.get("usage") or metadata.get("token_usage", {})
            result.token_usage = usage
            result.model_used = metadata.get("model_used", "unknown")

            if usage and version == "optimized":
                result.cost_usd = metadata.get("cost_usd", 0)
            else:
                result.cost_usd = estimate_original_cost("v3/lessons/generate_artifact", lesson_count=len(lessons))
        else:
            result.status = "error"
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        result.latency_ms = int((time.monotonic() - start) * 1000)
        result.status = "error"
        result.error = str(e)

    return result


def generate_sample_lessons(count: int) -> list[dict]:
    """Generate sample lessons for artifact testing"""
    templates = [
        {"title": "Learn Colors", "content": "Red, Blue, Green, Yellow", "option": "vocabulary"},
        {"title": "Animal Names", "content": "Dog, Cat, Bird, Fish", "option": "vocabulary"},
        {"title": "Numbers 1-10", "content": "One, Two, Three, Four, Five", "option": "vocabulary"},
        {"title": "Family Members", "content": "Mom, Dad, Brother, Sister", "option": "vocabulary"},
        {"title": "Body Parts", "content": "Head, Hand, Foot, Eye, Ear", "option": "vocabulary"},
        {"title": "Fruits", "content": "Apple, Banana, Orange, Grape", "option": "vocabulary"},
        {"title": "Vegetables", "content": "Carrot, Tomato, Potato, Onion", "option": "vocabulary"},
        {"title": "Weather", "content": "Sunny, Rainy, Cloudy, Windy", "option": "vocabulary"},
        {"title": "Days of Week", "content": "Monday, Tuesday, Wednesday", "option": "vocabulary"},
    ]

    lessons = []
    for i in range(count):
        template = templates[i % len(templates)]
        lessons.append({
            "title": f"{template['title']} #{i+1}",
            "content": template["content"],
            "option": template["option"],
            "agent_mode": "learn_agent",
            "template_id": "PTL_LEARN_GENERIC",
        })
    return lessons


async def check_service(client: httpx.AsyncClient, base_url: str, name: str) -> bool:
    """Check if service is available"""
    try:
        # Try common health endpoints
        for endpoint in ["/health", "/docs", "/"]:
            response = await client.get(f"{base_url}{endpoint}", timeout=10.0)
            if response.status_code in [200, 401, 404]:  # Service is responding
                print(f"✓ {name} service at {base_url} is reachable")
                return True
    except Exception as e:
        print(f"✗ {name} service at {base_url} is NOT reachable: {e}")
    return False


async def run_benchmark():
    """Run full benchmark comparing original vs optimized"""
    print("=" * 60)
    print("BENCHMARK: Original vs Optimized Service")
    print("=" * 60)
    print(f"Original: {ORIGINAL_BASE_URL}")
    print(f"Optimized: {OPTIMIZED_BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    results = []

    async with httpx.AsyncClient() as client:
        # Check services
        original_available = await check_service(client, ORIGINAL_BASE_URL, "Original")
        optimized_available = await check_service(client, OPTIMIZED_BASE_URL, "Optimized")

        if not original_available and not optimized_available:
            print("\n❌ Neither service is available. Exiting.")
            return

        print()

        # Test V1 Generate
        print("=" * 40)
        print("Testing: v1/lessons/generate")
        print("=" * 40)

        for img_count in IMAGE_TEST_CASES:
            images = IMAGE_URLS[:img_count]
            print(f"\n[{img_count} images]")

            if original_available:
                print(f"  Original: ", end="", flush=True)
                result = await test_v1_generate(client, ORIGINAL_BASE_URL, images, "original")
                results.append(result)
                if result.status == "success":
                    print(f"{result.latency_ms}ms, {result.lessons_count} lessons, ~${result.cost_usd:.4f}")
                else:
                    print(f"ERROR: {result.error[:50]}")

            if optimized_available:
                print(f"  Optimized: ", end="", flush=True)
                result = await test_v1_generate(client, OPTIMIZED_BASE_URL, images, "optimized")
                results.append(result)
                if result.status == "success":
                    print(f"{result.latency_ms}ms, {result.lessons_count} lessons, ${result.cost_usd:.4f}")
                else:
                    print(f"ERROR: {result.error[:50]}")

        # Test V3 Generate
        print("\n" + "=" * 40)
        print("Testing: v3/lessons/generate")
        print("=" * 40)

        for img_count in IMAGE_TEST_CASES:
            images = IMAGE_URLS[:img_count]
            print(f"\n[{img_count} images]")

            if original_available:
                print(f"  Original: ", end="", flush=True)
                result = await test_v3_generate(client, ORIGINAL_BASE_URL, images, "original")
                results.append(result)
                if result.status == "success":
                    print(f"{result.latency_ms}ms, {result.lessons_count} suggestions, ~${result.cost_usd:.4f}")
                else:
                    print(f"ERROR: {result.error[:50]}")

            if optimized_available:
                print(f"  Optimized: ", end="", flush=True)
                result = await test_v3_generate(client, OPTIMIZED_BASE_URL, images, "optimized")
                results.append(result)
                if result.status == "success":
                    print(f"{result.latency_ms}ms, {result.lessons_count} suggestions, ${result.cost_usd:.4f}")
                else:
                    print(f"ERROR: {result.error[:50]}")

        # Test V3 Artifact
        print("\n" + "=" * 40)
        print("Testing: v3/lessons/generate_artifact")
        print("=" * 40)

        for lesson_count in LESSON_TEST_CASES:
            lessons = generate_sample_lessons(lesson_count)
            print(f"\n[{lesson_count} lessons]")

            if original_available:
                print(f"  Original: ", end="", flush=True)
                result = await test_v3_artifact(client, ORIGINAL_BASE_URL, lessons, "original")
                results.append(result)
                if result.status == "success":
                    print(f"{result.latency_ms}ms ({result.latency_ms/1000:.1f}s), ~${result.cost_usd:.4f}")
                else:
                    print(f"ERROR: {result.error[:50]}")

            if optimized_available:
                print(f"  Optimized: ", end="", flush=True)
                result = await test_v3_artifact(client, OPTIMIZED_BASE_URL, lessons, "optimized")
                results.append(result)
                if result.status == "success":
                    print(f"{result.latency_ms}ms ({result.latency_ms/1000:.1f}s), ${result.cost_usd:.4f}")
                else:
                    print(f"ERROR: {result.error[:50]}")

    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "original_url": ORIGINAL_BASE_URL,
        "optimized_url": OPTIMIZED_BASE_URL,
        "results": [
            {
                "api": r.api,
                "version": r.version,
                "image_count": r.image_count,
                "lesson_count": r.lesson_count,
                "latency_ms": r.latency_ms,
                "status": r.status,
                "lessons_count": r.lessons_count,
                "token_usage": r.token_usage,
                "cost_usd": r.cost_usd,
                "model_used": r.model_used,
                "error": r.error,
            }
            for r in results
        ]
    }

    with open("tests/benchmark_compare_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("Report saved to: tests/benchmark_compare_report.json")
    print("=" * 60)

    # Print summary
    print("\n📊 SUMMARY")
    print("-" * 40)

    for api in ["v1/lessons/generate", "v3/lessons/generate", "v3/lessons/generate_artifact"]:
        api_results = [r for r in results if r.api == api and r.status == "success"]
        if not api_results:
            continue

        print(f"\n{api}:")
        original = [r for r in api_results if r.version == "original"]
        optimized = [r for r in api_results if r.version == "optimized"]

        if original and optimized:
            orig_avg = sum(r.latency_ms for r in original) / len(original)
            opt_avg = sum(r.latency_ms for r in optimized) / len(optimized)
            improvement = (orig_avg - opt_avg) / orig_avg * 100
            print(f"  Original avg: {orig_avg/1000:.1f}s")
            print(f"  Optimized avg: {opt_avg/1000:.1f}s")
            print(f"  Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
