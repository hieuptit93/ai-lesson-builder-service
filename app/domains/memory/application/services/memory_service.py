import time

import structlog

from app.domains.memory.domain.entities import MemoryFact, UserMemory
from app.domains.memory.infrastructure.mem0_client import Mem0Client

logger = structlog.get_logger()


class MemoryService:
    def __init__(self, mem0_client: Mem0Client):
        self._client = mem0_client

    async def fetch_user_memory(
        self,
        user_id: str,
        topic: str = "",
        subject: str = "",
    ) -> UserMemory:
        query = f"{topic} {subject} child learning".strip()
        start = time.monotonic()

        logger.info(
            "memory.fetch.start",
            log_type="external_api",
            feature="MEMORY",
            target_service="mem0",
            user_id=user_id,
            query=query,
        )

        try:
            # Build request body for logging
            req_body = {
                "query": query,
                "user_id": user_id,
                "top_k": 10,
                "limit": 10,
                "score_threshold": 0.5,
            }

            raw_facts = await self._client.search_facts(query=query, user_id=user_id)
            facts = [
                MemoryFact(
                    id=f.get("id", ""),
                    text=f.get("text", str(f)),
                    score=float(f.get("score", 0.0)),
                    metadata=f.get("metadata", {}),
                )
                for f in raw_facts
            ]

            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Build output content from facts
            output_content = "\n".join(f"- {f.text}" for f in facts) if facts else ""

            logger.info(
                "memory.fetch.success",
                log_type="external_api",
                feature="MEMORY",
                target_service="mem0",
                user_id=user_id,
                query=query,
                facts_count=len(facts),
                duration_ms=elapsed_ms,
                req_body=str(req_body),
                input_content=query,
                res_body=str(raw_facts),
                output_content=output_content,
            )

            return UserMemory(user_id=user_id, facts=facts, query_used=query, total_found=len(facts))

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)

            logger.error(
                "memory.fetch.error",
                log_type="external_api",
                feature="MEMORY",
                target_service="mem0",
                user_id=user_id,
                query=query,
                duration_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

            return UserMemory(user_id=user_id, facts=[], query_used=query, total_found=0)
