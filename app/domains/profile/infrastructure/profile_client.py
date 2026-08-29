"""Profile API client for fetching user profiles."""

import httpx

import structlog

logger = structlog.get_logger()


class ProfileClientError(Exception):
    """Profile client error."""
    pass


class ProfileClient:
    """Client for fetching user profile from external API."""

    def __init__(self, base_url: str, api_token: str):
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token

    async def fetch_profile(
        self,
        user_id: str,
        token: str | None = None,
        conversation_id: str | None = None,
    ) -> dict:
        """Fetch user profile from external API.

        Args:
            user_id: User ID
            token: Optional API token (overrides default)
            conversation_id: Optional conversation ID

        Returns:
            Raw profile data from API
        """
        url = f"{self._base_url}/user_profile"
        params = {
            "user_id": user_id,
            "token": token or self._api_token,
        }
        if conversation_id:
            params["conversation_id"] = conversation_id

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                logger.info("profile_api_request", url=url, params=params)
                response = await client.get(url, params=params)
                response.raise_for_status()
                logger.info("profile_fetched", user_id=user_id, status=response.status_code)
                return response.json()

        except httpx.HTTPStatusError as exc:
            logger.error("profile_http_error", user_id=user_id, status=exc.response.status_code, detail=exc.response.text[:200] if exc.response.text else None)
            raise ProfileClientError(f"HTTP {exc.response.status_code}") from exc
        except httpx.TimeoutException as exc:
            logger.error("profile_timeout", user_id=user_id, error=str(exc))
            raise ProfileClientError(f"Request timeout: {exc}") from exc
        except httpx.RequestError as exc:
            logger.error("profile_request_error", user_id=user_id, error=str(exc))
            raise ProfileClientError(f"Request failed: {exc}") from exc
