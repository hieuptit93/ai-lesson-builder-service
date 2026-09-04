"""Profile service for fetching and parsing user profiles."""

import time
from datetime import date, datetime

import structlog
from langfuse import observe

from app.domains.profile.domain.entities import ChildProfile, UserProfile
from app.domains.profile.infrastructure.profile_client import ProfileClient, ProfileClientError

# Alert system (graceful fallback if not configured)
try:
    from app.infrastructure.alerts.helpers.send_alert_safe import send_alert_safe
    from app.infrastructure.alerts.alert_types import AlertType, AlertLevel
except ImportError:
    send_alert_safe = lambda **kwargs: None  # noqa: E731
    AlertType = None
    AlertLevel = None

logger = structlog.get_logger()


class ProfileService:
    def __init__(self, profile_client: ProfileClient):
        self._client = profile_client

    @staticmethod
    def _parse_age(raw_value: str | int | None) -> int | None:
        """Parse age from API response.

        Handles two formats from external API:
          - Integer or numeric string: "4", 4
          - Date string (dd/mm/yyyy): "27/10/2017" -> calculate age from today
        """
        if raw_value is None:
            return None

        raw_str = str(raw_value).strip()
        if not raw_str:
            return None

        try:
            age = int(raw_str)
            if age > 0:
                return age
        except ValueError:
            pass

        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                birth = datetime.strptime(raw_str, fmt).date()
                today = date.today()
                age = today.year - birth.year - (
                    (today.month, today.day) < (birth.month, birth.day)
                )
                if age > 0:
                    return age
            except ValueError:
                continue

        logger.warning(
            "profile.age_parse_failed",
            raw_value=raw_value,
        )
        return None

    @observe(name="profile_fetch", capture_input=True, capture_output=True)
    async def fetch_profile(
        self,
        profile_id: str,
        token: str | None = None,
        conversation_id: str | None = None,
    ) -> UserProfile:
        """Fetch user profile from external API.

        Args:
            profile_id: Profile ID (user_id)
            token: Optional API token
            conversation_id: Optional conversation ID

        Returns:
            Parsed UserProfile
        """
        start = time.monotonic()

        logger.info(
            "profile.fetch.start",
            log_type="external_api",
            feature="USER",
            target_service="profile_api",
            user_id=profile_id,
        )

        try:
            raw_data = await self._client.fetch_profile(
                user_id=profile_id,
                token=token,
                conversation_id=conversation_id,
            )
        except ProfileClientError as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)

            logger.error(
                "profile.fetch.api_error",
                log_type="external_api",
                feature="USER",
                target_service="profile_api",
                user_id=profile_id,
                duration_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
                alert="profile_api_unreachable",
            )

            # Send alert to Google Chat
            if AlertType is not None:
                send_alert_safe(
                    alert_type=AlertType.EXTERNAL_API_ERROR,
                    level=AlertLevel.MEDIUM,
                    message=f"Profile API error → using default profile for profile_id={profile_id}",
                    context={"profile_id": profile_id, "error": str(exc), "error_type": type(exc).__name__},
                    component="ProfileService",
                )

            return UserProfile(
                user_id=profile_id,
                profile_id=profile_id,
                child=None,
                language_preference="vi",
                raw_data=None,
                is_degraded=True,
                degraded_reason=f"API error: {type(exc).__name__}: {exc}",
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)

            logger.error(
                "profile.fetch.unexpected_error",
                log_type="external_api",
                feature="USER",
                target_service="profile_api",
                user_id=profile_id,
                duration_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
                alert="profile_api_unexpected",
            )

            # Send alert to Google Chat
            if AlertType is not None:
                send_alert_safe(
                    alert_type=AlertType.EXTERNAL_API_ERROR,
                    level=AlertLevel.HIGH,
                    message=f"Profile API failed → child=None for profile_id={profile_id}",
                    context={"profile_id": profile_id, "error": str(exc), "error_type": type(exc).__name__},
                    component="ProfileService",
                )

            return UserProfile(
                user_id=profile_id,
                profile_id=profile_id,
                child=None,
                language_preference="vi",
                raw_data=None,
                is_degraded=True,
                degraded_reason=f"Unexpected error: {type(exc).__name__}: {exc}",
            )

        # --- Parse response (defensive: each field isolated) ---
        degraded_fields: list[str] = []

        # 1. Extract data envelope
        try:
            data = raw_data.get("data", {}) if isinstance(raw_data, dict) else {}
            if not isinstance(data, dict):
                logger.error(
                    "profile.parse.invalid_data_type",
                    log_type="external_api",
                    feature="USER",
                    target_service="profile_api",
                    user_id=profile_id,
                    data_type=type(data).__name__,
                    alert="profile_parse_unexpected_type",
                )
                data = {}
                degraded_fields.append("data_envelope")
        except Exception as exc:
            logger.error(
                "profile.parse.data_envelope_error",
                log_type="external_api",
                feature="USER",
                target_service="profile_api",
                user_id=profile_id,
                error=str(exc),
                alert="profile_parse_data_envelope",
            )
            data = {}
            degraded_fields.append("data_envelope")

        # 2. Parse child profile fields individually
        child = None
        child_name = data.get("name")
        if child_name:
            # 2a. Parse age
            raw_age = data.get("age")
            parsed_age = self._parse_age(raw_age)
            if raw_age is not None and parsed_age is None:
                degraded_fields.append("age")

            # 2b. Parse learning history
            learning_history: list[str] = []
            try:
                last_vocab = data.get("last_vocabulary")
                if last_vocab and isinstance(last_vocab, str):
                    learning_history = [v.strip() for v in last_vocab.split(",") if v.strip()]
            except Exception as exc:
                logger.error(
                    "profile.parse.learning_history_error",
                    log_type="external_api",
                    feature="USER",
                    target_service="profile_api",
                    user_id=profile_id,
                    error=str(exc),
                    alert="profile_parse_learning_history",
                )
                degraded_fields.append("learning_history")

            child = ChildProfile(
                child_id=profile_id,
                child_name=str(child_name),
                age=parsed_age,
                interests=[],
                learning_history=learning_history,
            )

        # 3. Parse language preference
        lang_map = {
            "VI": "vi",
            "EN": "en",
            "VI-EN": "bilingual",
        }
        language_preference = "vi"
        try:
            raw_lang = data.get("language_mode")
            if raw_lang is not None:
                lang_str = str(raw_lang).strip()
                language_preference = lang_map.get(lang_str, lang_str.lower() if lang_str else "vi")
        except Exception as exc:
            logger.error(
                "profile.parse.language_mode_error",
                log_type="external_api",
                feature="USER",
                target_service="profile_api",
                user_id=profile_id,
                raw_language_mode=str(data.get("language_mode")),
                error=str(exc),
                alert="profile_parse_language_mode",
            )
            degraded_fields.append("language_mode")

        # Build profile
        is_degraded = len(degraded_fields) > 0
        degraded_reason = f"Parse failed for: {', '.join(degraded_fields)}" if is_degraded else None

        user_profile = UserProfile(
            user_id=profile_id,
            profile_id=profile_id,
            child=child,
            language_preference=language_preference,
            raw_data=raw_data,
            is_degraded=is_degraded,
            degraded_reason=degraded_reason,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Build output content from profile
        output_content = ""
        if child:
            output_content = f"child_name={child.child_name}, age={child.age}, language={language_preference}"

        log_level = "warning" if is_degraded else "info"
        getattr(logger, log_level)(
            "profile.fetch.success",
            log_type="external_api",
            feature="USER",
            target_service="profile_api",
            user_id=profile_id,
            duration_ms=elapsed_ms,
            has_child=child is not None,
            is_degraded=is_degraded,
            degraded_fields=degraded_fields if degraded_fields else None,
            req_body=str({"user_id": profile_id, "token": "***" if token else None}),
            input_content=profile_id,
            res_body=str(data)[:500] if data else "",
            output_content=output_content,
            **({"alert": "profile_degraded_response"} if is_degraded else {}),
        )

        return user_profile
