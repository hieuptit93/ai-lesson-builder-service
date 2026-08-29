from .v1_generate import router as v1_router
from .v3_generate import router as v3_generate_router
from .v3_artifact import router as v3_artifact_router

__all__ = ["v1_router", "v3_generate_router", "v3_artifact_router"]
