# src/genie_tooling/lookup/__init__.py
"""Tool Lookup functionality: Service and Types.
Providers are now in their own top-level package: genie_tooling.tool_lookup_providers
"""

from genie_tooling.tool_lookup_providers.abc import (
    ToolLookupProvider as ToolLookupProviderPlugin,
)
from genie_tooling.tool_lookup_providers.impl.embedding_similarity import (
    EmbeddingSimilarityLookupProvider,
)
from genie_tooling.tool_lookup_providers.impl.keyword_match import (
    KeywordMatchLookupProvider,
)

from .service import ToolLookupService
from .types import RankedToolResult

__all__ = [
    "ToolLookupService",
    "RankedToolResult",
    "ToolLookupProviderPlugin",         # Re-exporting the ABC
    "EmbeddingSimilarityLookupProvider",# Re-exporting common implementations
    "KeywordMatchLookupProvider",       # Re-exporting common implementations
]
