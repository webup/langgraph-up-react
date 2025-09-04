"""SiliconFlow model integrations for ReAct agent."""

import os
from typing import Any, Optional

from langchain_siliconflow import ChatSiliconFlow

from ..utils import normalize_region


def create_siliconflow_model(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    region: Optional[str] = None,
    **kwargs: Any,
) -> ChatSiliconFlow:
    """Create a SiliconFlow model using ChatSiliconFlow.

    Args:
        model_name: The model name (e.g., 'Qwen/Qwen3-8B', 'THUDM/GLM-4.1V-9B-Thinking')
        api_key: SiliconFlow API key (defaults to env var SILICONFLOW_API_KEY)
        base_url: Custom base URL for API (optional)
        region: Region setting ('prc'/'cn' for China, 'international'/'en' for global)
                Defaults to env var REGION
        **kwargs: Additional model parameters

    Returns:
        Configured ChatSiliconFlow instance
    """
    # Get API key from env if not provided
    if api_key is None:
        api_key = os.getenv("SILICONFLOW_API_KEY")

    # Get region from env if not provided
    if region is None:
        region = os.getenv("REGION")

    # Set base URL based on region if not explicitly provided
    if base_url is None and region:
        # Normalize region aliases
        normalized_region = normalize_region(region)
        if normalized_region == "prc":
            base_url = "https://api.siliconflow.cn/v1"
        elif normalized_region == "international":
            base_url = "https://api.siliconflow.com/v1"

    # Create ChatSiliconFlow configuration
    config = {"model": model_name, "api_key": api_key, **kwargs}

    # Only add base_url if explicitly provided
    if base_url is not None:
        config["base_url"] = base_url

    return ChatSiliconFlow(**config)
