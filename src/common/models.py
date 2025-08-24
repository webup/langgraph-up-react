"""Custom model integrations for ReAct agent."""

import os
from typing import Any, List, Optional, Union

from langchain_qwq import ChatQwen, ChatQwQ


def create_qwen_model(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    region: Optional[str] = None,
    **kwargs: Any,
) -> Union[ChatQwQ, ChatQwen]:
    """Create a Qwen model with proper configuration.

    Args:
        model_name: The model name (e.g., 'qwq-32b-preview', 'qwen-plus')
        api_key: DashScope API key (defaults to env var DASHSCOPE_API_KEY)
        base_url: Custom base URL for API (optional)
        region: Region setting ('prc' for China, 'international' for global)
                Defaults to env var REGION
        **kwargs: Additional model parameters

    Returns:
        Configured ChatQwQ instance for QwQ/QvQ models or ChatQwen for other Qwen models
    """
    # Get API key from env if not provided
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")

    # Get region from env if not provided
    if region is None:
        region = os.getenv("REGION")

    # Set base URL based on region if not explicitly provided
    if base_url is None and region:
        if region.lower() == "prc":
            # China mainland endpoint
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif region.lower() == "international":
            # International endpoint
            base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # Create model configuration
    config = {"model": model_name, "api_key": api_key, **kwargs}

    if base_url:
        config["base_url"] = base_url

    # Select the appropriate chat model based on model name
    # Use ChatQwQ for QwQ and QvQ models, ChatQwen for other Qwen models
    if model_name.startswith(("qwq", "qvq")):
        return ChatQwQ(**config)
    else:
        return ChatQwen(**config)


def get_supported_qwen_models() -> List[str]:
    """Get list of supported Qwen models."""
    return [
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "qwq-32b-preview",
        "qvq-72b-preview",
        # Add more Qwen models as they become available
    ]
