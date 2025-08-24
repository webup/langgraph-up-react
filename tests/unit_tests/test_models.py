"""Test custom model integrations."""

import os
from unittest.mock import Mock, patch

import pytest

from common.models import create_qwen_model, get_supported_qwen_models
from common.utils import load_chat_model


def test_get_supported_qwen_models():
    """Test that supported Qwen models are returned."""
    models = get_supported_qwen_models()
    assert isinstance(models, list)
    assert "qwq-32b-preview" in models


@patch("common.models.ChatQwQ")
@patch.dict(os.environ, {"REGION": ""}, clear=False)
def test_create_qwen_model_qwq(mock_chat_qwq):
    """Test QwQ model creation uses ChatQwQ."""
    mock_instance = Mock()
    mock_chat_qwq.return_value = mock_instance

    result = create_qwen_model("qwq-32b-preview", api_key="test-key")

    assert result == mock_instance
    mock_chat_qwq.assert_called_once_with(model="qwq-32b-preview", api_key="test-key")


@patch("common.models.ChatQwQ")
@patch.dict(os.environ, {"REGION": ""}, clear=False)
def test_create_qwen_model_qvq(mock_chat_qwq):
    """Test QvQ model creation also uses ChatQwQ."""
    mock_instance = Mock()
    mock_chat_qwq.return_value = mock_instance

    result = create_qwen_model("qvq-72b-preview", api_key="test-key")

    assert result == mock_instance
    mock_chat_qwq.assert_called_once_with(model="qvq-72b-preview", api_key="test-key")


@patch("common.models.ChatQwen")
@patch.dict(os.environ, {"REGION": ""}, clear=False)
def test_create_qwen_model_qwen_plus(mock_chat_qwen):
    """Test Qwen+ model creation uses ChatQwen."""
    mock_instance = Mock()
    mock_chat_qwen.return_value = mock_instance

    result = create_qwen_model("qwen-plus", api_key="test-key")

    assert result == mock_instance
    mock_chat_qwen.assert_called_once_with(model="qwen-plus", api_key="test-key")


@patch("common.models.ChatQwQ")
def test_create_qwen_model_qwq_with_prc_region(mock_chat_qwq):
    """Test QwQ model creation with PRC region."""
    mock_instance = Mock()
    mock_chat_qwq.return_value = mock_instance

    result = create_qwen_model("qwq-32b-preview", api_key="test-key", region="prc")

    assert result == mock_instance
    mock_chat_qwq.assert_called_once_with(
        model="qwq-32b-preview",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


@patch("common.models.ChatQwen")
def test_create_qwen_model_qwen_plus_with_international_region(mock_chat_qwen):
    """Test Qwen+ model creation with international region."""
    mock_instance = Mock()
    mock_chat_qwen.return_value = mock_instance

    result = create_qwen_model("qwen-plus", api_key="test-key", region="international")

    assert result == mock_instance
    mock_chat_qwen.assert_called_once_with(
        model="qwen-plus",
        api_key="test-key",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


@patch("common.models.ChatQwen")
@patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key", "REGION": ""})
def test_create_qwen_model_with_env_key(mock_chat_qwen):
    """Test Qwen model creation using environment variable for API key."""
    mock_instance = Mock()
    mock_chat_qwen.return_value = mock_instance

    result = create_qwen_model("qwen-plus")

    assert result == mock_instance
    mock_chat_qwen.assert_called_once_with(model="qwen-plus", api_key="env-key")


@patch("common.models.ChatQwen")
@patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key", "REGION": "prc"})
def test_create_qwen_model_with_env_region_prc(mock_chat_qwen):
    """Test Qwen model creation using environment variable for region (PRC)."""
    mock_instance = Mock()
    mock_chat_qwen.return_value = mock_instance

    result = create_qwen_model("qwen-plus")

    assert result == mock_instance
    mock_chat_qwen.assert_called_once_with(
        model="qwen-plus",
        api_key="env-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


@patch("common.models.ChatQwQ")
@patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key", "REGION": "international"})
def test_create_qwq_model_with_env_region_international(mock_chat_qwq):
    """Test QwQ model creation using environment variable for region (international)."""
    mock_instance = Mock()
    mock_chat_qwq.return_value = mock_instance

    result = create_qwen_model("qwq-32b-preview")

    assert result == mock_instance
    mock_chat_qwq.assert_called_once_with(
        model="qwq-32b-preview",
        api_key="env-key",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


@patch("common.models.ChatQwQ")
def test_create_qwq_model_with_custom_base_url(mock_chat_qwq):
    """Test QwQ model creation with custom base URL."""
    mock_instance = Mock()
    mock_chat_qwq.return_value = mock_instance

    custom_url = "https://custom.example.com/v1"
    result = create_qwen_model(
        "qwq-32b-preview", api_key="test-key", base_url=custom_url
    )

    assert result == mock_instance
    mock_chat_qwq.assert_called_once_with(
        model="qwq-32b-preview", api_key="test-key", base_url=custom_url
    )


@patch("common.utils.create_qwen_model")
def test_load_chat_model_qwen_provider(mock_create_qwen):
    """Test load_chat_model with Qwen provider."""
    mock_instance = Mock()
    mock_create_qwen.return_value = mock_instance

    result = load_chat_model("qwen:qwq-32b-preview")

    assert result == mock_instance
    mock_create_qwen.assert_called_once_with("qwq-32b-preview")


@patch("common.utils.init_chat_model")
def test_load_chat_model_standard_provider(mock_init_chat_model):
    """Test load_chat_model with standard provider (non-QwQ)."""
    mock_instance = Mock()
    mock_init_chat_model.return_value = mock_instance

    result = load_chat_model("openai:gpt-4o-mini")

    assert result == mock_instance
    mock_init_chat_model.assert_called_once_with("gpt-4o-mini", model_provider="openai")


def test_load_chat_model_colon_separator_parsing():
    """Test that colon separator is parsed correctly."""
    with patch("common.utils.init_chat_model") as mock_init:
        mock_init.return_value = Mock()
        load_chat_model("anthropic:claude-3-sonnet")
        mock_init.assert_called_once_with("claude-3-sonnet", model_provider="anthropic")


def test_load_chat_model_invalid_format():
    """Test that load_chat_model raises error for invalid format."""
    with pytest.raises(ValueError):
        load_chat_model("invalid-format-without-separator")
