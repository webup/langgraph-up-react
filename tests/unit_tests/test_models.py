"""Test custom model integrations."""

import os
from unittest.mock import Mock, patch

import pytest

from common.models.qwen import create_qwen_model
from common.models.siliconflow import create_siliconflow_model
from common.utils import load_chat_model


@patch("common.models.qwen.ChatQwQ")
@patch.dict(os.environ, {"REGION": ""}, clear=False)
def test_create_qwen_model_qwq(mock_chat_qwq):
    """Test QwQ model creation uses ChatQwQ."""
    assert mock_chat_qwq.return_value == create_qwen_model("qwq-32b-preview", api_key="test-key")
    mock_chat_qwq.assert_called_once_with(model="qwq-32b-preview", api_key="test-key")


@patch("common.models.qwen.ChatQwQ")
@patch.dict(os.environ, {"REGION": ""}, clear=False)
def test_create_qwen_model_qvq(mock_chat_qwq):
    """Test QvQ model creation also uses ChatQwQ."""
    assert mock_chat_qwq.return_value == create_qwen_model("qvq-72b-preview", api_key="test-key")
    mock_chat_qwq.assert_called_once_with(model="qvq-72b-preview", api_key="test-key")


@patch("common.models.qwen.ChatQwen")
@patch.dict(os.environ, {"REGION": ""}, clear=False)
def test_create_qwen_model_qwen_plus(mock_chat_qwen):
    """Test Qwen+ model creation uses ChatQwen."""
    assert mock_chat_qwen.return_value == create_qwen_model("qwen-plus", api_key="test-key")
    mock_chat_qwen.assert_called_once_with(model="qwen-plus", api_key="test-key")


@patch("common.models.qwen.ChatQwQ")
def test_create_qwen_model_qwq_with_prc_region(mock_chat_qwq):
    """Test QwQ model creation with PRC region."""
    assert mock_chat_qwq.return_value == create_qwen_model("qwq-32b-preview", api_key="test-key", region="prc")
    mock_chat_qwq.assert_called_once_with(
        model="qwq-32b-preview",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


@patch("common.models.qwen.ChatQwen")
def test_create_qwen_model_qwen_plus_with_international_region(mock_chat_qwen):
    """Test Qwen+ model creation with international region."""
    assert mock_chat_qwen.return_value == create_qwen_model("qwen-plus", api_key="test-key", region="international")
    mock_chat_qwen.assert_called_once_with(
        model="qwen-plus",
        api_key="test-key",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


@patch("common.models.qwen.ChatQwen")
@patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key", "REGION": ""})
def test_create_qwen_model_with_env_key(mock_chat_qwen):
    """Test Qwen model creation using environment variable for API key."""
    assert mock_chat_qwen.return_value == create_qwen_model("qwen-plus")
    mock_chat_qwen.assert_called_once_with(model="qwen-plus", api_key="env-key")


@patch("common.models.qwen.ChatQwen")
@patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key", "REGION": "prc"})
def test_create_qwen_model_with_env_region_prc(mock_chat_qwen):
    """Test Qwen model creation using environment variable for region (PRC)."""
    assert mock_chat_qwen.return_value == create_qwen_model("qwen-plus")
    mock_chat_qwen.assert_called_once_with(
        model="qwen-plus",
        api_key="env-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


@patch("common.models.qwen.ChatQwQ")
@patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key", "REGION": "international"})
def test_create_qwq_model_with_env_region_international(mock_chat_qwq):
    """Test QwQ model creation using environment variable for region (international)."""
    assert mock_chat_qwq.return_value == create_qwen_model("qwq-32b-preview")
    mock_chat_qwq.assert_called_once_with(
        model="qwq-32b-preview",
        api_key="env-key",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


@patch("common.models.qwen.ChatQwQ")
def test_create_qwq_model_with_custom_base_url(mock_chat_qwq):
    """Test QwQ model creation with custom base URL."""
    custom_url = "https://custom.example.com/v1"
    assert mock_chat_qwq.return_value == create_qwen_model(
        "qwq-32b-preview", api_key="test-key", base_url=custom_url
    )
    mock_chat_qwq.assert_called_once_with(
        model="qwq-32b-preview", api_key="test-key", base_url=custom_url
    )


@patch("common.models.create_qwen_model")
def test_load_chat_model_qwen_provider(mock_create_qwen):
    """Test load_chat_model with Qwen provider."""
    assert mock_create_qwen.return_value == load_chat_model("qwen:qwq-32b-preview")
    mock_create_qwen.assert_called_once_with("qwq-32b-preview")


# SiliconFlow Tests
@patch("common.models.siliconflow.ChatSiliconFlow")
@patch.dict(os.environ, {"REGION": ""}, clear=False)
def test_create_siliconflow_model_basic(mock_chat_siliconflow):
    """Test basic SiliconFlow model creation."""
    assert mock_chat_siliconflow.return_value == create_siliconflow_model("Qwen/Qwen2.5-72B-Instruct", api_key="test-key")
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="test-key",
    )


@patch("common.models.siliconflow.ChatSiliconFlow")
def test_create_siliconflow_model_with_prc_region(mock_chat_siliconflow):
    """Test SiliconFlow model creation with PRC region."""
    assert mock_chat_siliconflow.return_value == create_siliconflow_model(
        "Qwen/Qwen2.5-72B-Instruct", api_key="test-key", region="prc"
    )
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
    )


@patch("common.models.siliconflow.ChatSiliconFlow")
def test_create_siliconflow_model_with_international_region(mock_chat_siliconflow):
    """Test SiliconFlow model creation with international region."""
    assert mock_chat_siliconflow.return_value == create_siliconflow_model(
        "Qwen/Qwen2.5-72B-Instruct", api_key="test-key", region="international"
    )
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="test-key",
        base_url="https://api.siliconflow.com/v1",
    )


@patch("common.models.siliconflow.ChatSiliconFlow")
def test_create_siliconflow_model_with_cn_alias(mock_chat_siliconflow):
    """Test SiliconFlow model creation with 'cn' alias for PRC."""
    assert mock_chat_siliconflow.return_value == create_siliconflow_model(
        "Qwen/Qwen2.5-72B-Instruct", api_key="test-key", region="cn"
    )
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
    )


@patch("common.models.siliconflow.ChatSiliconFlow")
def test_create_siliconflow_model_with_en_alias(mock_chat_siliconflow):
    """Test SiliconFlow model creation with 'en' alias for international."""
    assert mock_chat_siliconflow.return_value == create_siliconflow_model(
        "Qwen/Qwen2.5-72B-Instruct", api_key="test-key", region="en"
    )
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="test-key",
        base_url="https://api.siliconflow.com/v1",
    )


@patch("common.models.siliconflow.ChatSiliconFlow")
@patch.dict(os.environ, {"SILICONFLOW_API_KEY": "env-key", "REGION": ""})
def test_create_siliconflow_model_with_env_key(mock_chat_siliconflow):
    """Test SiliconFlow model creation using environment variable for API key."""
    assert mock_chat_siliconflow.return_value == create_siliconflow_model("Qwen/Qwen2.5-72B-Instruct")
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="env-key",
    )


@patch("common.models.siliconflow.ChatSiliconFlow")
@patch.dict(os.environ, {"SILICONFLOW_API_KEY": "env-key", "REGION": "prc"})
def test_create_siliconflow_model_with_env_region_prc(mock_chat_siliconflow):
    """Test SiliconFlow model creation using environment variable for region (PRC)."""
    assert mock_chat_siliconflow.return_value == create_siliconflow_model("Qwen/Qwen2.5-72B-Instruct")
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="env-key",
        base_url="https://api.siliconflow.cn/v1",
    )


@patch("common.models.siliconflow.ChatSiliconFlow")
@patch.dict(os.environ, {"SILICONFLOW_API_KEY": "env-key", "REGION": "international"})
def test_create_siliconflow_model_with_env_region_international(mock_chat_siliconflow):
    """Test SiliconFlow model creation using environment variable for region (international)."""
    assert mock_chat_siliconflow.return_value == create_siliconflow_model("Qwen/Qwen2.5-72B-Instruct")
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="env-key",
        base_url="https://api.siliconflow.com/v1",
    )


@patch("common.models.siliconflow.ChatSiliconFlow")
def test_create_siliconflow_model_with_custom_base_url(mock_chat_siliconflow):
    """Test SiliconFlow model creation with custom base URL."""
    custom_url = "https://custom.example.com/v1"
    assert mock_chat_siliconflow.return_value == create_siliconflow_model(
        "Qwen/Qwen2.5-72B-Instruct", api_key="test-key", base_url=custom_url
    )
    mock_chat_siliconflow.assert_called_once_with(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="test-key",
        base_url=custom_url,
    )


@patch("common.models.create_siliconflow_model")
def test_load_chat_model_siliconflow_provider(mock_create_siliconflow):
    """Test load_chat_model with SiliconFlow provider."""
    assert mock_create_siliconflow.return_value == load_chat_model("siliconflow:Qwen/Qwen2.5-72B-Instruct")
    mock_create_siliconflow.assert_called_once_with("Qwen/Qwen2.5-72B-Instruct")


@patch("common.utils.init_chat_model")
def test_load_chat_model_standard_provider(mock_init_chat_model):
    """Test load_chat_model with standard provider (non-QwQ)."""
    assert mock_init_chat_model.return_value == load_chat_model("openai:gpt-4o-mini")
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
