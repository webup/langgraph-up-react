import os
from unittest.mock import patch

import pytest

from common.context import Context
from tests.test_data import TestModels


class TestContextConfiguration:
    """Test suite for Context configuration."""

    def test_context_init_with_explicit_model(self) -> None:
        """Test context initialization with explicit model parameter."""
        context = Context(model=TestModels.OPENAI_GPT4O_MINI)
        assert context.model == TestModels.OPENAI_GPT4O_MINI
        assert context.system_prompt  # Should have default prompt
        assert context.max_search_results == 5  # Default value

    @patch.dict(os.environ, {"MODEL": TestModels.OPENAI_GPT4O_MINI}, clear=False)
    def test_context_init_with_env_vars(self) -> None:
        """Test context initialization using environment variables."""
        context = Context()
        assert context.model == TestModels.OPENAI_GPT4O_MINI

    @patch.dict(os.environ, {"MODEL": TestModels.OPENAI_GPT4O_MINI}, clear=False)
    def test_context_init_explicit_overrides_env(self) -> None:
        """Test that explicit parameters override environment variables."""
        context = Context(model=TestModels.ANTHROPIC_SONNET)
        assert context.model == TestModels.ANTHROPIC_SONNET

    @pytest.mark.parametrize(
        "model",
        [
            TestModels.QWEN_PLUS,
            TestModels.QWEN_TURBO,
            TestModels.QWQ_32B,
            TestModels.QVQ_72B,
            TestModels.OPENAI_GPT4O_MINI,
            TestModels.ANTHROPIC_SONNET,
        ],
    )
    def test_colon_separator_format(self, model: str) -> None:
        """Test that colon separator format works for all supported models."""
        context = Context(model=model)
        assert context.model == model
        assert ":" in context.model, "Model should use colon separator format"

    def test_custom_system_prompt(self) -> None:
        """Test context initialization with custom system prompt."""
        custom_prompt = "You are a specialized AI assistant for testing purposes."
        context = Context(model=TestModels.QWEN_PLUS, system_prompt=custom_prompt)
        assert context.system_prompt == custom_prompt

    def test_custom_max_search_results(self) -> None:
        """Test context initialization with custom max search results."""
        custom_max = 10
        context = Context(model=TestModels.QWEN_PLUS, max_search_results=custom_max)
        assert context.max_search_results == custom_max

    @patch.dict(
        os.environ,
        {"SYSTEM_PROMPT": "Custom env prompt", "MAX_SEARCH_RESULTS": "15"},
        clear=False,
    )
    def test_all_env_var_loading(self) -> None:
        """Test that all context fields can be loaded from environment variables."""
        context = Context()
        assert context.system_prompt == "Custom env prompt"
        # Environment variables are strings, so compare as string
        assert str(context.max_search_results) == "15"

    def test_qwen_model_variants(self) -> None:
        """Test different Qwen model variants."""
        test_cases = [
            (TestModels.QWEN_PLUS, "qwen-plus"),
            (TestModels.QWQ_32B, "qwq-32b-preview"),
            (TestModels.QVQ_72B, "qvq-72b-preview"),
        ]

        for full_model, expected_name in test_cases:
            context = Context(model=full_model)
            assert context.model == full_model
            assert expected_name in full_model

    @patch.dict(os.environ, {}, clear=True)
    def test_deepwiki_disabled_by_default(self) -> None:
        """Test that deepwiki is disabled by default."""
        context = Context()
        assert context.enable_deepwiki is False

    def test_deepwiki_can_be_enabled(self) -> None:
        """Test that deepwiki can be enabled explicitly."""
        context = Context(enable_deepwiki=True)
        assert context.enable_deepwiki is True

    @patch.dict(os.environ, {"ENABLE_DEEPWIKI": "true"}, clear=False)
    def test_deepwiki_env_var_loading(self) -> None:
        """Test that deepwiki can be enabled via environment variable."""
        context = Context()
        # Environment variable should be converted to boolean when current value equals default
        assert context.enable_deepwiki is True
