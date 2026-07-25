"""
Regression tests for LiteLLM provider tool-choice serialization.

The reflect agent selects a named tool through the canonical typed contract.

Responses-mode providers (e.g., GitHub Copilot via LiteLLM) expect the function
name at the top level instead of the Chat Completions shape:
  {"type": "function", "name": "recall"}

Sending the Chat Completions wire shape to Responses-mode models would fail with:
  "Invalid tool choice ... LLMToolChoice ... Expecting str, or dict"

This test verifies that litellm_llm.py serializes to the flat Responses shape.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_interface import (
    LLM_TOOL_CHOICE_AUTO,
    LLM_TOOL_CHOICE_NONE,
    LLMToolChoice,
)
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall semantic memories",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


@pytest.fixture
def mock_litellm():
    """Mock litellm module."""
    litellm_mock = MagicMock()
    litellm_mock.acompletion = AsyncMock()
    litellm_mock.drop_params = True
    litellm_mock.suppress_debug_info = True
    litellm_mock.get_max_tokens = MagicMock(return_value=8192)

    # Mock a successful response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = ""
    mock_message.tool_calls = []
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    litellm_mock.acompletion.return_value = mock_response

    return litellm_mock


@pytest.mark.asyncio
async def test_litellm_serializes_named_tool_choice_for_responses(mock_litellm):
    """Verify named tool choice is serialized to flat Responses shape."""
    with patch("hindsight_api.engine.providers.litellm_llm.litellm", mock_litellm):
        llm = LiteLLMLLM(
            provider="litellm",
            api_key="ignored",
            base_url="ignored",
            model="github_copilot/gpt-4o",
        )

        await llm.call_with_tools(
            messages=[{"role": "user", "content": "recall the memory"}],
            tools=TOOLS,
            tool_choice=LLMToolChoice.named("recall"),
            max_retries=0,
        )

        # Verify the payload sent to litellm.acompletion()
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "function", "name": "recall"}


@pytest.mark.asyncio
async def test_litellm_auto_mode_serializes_to_string(mock_litellm):
    """Verify AUTO mode serializes to string value."""
    with patch("hindsight_api.engine.providers.litellm_llm.litellm", mock_litellm):
        llm = LiteLLMLLM(
            provider="litellm",
            api_key="ignored",
            base_url="ignored",
            model="github_copilot/gpt-4o",
        )

        await llm.call_with_tools(
            messages=[{"role": "user", "content": "use tools as needed"}],
            tools=TOOLS,
            tool_choice=LLM_TOOL_CHOICE_AUTO,
            max_retries=0,
        )

        # Verify the payload sent to litellm.acompletion()
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_litellm_none_mode_serializes_to_string(mock_litellm):
    """Verify NONE mode serializes to string value."""
    with patch("hindsight_api.engine.providers.litellm_llm.litellm", mock_litellm):
        llm = LiteLLMLLM(
            provider="litellm",
            api_key="ignored",
            base_url="ignored",
            model="github_copilot/gpt-4o",
        )

        await llm.call_with_tools(
            messages=[{"role": "user", "content": "no tools"}],
            tools=TOOLS,
            tool_choice=LLM_TOOL_CHOICE_NONE,
            max_retries=0,
        )

        # Verify the payload sent to litellm.acompletion()
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["tool_choice"] == "none"