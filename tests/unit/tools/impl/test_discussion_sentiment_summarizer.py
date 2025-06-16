### tests/unit/tools/impl/test_discussion_sentiment_summarizer.py
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.tools.impl.discussion_sentiment_summarizer import (
    DiscussionSentimentSummarizerTool,
)

# Constants for patching and logging
SUMMARIZER_MODULE_PATH = "genie_tooling.tools.impl.discussion_sentiment_summarizer"
SUMMARIZER_LOGGER_NAME = SUMMARIZER_MODULE_PATH
SENTIMENT_MODEL_AVAILABLE_PATH = f"{SUMMARIZER_MODULE_PATH}.SENTIMENT_MODEL_AVAILABLE"
PIPELINE_PATH = f"{SUMMARIZER_MODULE_PATH}.pipeline"


@pytest.fixture()
def mock_pipeline_instance() -> MagicMock:
    """Creates a mock of a Hugging Face pipeline instance."""
    pipeline_mock = MagicMock()
    pipeline_mock.return_value = [
        [
            {"label": "positive", "score": 0.9},
            {"label": "neutral", "score": 0.08},
            {"label": "negative", "score": 0.02},
        ]
    ]
    return pipeline_mock


@pytest.fixture()
def mock_key_provider() -> AsyncMock:
    """Provides a mock KeyProvider for the tool's execute signature."""
    return AsyncMock()


@pytest.mark.asyncio()
class TestSentimentSummarizerSetup:
    """Tests the setup logic of the summarizer tool."""

    @patch(SENTIMENT_MODEL_AVAILABLE_PATH, True)
    @patch(PIPELINE_PATH)
    async def test_setup_with_model_success(self, mock_pipeline: MagicMock, mock_key_provider: AsyncMock):
        """Test successful setup when the transformers library is available."""
        mock_pipeline.return_value = "mock_pipeline_object"
        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()

        assert tool._sentiment_pipeline == "mock_pipeline_object"
        mock_pipeline.assert_called_once()
        assert mock_pipeline.call_args.kwargs["model"] == tool._model_name
        assert mock_pipeline.call_args.kwargs["top_k"] is None

    @patch(SENTIMENT_MODEL_AVAILABLE_PATH, True)
    @patch(PIPELINE_PATH)
    async def test_setup_with_model_load_failure(self, mock_pipeline: MagicMock, caplog: pytest.LogCaptureFixture):
        """Test graceful failure during model loading."""
        caplog.set_level(logging.ERROR, logger=SUMMARIZER_LOGGER_NAME)
        mock_pipeline.side_effect = RuntimeError("Model not found on Hub")
        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()

        assert tool._sentiment_pipeline is None
        assert f"Failed to load sentiment pipeline for '{tool._model_name}'" in caplog.text

    @patch(SENTIMENT_MODEL_AVAILABLE_PATH, False)
    async def test_setup_library_not_available(self, caplog: pytest.LogCaptureFixture):
        """Test setup when the transformers library is not installed."""
        caplog.set_level(logging.WARNING, logger=SUMMARIZER_LOGGER_NAME)
        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()

        assert tool._sentiment_pipeline is None
        assert "Using naive keyword-based sentiment analysis" in caplog.text


@pytest.mark.asyncio()
class TestSentimentSummarizerExecution:
    """Tests the execute method under different conditions."""

    @patch(SENTIMENT_MODEL_AVAILABLE_PATH, True)
    @patch(PIPELINE_PATH)
    async def test_execute_with_model_positive(
        self, mock_pipeline_constructor: MagicMock, mock_pipeline_instance: MagicMock, mock_key_provider: AsyncMock
    ):
        """Test execution with the model returning a positive sentiment."""
        mock_pipeline_constructor.return_value = mock_pipeline_instance
        mock_pipeline_instance.return_value = [
            [
                {"label": "positive", "score": 0.98},
                {"label": "neutral", "score": 0.01},
                {"label": "negative", "score": 0.01},
            ]
        ]

        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()
        snippets = ["This new feature is amazing!", "I love the new update."]
        result = await tool.execute({"text_snippets": snippets}, mock_key_provider, context={})

        assert result["overall_sentiment"] == "positive"
        assert result["sentiment_scores"]["positive"] > 0.9

    @patch(SENTIMENT_MODEL_AVAILABLE_PATH, True)
    @patch(PIPELINE_PATH)
    async def test_execute_with_model_mixed_sentiment(
        self, mock_pipeline_constructor: MagicMock, mock_pipeline_instance: MagicMock, mock_key_provider: AsyncMock
    ):
        """Test execution with the model where sentiment is mixed."""
        mock_pipeline_constructor.return_value = mock_pipeline_instance
        mock_pipeline_instance.return_value = [
            [
                {"label": "positive", "score": 0.9},
                {"label": "neutral", "score": 0.08},
                {"label": "negative", "score": 0.02},
            ],
            [
                {"label": "positive", "score": 0.1},
                {"label": "neutral", "score": 0.1},
                {"label": "negative", "score": 0.8},
            ],
        ]
        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()
        snippets = ["It's great!", "It's terrible!"]
        result = await tool.execute({"text_snippets": snippets}, mock_key_provider, context={})

        assert result["overall_sentiment"] == "mixed"

    @patch(SENTIMENT_MODEL_AVAILABLE_PATH, False)
    async def test_execute_with_keyword_fallback_negative(self, mock_key_provider: AsyncMock):
        """Test the keyword-based fallback for negative sentiment."""
        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()
        snippets = [
            "This has too many issues.",
            "The performance is slow and buggy.",
            "I'm disappointed with the results.",
        ]
        result = await tool.execute({"text_snippets": snippets}, mock_key_provider, context={})

        assert result["overall_sentiment"] == "negative"
        assert result["sentiment_scores"]["negative_matches"] == 4

    @patch(SENTIMENT_MODEL_AVAILABLE_PATH, False)
    async def test_execute_with_keyword_fallback_mixed(self, mock_key_provider: AsyncMock):
        """Test the keyword-based fallback for mixed sentiment."""
        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()
        snippets = ["It has some good features, but also some problems."]
        result = await tool.execute({"text_snippets": snippets}, mock_key_provider, context={})

        assert result["overall_sentiment"] == "mixed"

    async def test_execute_no_snippets(self, mock_key_provider: AsyncMock):
        """Test execution with an empty list of snippets."""
        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()
        result = await tool.execute({"text_snippets": []}, mock_key_provider, context={})

        assert "No valid list of text snippets provided" in result["notes"]
        assert result["overall_sentiment"] == "neutral"

    async def test_key_theme_extraction(self, mock_key_provider: AsyncMock):
        """Test the simple keyword-based theme extraction."""
        tool = DiscussionSentimentSummarizerTool()
        await tool.setup()
        snippets = ["The new API is very fast.", "API performance is great.", "The documentation for the API could be better."]
        result = await tool.execute({"text_snippets": snippets}, mock_key_provider, context={})

        expected_themes = {"api", "better", "documentation", "fast", "great"}
        assert set(result["key_themes"]) == expected_themes
