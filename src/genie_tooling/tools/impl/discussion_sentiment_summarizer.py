# src/genie_tooling/tools/impl/discussion_sentiment_summarizer.py
import asyncio
import functools
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline

    SENTIMENT_MODEL_AVAILABLE = True
except ImportError:
    pipeline = None
    SENTIMENT_MODEL_AVAILABLE = False
    logger.warning(
        "discussion_sentiment_summarizer: 'transformers' library not installed. "
        "This tool will use a naive keyword-based fallback. "
        "For accurate sentiment analysis, please install with: poetry add transformers torch"  # Or your preferred backend
    )


class DiscussionSentimentSummarizerTool(Tool):
    identifier: str = "discussion_sentiment_summarizer"
    plugin_id: str = "discussion_sentiment_summarizer"

    _sentiment_pipeline: Optional[Any] = None
    _model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Default model

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        # Allow model_name to be configured
        self._model_name = cfg.get("sentiment_model_name", self._model_name)

        if SENTIMENT_MODEL_AVAILABLE and pipeline:
            try:
                loop = asyncio.get_running_loop()
                partial_pipeline_call = functools.partial(
                    pipeline, "sentiment-analysis", model=self._model_name, top_k=None
                )
                self._sentiment_pipeline = await loop.run_in_executor(
                    None, partial_pipeline_call
                )
                logger.info(
                    f"{self.identifier}: Sentiment analysis pipeline for '{self._model_name}' loaded successfully."
                )
            except Exception as e:
                logger.error(
                    f"{self.identifier}: Failed to load sentiment pipeline for '{self._model_name}': {e}. Will use fallback.",
                    exc_info=True,
                )
                self._sentiment_pipeline = None
        else:
            logger.warning(
                f"{self.identifier}: Using naive keyword-based sentiment analysis as 'transformers' is not available."
            )

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Discussion Sentiment Summarizer",
            "description_human": "Analyzes a list of text snippets (e.g., from forum discussions or search results) to determine overall sentiment, key themes, and common concerns.",
            "description_llm": "SentimentAnalyzer: Analyzes a list of text snippets to determine overall sentiment, key themes, and common concerns. Args: text_snippets (List[str]).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text_snippets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of text strings to analyze.",
                    }
                },
                "required": ["text_snippets"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "overall_sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative", "mixed"],
                    },
                    "sentiment_scores": {
                        "type": "object",
                        "description": "Average scores for each category across all snippets.",
                    },
                    "key_themes": {"type": "array", "items": {"type": "string"}},
                    "common_concerns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "notes": {"type": "string"},
                },
            },
            "key_requirements": [],
            "tags": ["analysis", "sentiment", "text", "nlp"],
            "version": "2.2.0",
            "cacheable": True,
            "cache_ttl_seconds": 3600,
        }

    async def execute(
        self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        text_snippets = params.get("text_snippets", [])
        if (
            not text_snippets
            or not isinstance(text_snippets, list)
            or not all(isinstance(s, str) for s in text_snippets)
        ):
            return {
                "overall_sentiment": "neutral",
                "sentiment_scores": {},
                "key_themes": [],
                "common_concerns": [],
                "notes": "No valid list of text snippets provided.",
            }

        if self._sentiment_pipeline:
            return await self._execute_with_model(text_snippets)
        else:
            return self._execute_with_keywords(text_snippets)

    async def _execute_with_model(
        self, text_snippets: List[str]
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        predictions_nested_list = await loop.run_in_executor(
            None, self._sentiment_pipeline, text_snippets
        )
        aggregated_scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        valid_snippets_count = 0

        for snippet_prediction_result in predictions_nested_list:
            actual_predictions_for_snippet: List[Dict[str, Any]] = []
            if isinstance(snippet_prediction_result, list):
                actual_predictions_for_snippet = snippet_prediction_result
            elif isinstance(snippet_prediction_result, dict):
                actual_predictions_for_snippet = [snippet_prediction_result]

            if not actual_predictions_for_snippet:
                continue

            valid_snippets_count += 1
            for pred in actual_predictions_for_snippet:
                label = str(pred.get("label", "unknown")).lower()
                score = pred.get("score")
                if label.startswith("neg"):
                    label = "negative"
                elif label.startswith("neu"):
                    label = "neutral"
                elif label.startswith("pos"):
                    label = "positive"
                if label in aggregated_scores and isinstance(score, (float, int)):
                    aggregated_scores[label] += float(score)

        if valid_snippets_count > 0:
            for label in aggregated_scores:
                aggregated_scores[label] /= valid_snippets_count

        dominant_sentiment = "neutral"
        if aggregated_scores:
            max_score = -1.0
            for label, score_val in aggregated_scores.items():
                if score_val > max_score:
                    max_score = score_val
                    dominant_sentiment = label
            if valid_snippets_count > 1 and max_score < 0.6:
                sorted_scores = sorted(aggregated_scores.values(), reverse=True)
                if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.2:
                    dominant_sentiment = "mixed"

        full_text = " ".join(text_snippets).lower()
        words = re.findall(r"\b\w{3,}\b", full_text)
        common_stopwords = {
            "the", "and", "for", "with", "this", "that", "was", "are", "its", "from",
            "has", "been", "have", "will", "about", "also", "what", "which", "when",
            "where", "their", "they", "them", "some", "other", "more", "most", "just",
            "like", "into", "than", "then", "very", "such", "many", "much", "even",
            "here", "there", "these", "those", "could",
        }
        filtered_words = [
            word for word in words if word not in common_stopwords and not word.isdigit()
        ]
        counts = Counter(filtered_words)
        sorted_themes = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        key_themes = [item[0] for item in sorted_themes[:5]]

        return {
            "overall_sentiment": dominant_sentiment,
            "sentiment_scores": {k: round(v, 4) for k, v in aggregated_scores.items()},
            "key_themes": key_themes,
            "common_concerns": [],
            "notes": f"Sentiment analyzed using '{self._model_name}' pipeline on {valid_snippets_count} snippets.",
        }

    def _execute_with_keywords(
        self, text_snippets: List[str]
    ) -> Dict[str, Any]:
        full_text = " ".join(text_snippets).lower()
        words = re.findall(r"\b\w{3,}\b", full_text)
        total_valid_words = len(words)
        if total_valid_words == 0:
            return {
                "overall_sentiment": "neutral", "sentiment_scores": {}, "key_themes": [],
                "common_concerns": [], "notes": "No valid words found.",
            }

        positive_keywords = {
            "good", "great", "excellent", "amazing", "impressive", "strong",
            "positive", "excited", "love", "best", "success", "powerful",
            "efficient", "fast", "happy", "glad", "benefit", "helpful", "support",
        }
        negative_keywords = {
            "bad", "poor", "terrible", "disappointed", "overhyped", "expensive",
            "late", "problems", "concern", "issue", "slow", "buggy", "fail", "error",
            "worst", "negative", "sad", "angry", "unable", "issues",
        }

        positive_score = sum(1 for word in words if word in positive_keywords)
        negative_score = sum(1 for word in words if word in negative_keywords)

        sentiment_score_raw = (
            (positive_score - negative_score) / total_valid_words
            if total_valid_words > 0
            else 0.0
        )
        sentiment_label = "neutral"
        if sentiment_score_raw > 0.05:
            sentiment_label = "positive"
        elif sentiment_score_raw < -0.05:
            sentiment_label = "negative"
        elif positive_score > 0 and negative_score > 0:
            sentiment_label = "mixed"

        common_stopwords = {
            "the", "and", "for", "with", "this", "that", "was", "are", "its", "from",
            "has", "been", "have", "will", "about", "also", "what", "which", "when",
            "where", "their", "they", "them", "some", "other", "more", "most", "just",
            "like", "into", "than", "then", "very", "such", "many", "much", "even",
            "here", "there", "these", "those", "could",
        }
        filtered_words = [
            word for word in words if word not in common_stopwords and not word.isdigit() and len(word) >= 3
        ]
        counts = Counter(filtered_words)
        sorted_themes = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        key_themes = [item[0] for item in sorted_themes[:5]]

        return {
            "overall_sentiment": sentiment_label,
            "sentiment_scores": {
                "positive_matches": positive_score,
                "negative_matches": negative_score,
                "raw_score_normalized": round(sentiment_score_raw, 3),
            },
            "key_themes": key_themes,
            "common_concerns": [],
            "notes": "Sentiment analyzed using naive keyword matching (fallback).",
        }

    async def teardown(self) -> None:
        self._sentiment_pipeline = None
        logger.debug(f"{self.identifier}: Teardown complete, sentiment model released.")