# src/genie_tooling/tools/impl/intelligent_search_aggregator_tool.py
import asyncio
import logging
import re
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
)

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    logger.warning("IntelligentSearchAggregatorTool: NumPy not found. Semantic scoring might be limited.")

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    RANK_BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None # type: ignore
    RANK_BM25_AVAILABLE = False
    logger.warning("IntelligentSearchAggregatorTool: rank_bm25 library not found. BM25 scoring will be disabled.")

class CandidateItem:
    def __init__(self, id: str, title: str, text_for_scoring: str, url: str, source: str, original_rank: int, raw_item: Dict[str, Any]):
        self.id = id
        self.title = title
        self.text_for_scoring = text_for_scoring
        self.url = url
        self.source = source
        self.original_rank = original_rank
        self.raw_item = raw_item
        self.scores: Dict[str, Optional[float]] = {"keyword": None, "semantic": None, "bm25": None, "combined_weighted": 0.0}

class _TempChunkForEmbedding(Chunk):
    def __init__(self, id: str, content: str):
        self.id: Optional[str] = id
        self.content: str = content
        self.metadata: Dict[str, Any] = {}

class IntelligentSearchAggregatorTool(Tool):
    plugin_id: str = "intelligent_search_aggregator_v1"
    identifier: str = "intelligent_search_aggregator_v1"
    _plugin_manager: PluginManager
    _key_provider: Optional[KeyProvider] = None
    _embedder: Optional[EmbeddingGeneratorPlugin] = None
    _default_google_search_tool_id: str = "community_google_search"
    _default_arxiv_search_tool_id: str = "arxiv_search_tool"
    _default_embedder_id: str = "sentence_transformer_embedder_v1"
    _google_search_tool_id: str
    _arxiv_search_tool_id: str
    _embedder_id: str
    _default_top_n_to_return: int = 10
    _default_weight_keyword: float = 0.3
    _default_weight_semantic: float = 0.4
    _default_weight_bm25: float = 0.3

    def __init__(self, plugin_manager: PluginManager):
        if not plugin_manager:
            raise ValueError("PluginManager is required for IntelligentSearchAggregatorTool.")
        self._plugin_manager = plugin_manager
        logger.info(f"{self.plugin_id}: Initialized with PluginManager.")

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._google_search_tool_id = cfg.get("google_search_tool_id", self._default_google_search_tool_id)
        self._arxiv_search_tool_id = cfg.get("arxiv_search_tool_id", self._default_arxiv_search_tool_id)
        self._embedder_id = cfg.get("embedder_id", self._default_embedder_id)
        embedder_config = cfg.get("embedder_config", {})
        embedder_config.setdefault("plugin_manager", self._plugin_manager)
        embedder_instance = await self._plugin_manager.get_plugin_instance(self._embedder_id, config=embedder_config)
        if embedder_instance and isinstance(embedder_instance, EmbeddingGeneratorPlugin):
            self._embedder = embedder_instance
            logger.info(f"{self.plugin_id}: Embedder '''{self._embedder_id}''' loaded successfully.")
        else:
            logger.error(f"{self.plugin_id}: Failed to load embedder '''{self._embedder_id}'''. Semantic scoring will be impaired.")
            self._embedder = None
        logger.info(f"{self.plugin_id}: Setup complete. Using Google Search: '''{self._google_search_tool_id}''', ArXiv Search: '''{self._arxiv_search_tool_id}''', Embedder: '''{self._embedder_id}'''.")

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier, "name": "Intelligent Search Aggregator",
            "description_human": "Performs a comprehensive search across multiple sources (Google, ArXiv), then re-ranks results using a hybrid scoring model to find the most relevant information.",
            "description_llm": "IntelligentSearch: Searches Google and ArXiv for a query, then re-ranks results. Args: query (str, req), num_google_results (int, opt, def:20), num_arxiv_results (int, opt, def:5), top_n_to_return (int, opt, def:10), weight_keyword (float, opt, def:0.3), weight_semantic (float, opt, def:0.4), weight_bm25 (float, opt, def:0.3), lang (str, opt, def:'en'), region (str, opt, def:None), google_safe_search (str, opt, enum['active','off'], def:'active'), google_sleep_interval (int, opt, def:0). Output is a dict with 'results' (list of dicts: title, url, snippet_or_summary, source, original_rank_in_source, scores, raw_item_from_source) and 'error'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The research query."},
                    "num_google_results": {"type": "integer", "default": 20, "description": "Number of results to request from Google."},
                    "num_arxiv_results": {"type": "integer", "default": 5, "description": "Number of results to request from ArXiv."},
                    "top_n_to_return": {"type": "integer", "default": self._default_top_n_to_return, "description": "Number of final re-ranked results to return."},
                    "weight_keyword": {"type": "number", "default": self._default_weight_keyword}, "weight_semantic": {"type": "number", "default": self._default_weight_semantic}, "weight_bm25": {"type": "number", "default": self._default_weight_bm25},
                    "lang": {"type": "string", "default": "en"}, "region": {"type": ["string", "null"], "default": None},
                    "google_safe_search": {"type": "string", "enum": ["active", "off"], "default": "active"}, "google_sleep_interval": {"type": "integer", "default": 0}
                }, "required": ["query"]
            },
            "output_schema": {
                "type": "object", "properties": {
                    "results": {"type": "array", "items": {"type": "object", "properties": {
                        "title": {"type": "string"}, "url": {"type": "string"}, "snippet_or_summary": {"type": "string"},
                        "source": {"type": "string", "enum": ["google", "arxiv"]}, "original_rank_in_source": {"type": "integer"},
                        "scores": {"type": "object", "properties": {"keyword": {"type": ["number", "null"]}, "semantic": {"type": ["number", "null"]}, "bm25": {"type": ["number", "null"]}, "combined_weighted": {"type": "number"}}},
                        "raw_item_from_source": {"type": "object"}
                    }}}, "error": {"type": ["string", "null"]}
                }, "required": ["results"]
            },
            "key_requirements": [], "tags": ["research", "search", "aggregation", "ranking", "hybrid"], "version": "1.3.1", "cacheable": True, "cache_ttl_seconds": 3600
        }

    async def execute(self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]) -> Dict[str, Any]:
        self._key_provider = key_provider
        query = params["query"]
        num_google = params.get("num_google_results", 20)
        num_arxiv = params.get("num_arxiv_results", 5)
        top_n_return = params.get("top_n_to_return", self._default_top_n_to_return)
        w_kw = params.get("weight_keyword", self._default_weight_keyword)
        w_sem = params.get("weight_semantic", self._default_weight_semantic)
        w_bm25 = params.get("weight_bm25", self._default_weight_bm25)
        genie_instance: Optional["Genie"] = context.get("genie_framework_instance")
        if not genie_instance:
            return {"results": [], "error": "Genie framework instance not found in context."}

        google_params = {"query": query, "num_results": num_google, "advanced": True, "lang": params.get("lang", "en"), "region": params.get("region"), "safe": params.get("google_safe_search", "active"), "sleep_interval": params.get("google_sleep_interval", 0)}
        arxiv_params = {"query": query, "max_results": num_arxiv}
        google_task = genie_instance.execute_tool(self._google_search_tool_id, **google_params)
        arxiv_task = genie_instance.execute_tool(self._arxiv_search_tool_id, **arxiv_params)
        raw_google_response, raw_arxiv_response = await asyncio.gather(google_task, arxiv_task, return_exceptions=True)

        candidate_items: List[CandidateItem] = []
        if isinstance(raw_google_response, dict) and raw_google_response.get("results"):
            for i, res in enumerate(raw_google_response["results"]):
                url = res.get("url", "")
                if url:
                    candidate_items.append(CandidateItem(id=url, title=res.get("title", ""), text_for_scoring=f"{res.get('title', '')} {res.get('description', '')}", url=url, source="google", original_rank=i + 1, raw_item=res))
        elif isinstance(raw_google_response, Exception):
            logger.warning(f"{self.plugin_id}: Google search failed: {raw_google_response}")

        if isinstance(raw_arxiv_response, dict) and raw_arxiv_response.get("results"):
            for i, res in enumerate(raw_arxiv_response["results"]):
                url = res.get("pdf_url", res.get("entry_id", ""))
                if url:
                    candidate_items.append(CandidateItem(id=res.get("entry_id", ""), title=res.get("title", ""), text_for_scoring=f"{res.get('title', '')} {res.get('summary', '')}", url=url, source="arxiv", original_rank=i + 1, raw_item=res))
        elif isinstance(raw_arxiv_response, Exception):
            logger.warning(f"{self.plugin_id}: ArXiv search failed: {raw_arxiv_response}")

        if not candidate_items:
            return {"results": [], "error": "No search results from any source."}

        self._calculate_keyword_scores(query, candidate_items)
        query_embedding_vector = await self._get_query_embedding(query)
        item_embeddings_list = await self._get_item_embeddings(candidate_items)
        self._calculate_semantic_scores(query_embedding_vector, candidate_items, item_embeddings_list)
        self._calculate_bm25_scores(query, candidate_items)

        for item in candidate_items:
            item.scores["combined_weighted"] = ((item.scores.get("keyword", 0.0) or 0.0) * w_kw) + ((item.scores.get("semantic", 0.0) or 0.0) * w_sem) + ((item.scores.get("bm25", 0.0) or 0.0) * w_bm25)

        candidate_items.sort(key=lambda x: x.scores["combined_weighted"] or 0.0, reverse=True)

        final_results = [{"title": item.title, "url": item.url, "snippet_or_summary": item.text_for_scoring, "source": item.source, "original_rank_in_source": item.original_rank, "scores": item.scores, "raw_item_from_source": item.raw_item} for item in candidate_items[:top_n_return]]
        return {"results": final_results, "error": None}

    async def teardown(self) -> None:
        self._embedder = None
        logger.info(f"{self.plugin_id}: Teardown complete.")

    async def _get_query_embedding(self, query: str) -> Optional[EmbeddingVector]:
        if not self._embedder:
            return None
        async def query_chunk_provider() -> AsyncIterable[Chunk]:
            yield _TempChunkForEmbedding(id="query_for_aggregator", content=query)
        try:
            async for _chunk, vector in self._embedder.embed(chunks=query_chunk_provider(), config={"plugin_manager": self._plugin_manager, "key_provider": self._key_provider}):
                return vector
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error generating query embedding: {e}", exc_info=True)
        return None
    async def _get_item_embeddings(self, items: List[CandidateItem]) -> List[Optional[EmbeddingVector]]:
        if not self._embedder or not items:
            return [None] * len(items)
        item_chunks: List[Chunk] = [_TempChunkForEmbedding(id=item.id, content=item.text_for_scoring) for item in items]
        async def item_chunk_provider() -> AsyncIterable[Chunk]:
            for chunk in item_chunks:
                yield chunk
        embeddings_map: Dict[str, EmbeddingVector] = {}
        try:
            async for chunk, vector in self._embedder.embed(chunks=item_chunk_provider(), config={"plugin_manager": self._plugin_manager, "key_provider": self._key_provider}):
                if chunk.id:
                    embeddings_map[chunk.id] = vector
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error generating item embeddings: {e}", exc_info=True)
            return [None] * len(items)
        return [embeddings_map.get(item.id) for item in items]
    def _calculate_keyword_scores(self, query: str, items: List[CandidateItem]) -> None:
        query_keywords = set(re.findall(r"\w+", query.lower()))
        if not query_keywords:
            return
        for item in items:
            item_text_keywords = set(re.findall(r"\w+", item.text_for_scoring.lower()))
            if not item_text_keywords:
                item.scores["keyword"] = 0.0
                continue
            common_keywords = query_keywords.intersection(item_text_keywords)
            score = len(common_keywords) / (len(query_keywords) + len(item_text_keywords) - len(common_keywords)) if (len(query_keywords) + len(item_text_keywords) - len(common_keywords)) > 0 else 0.0
            item.scores["keyword"] = score
    def _calculate_semantic_scores(self, query_embedding: Optional[EmbeddingVector], items: List[CandidateItem], item_embeddings: List[Optional[EmbeddingVector]]) -> None:
        if not query_embedding or not NUMPY_AVAILABLE:
            for item in items:
                item.scores["semantic"] = 0.0
                return
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            for item in items:
                item.scores["semantic"] = 0.0
                return
        for i, item in enumerate(items):
            item_vec_list = item_embeddings[i]
            if not item_vec_list:
                item.scores["semantic"] = 0.0
                continue
            item_vec_np = np.array(item_vec_list, dtype=np.float32)
            item_norm = np.linalg.norm(item_vec_np)
            if item_norm == 0:
                item.scores["semantic"] = 0.0
                continue
            similarity = np.dot(q_vec, item_vec_np) / (q_norm * item_norm)
            item.scores["semantic"] = max(0.0, min(1.0, (similarity + 1) / 2))
    def _calculate_bm25_scores(self, query: str, items: List[CandidateItem]) -> None:
        if not RANK_BM25_AVAILABLE or not items:
            for item in items:
                item.scores["bm25"] = 0.0
                return
        tokenized_corpus = [re.findall(r"\w+", item.text_for_scoring.lower()) for item in items]
        tokenized_query = re.findall(r"\w+", query.lower())
        if not tokenized_query or not any(tokenized_corpus):
             for item in items:
                item.scores["bm25"] = 0.0
             return
        try:
            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(tokenized_query)
            max_score = max(doc_scores) if any(s > 0 for s in doc_scores) else 1.0
            min_score = min(doc_scores)
            for i, item in enumerate(items):
                raw_score = doc_scores[i]
                if max_score == min_score:
                    normalized_score = 0.5 if max_score != 0 else 0.0
                elif max_score > min_score:
                    normalized_score = (raw_score - min_score) / (max_score - min_score)
                else:
                    normalized_score = 0.0
                item.scores["bm25"] = max(0.0, min(1.0, normalized_score))
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error calculating BM25 scores: {e}", exc_info=True)
            for item in items:
                item.scores["bm25"] = 0.0
