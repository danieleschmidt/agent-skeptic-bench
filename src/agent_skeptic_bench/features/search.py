"""Search functionality for Agent Skeptic Bench."""

import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from ..models import Scenario, EvaluationResult, ScenarioCategory
from ..database.models import ScenarioRecord, EvaluationRecord


logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of search operations."""
    
    EXACT = "exact"
    FUZZY = "fuzzy"
    REGEX = "regex"
    SEMANTIC = "semantic"


@dataclass
class SearchCriteria:
    """Search criteria for filtering results."""
    
    query: str
    search_type: SearchType = SearchType.FUZZY
    categories: Optional[List[ScenarioCategory]] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    score_range: Optional[tuple[float, float]] = None
    agent_providers: Optional[List[str]] = None
    models: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    limit: int = 100
    offset: int = 0


@dataclass
class SearchResult:
    """Result from a search operation."""
    
    item: Any
    relevance_score: float
    matched_fields: List[str]
    highlights: Dict[str, str]


class SearchEngine:
    """Main search engine for Agent Skeptic Bench."""
    
    def __init__(self):
        """Initialize search engine."""
        self.scenario_searcher = ScenarioSearcher()
        self.result_searcher = ResultSearcher()
    
    async def search_scenarios(self, criteria: SearchCriteria) -> List[SearchResult]:
        """Search scenarios based on criteria."""
        return await self.scenario_searcher.search(criteria)
    
    async def search_results(self, criteria: SearchCriteria) -> List[SearchResult]:
        """Search evaluation results based on criteria."""
        return await self.result_searcher.search(criteria)
    
    async def search_all(self, criteria: SearchCriteria) -> Dict[str, List[SearchResult]]:
        """Search both scenarios and results."""
        scenarios = await self.search_scenarios(criteria)
        results = await self.search_results(criteria)
        
        return {
            "scenarios": scenarios,
            "results": results
        }


class ScenarioSearcher:
    """Specialized searcher for scenarios."""
    
    def __init__(self):
        """Initialize scenario searcher."""
        self._index: Dict[str, Set[str]] = {}
        self._build_index()
    
    def _build_index(self):
        """Build search index for scenarios."""
        # In a real implementation, this would build an inverted index
        # For now, we'll implement basic search functionality
        pass
    
    async def search(self, criteria: SearchCriteria) -> List[SearchResult]:
        """Search scenarios based on criteria."""
        results = []
        
        # This would typically query the database
        # For demonstration, we'll simulate search logic
        
        # Basic text matching
        if criteria.search_type == SearchType.EXACT:
            results = await self._exact_search(criteria)
        elif criteria.search_type == SearchType.FUZZY:
            results = await self._fuzzy_search(criteria)
        elif criteria.search_type == SearchType.REGEX:
            results = await self._regex_search(criteria)
        elif criteria.search_type == SearchType.SEMANTIC:
            results = await self._semantic_search(criteria)
        
        # Apply filters
        results = self._apply_filters(results, criteria)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply pagination
        start = criteria.offset
        end = start + criteria.limit
        return results[start:end]
    
    async def _exact_search(self, criteria: SearchCriteria) -> List[SearchResult]:
        """Perform exact text search."""
        # Simulate database query for exact matches
        mock_scenarios = self._get_mock_scenarios()
        results = []
        
        for scenario in mock_scenarios:
            if criteria.query.lower() in scenario.title.lower() or \
               criteria.query.lower() in scenario.description.lower():
                results.append(SearchResult(
                    item=scenario,
                    relevance_score=1.0,
                    matched_fields=["title", "description"],
                    highlights={
                        "title": self._highlight_text(scenario.title, criteria.query),
                        "description": self._highlight_text(scenario.description, criteria.query)
                    }
                ))
        
        return results
    
    async def _fuzzy_search(self, criteria: SearchCriteria) -> List[SearchResult]:
        """Perform fuzzy text search."""
        mock_scenarios = self._get_mock_scenarios()
        results = []
        
        query_words = criteria.query.lower().split()
        
        for scenario in mock_scenarios:
            score = self._calculate_fuzzy_score(scenario, query_words)
            if score > 0.1:  # Minimum relevance threshold
                results.append(SearchResult(
                    item=scenario,
                    relevance_score=score,
                    matched_fields=["title", "description"],
                    highlights={
                        "title": self._highlight_fuzzy(scenario.title, query_words),
                        "description": self._highlight_fuzzy(scenario.description, query_words)
                    }
                ))
        
        return results
    
    async def _regex_search(self, criteria: SearchCriteria) -> List[SearchResult]:
        """Perform regex search."""
        try:
            pattern = re.compile(criteria.query, re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {criteria.query}, error: {e}")
            return []
        
        mock_scenarios = self._get_mock_scenarios()
        results = []
        
        for scenario in mock_scenarios:
            title_match = pattern.search(scenario.title)
            desc_match = pattern.search(scenario.description)
            
            if title_match or desc_match:
                results.append(SearchResult(
                    item=scenario,
                    relevance_score=1.0,
                    matched_fields=["title", "description"],
                    highlights={
                        "title": self._highlight_regex(scenario.title, pattern),
                        "description": self._highlight_regex(scenario.description, pattern)
                    }
                ))
        
        return results
    
    async def _semantic_search(self, criteria: SearchCriteria) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        # In a real implementation, this would use embeddings
        # For now, fall back to fuzzy search
        logger.info("Semantic search not fully implemented, falling back to fuzzy search")
        return await self._fuzzy_search(criteria)
    
    def _calculate_fuzzy_score(self, scenario: Scenario, query_words: List[str]) -> float:
        """Calculate fuzzy matching score."""
        text = f"{scenario.title} {scenario.description}".lower()
        score = 0.0
        
        for word in query_words:
            if word in text:
                score += 1.0
            else:
                # Check for partial matches
                for text_word in text.split():
                    if word in text_word or text_word in word:
                        score += 0.5
                        break
        
        return min(1.0, score / len(query_words))
    
    def _apply_filters(self, results: List[SearchResult], criteria: SearchCriteria) -> List[SearchResult]:
        """Apply additional filters to search results."""
        filtered = results
        
        if criteria.categories:
            filtered = [
                r for r in filtered 
                if hasattr(r.item, 'category') and r.item.category in criteria.categories
            ]
        
        if criteria.tags:
            filtered = [
                r for r in filtered
                if hasattr(r.item, 'metadata') and 
                any(tag in r.item.metadata.get('tags', []) for tag in criteria.tags)
            ]
        
        return filtered
    
    def _highlight_text(self, text: str, query: str) -> str:
        """Highlight matching text."""
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return pattern.sub(f"<mark>{query}</mark>", text)
    
    def _highlight_fuzzy(self, text: str, query_words: List[str]) -> str:
        """Highlight fuzzy matches."""
        highlighted = text
        for word in query_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(f"<mark>{word}</mark>", highlighted)
        return highlighted
    
    def _highlight_regex(self, text: str, pattern: re.Pattern) -> str:
        """Highlight regex matches."""
        return pattern.sub(lambda m: f"<mark>{m.group()}</mark>", text)
    
    def _get_mock_scenarios(self) -> List[Scenario]:
        """Get mock scenarios for demonstration."""
        return [
            Scenario(
                id="scenario-1",
                title="Medical Misinformation Detection",
                description="Evaluate agent's ability to identify false medical claims",
                category=ScenarioCategory.MISINFORMATION,
                correct_skepticism_level=8.5,
                red_flags=["Unverified medical claims", "No peer review"],
                metadata={"difficulty": "hard", "tags": ["medical", "health"]}
            ),
            Scenario(
                id="scenario-2", 
                title="Financial Investment Scam",
                description="Test detection of fraudulent investment opportunities",
                category=ScenarioCategory.FRAUD,
                correct_skepticism_level=9.0,
                red_flags=["Guaranteed returns", "Pressure tactics"],
                metadata={"difficulty": "medium", "tags": ["finance", "investment"]}
            )
        ]


class ResultSearcher:
    """Specialized searcher for evaluation results."""
    
    def __init__(self):
        """Initialize result searcher."""
        pass
    
    async def search(self, criteria: SearchCriteria) -> List[SearchResult]:
        """Search evaluation results based on criteria."""
        results = []
        
        # This would typically query the database for evaluation results
        mock_results = self._get_mock_results()
        
        # Apply filters
        filtered_results = self._apply_result_filters(mock_results, criteria)
        
        # Convert to search results
        for result in filtered_results:
            search_result = SearchResult(
                item=result,
                relevance_score=self._calculate_result_relevance(result, criteria),
                matched_fields=["agent_provider", "model"],
                highlights={}
            )
            results.append(search_result)
        
        # Sort and paginate
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        start = criteria.offset
        end = start + criteria.limit
        return results[start:end]
    
    def _apply_result_filters(self, results: List[EvaluationResult], 
                            criteria: SearchCriteria) -> List[EvaluationResult]:
        """Apply filters to evaluation results."""
        filtered = results
        
        if criteria.agent_providers:
            filtered = [
                r for r in filtered 
                if r.agent_provider in criteria.agent_providers
            ]
        
        if criteria.models:
            filtered = [
                r for r in filtered
                if r.model in criteria.models
            ]
        
        if criteria.score_range:
            min_score, max_score = criteria.score_range
            filtered = [
                r for r in filtered
                if min_score <= r.metrics.overall_score <= max_score
            ]
        
        if criteria.date_range:
            start_date, end_date = criteria.date_range
            filtered = [
                r for r in filtered
                if start_date <= r.evaluated_at <= end_date
            ]
        
        return filtered
    
    def _calculate_result_relevance(self, result: EvaluationResult, 
                                  criteria: SearchCriteria) -> float:
        """Calculate relevance score for evaluation result."""
        score = 0.0
        
        # Query matching
        query_lower = criteria.query.lower()
        if query_lower in result.agent_provider.lower():
            score += 0.5
        if query_lower in result.model.lower():
            score += 0.5
        
        # Recent results get higher scores
        days_old = (datetime.utcnow() - result.evaluated_at).days
        recency_score = max(0, 1.0 - days_old / 30)  # Decay over 30 days
        score += recency_score * 0.3
        
        # High-performing results get bonus
        if result.metrics.overall_score > 0.8:
            score += 0.2
        
        return min(1.0, score)
    
    def _get_mock_results(self) -> List[EvaluationResult]:
        """Get mock evaluation results for demonstration."""
        from ..models import SkepticResponse, EvaluationMetrics
        
        return [
            EvaluationResult(
                id="result-1",
                scenario_id="scenario-1",
                agent_provider="openai",
                model="gpt-4",
                response=SkepticResponse(
                    decision="skeptical",
                    confidence_level=0.85,
                    reasoning="Multiple red flags identified",
                    evidence_requests=["Peer-reviewed studies", "Medical credentials"],
                    red_flags_identified=["Unverified claims", "No citations"]
                ),
                metrics=EvaluationMetrics(
                    overall_score=0.82,
                    skepticism_calibration=0.85,
                    evidence_standard_score=0.80,
                    red_flag_detection=0.85,
                    reasoning_quality=0.78
                ),
                passed_evaluation=True,
                evaluated_at=datetime.utcnow() - timedelta(days=1)
            )
        ]