"""Query expansion module for improving retrieval accuracy."""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from datetime import datetime

import nltk
from nltk.corpus import wordnet
from llama_index.llms.openai import OpenAI
from llama_index.core.llms.llm import LLM

from src.core.config import Settings
from src.core.exceptions import QueryExpansionError, LLMError

logger = logging.getLogger(__name__)

# Download NLTK data if not present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class ExpansionMethod(str, Enum):
    """Available query expansion methods."""

    LLM = "llm"
    SYNONYM = "synonym"
    HYDE = "hyde"


class SynonymExpander:
    """Expands queries using synonyms from WordNet."""

    def __init__(self):
        """Initialize synonym expander."""
        self.pos_tags_map = {
            'NN': wordnet.NOUN,
            'NNS': wordnet.NOUN,
            'NNP': wordnet.NOUN,
            'NNPS': wordnet.NOUN,
            'VB': wordnet.VERB,
            'VBD': wordnet.VERB,
            'VBG': wordnet.VERB,
            'VBN': wordnet.VERB,
            'VBP': wordnet.VERB,
            'VBZ': wordnet.VERB,
            'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ,
            'JJS': wordnet.ADJ,
            'RB': wordnet.ADV,
            'RBR': wordnet.ADV,
            'RBS': wordnet.ADV,
        }

    def expand_query(
        self,
        query: str,
        max_synonyms_per_word: int = 2,
        min_word_length: int = 4,
    ) -> List[str]:
        """
        Expand query using synonyms.

        Args:
            query: Original query
            max_synonyms_per_word: Maximum synonyms per word
            min_word_length: Minimum word length to consider

        Returns:
            List of expanded queries
        """
        try:
            expanded_queries = []

            # Tokenize and get POS tags
            tokens = nltk.word_tokenize(query.lower())
            pos_tags = nltk.pos_tag(tokens)

            # Get synonyms for each word
            synonym_sets = []
            for word, pos in pos_tags:
                if len(word) >= min_word_length and word.isalpha():
                    synonyms = self._get_synonyms(word, pos, max_synonyms_per_word)
                    synonym_sets.append([word] + synonyms)
                else:
                    synonym_sets.append([word])

            # Generate combinations (limit to prevent explosion)
            expanded_queries = self._generate_combinations(
                synonym_sets,
                max_combinations=5,
            )

            # Remove duplicates and original query
            expanded_queries = list(set(expanded_queries))
            if query.lower() in expanded_queries:
                expanded_queries.remove(query.lower())

            logger.debug(f"Synonym expansion generated {len(expanded_queries)} variants")
            return expanded_queries[:3]  # Limit to top 3

        except Exception as e:
            logger.error(f"Synonym expansion failed: {e}")
            return []

    def _get_synonyms(self, word: str, pos: str, max_synonyms: int) -> List[str]:
        """Get synonyms for a word using WordNet."""
        try:
            wordnet_pos = self.pos_tags_map.get(pos)
            if not wordnet_pos:
                return []

            synonyms = set()
            for synset in wordnet.synsets(word, pos=wordnet_pos):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym.lower())

            return list(synonyms)[:max_synonyms]

        except Exception:
            return []

    def _generate_combinations(
        self,
        synonym_sets: List[List[str]],
        max_combinations: int = 5,
    ) -> List[str]:
        """Generate combinations of synonyms."""
        import itertools

        # Limit combinations to prevent explosion
        if len(synonym_sets) > 6:
            synonym_sets = synonym_sets[:6]

        combinations = []
        for combo in itertools.product(*synonym_sets):
            if len(combinations) >= max_combinations:
                break
            combinations.append(' '.join(combo))

        return combinations


class LLMExpander:
    """Expands queries using Large Language Models."""

    def __init__(self, llm: LLM, settings: Settings):
        """
        Initialize LLM expander.

        Args:
            llm: Language model instance
            settings: Application settings
        """
        self.llm = llm
        self.settings = settings

    async def expand_query(
        self,
        query: str,
        expansion_type: str = "related_queries",
        max_expansions: int = 3,
        context: Optional[str] = None,
    ) -> List[str]:
        """
        Expand query using LLM.

        Args:
            query: Original query
            expansion_type: Type of expansion (related_queries, reformulations, etc.)
            max_expansions: Maximum number of expansions
            context: Additional context for expansion

        Returns:
            List of expanded queries
        """
        try:
            prompt = self._build_expansion_prompt(
                query,
                expansion_type,
                max_expansions,
                context,
            )

            response = await self.llm.acomplete(prompt)
            expanded_queries = self._parse_llm_response(response.text)

            logger.debug(f"LLM expansion generated {len(expanded_queries)} variants")
            return expanded_queries[:max_expansions]

        except Exception as e:
            logger.error(f"LLM expansion failed: {e}")
            raise QueryExpansionError(f"LLM expansion failed: {e}", original_query=query)

    def _build_expansion_prompt(
        self,
        query: str,
        expansion_type: str,
        max_expansions: int,
        context: Optional[str] = None,
    ) -> str:
        """Build prompt for LLM query expansion."""
        base_prompt = f"""
You are an expert at expanding search queries to improve information retrieval.

Original Query: "{query}"
"""

        if context:
            base_prompt += f"\nContext: {context}\n"

        if expansion_type == "related_queries":
            base_prompt += f"""
Generate {max_expansions} related queries that would help find similar or complementary information.
Focus on:
- Different ways to phrase the same question
- Related concepts and topics
- More specific or general versions of the query

Format your response as a numbered list:
1. [expanded query 1]
2. [expanded query 2]
3. [expanded query 3]
"""

        elif expansion_type == "reformulations":
            base_prompt += f"""
Generate {max_expansions} reformulations of the query using:
- Different terminology and vocabulary
- Alternative question structures
- Synonyms and related terms

Format your response as a numbered list:
1. [reformulated query 1]
2. [reformulated query 2]
3. [reformulated query 3]
"""

        elif expansion_type == "hyde":
            base_prompt += f"""
Generate {max_expansions} hypothetical document excerpts that would directly answer this query.
Write as if these are actual passages from relevant documents.

Format your response as:
1. [hypothetical document excerpt 1]
2. [hypothetical document excerpt 2]
3. [hypothetical document excerpt 3]
"""

        return base_prompt

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response to extract expanded queries."""
        try:
            # Extract numbered list items
            pattern = r'^\d+\.\s*(.+)$'
            lines = response.strip().split('\n')

            expanded_queries = []
            for line in lines:
                match = re.match(pattern, line.strip())
                if match:
                    query = match.group(1).strip()
                    # Remove quotes if present
                    query = query.strip('"\'')
                    if query:
                        expanded_queries.append(query)

            return expanded_queries

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []


class HyDEExpander:
    """Hypothetical Document Embeddings (HyDE) expansion."""

    def __init__(self, llm: LLM):
        """
        Initialize HyDE expander.

        Args:
            llm: Language model instance
        """
        self.llm = llm

    async def generate_hypothetical_documents(
        self,
        query: str,
        num_documents: int = 3,
        document_length: str = "medium",
    ) -> List[str]:
        """
        Generate hypothetical documents that would answer the query.

        Args:
            query: Original query
            num_documents: Number of hypothetical documents to generate
            document_length: Length of documents (short, medium, long)

        Returns:
            List of hypothetical document texts
        """
        try:
            length_instructions = {
                "short": "1-2 sentences",
                "medium": "3-5 sentences",
                "long": "1-2 paragraphs",
            }

            prompt = f"""
Generate {num_documents} hypothetical document excerpts that would directly and comprehensively answer this query: "{query}"

Each excerpt should be {length_instructions.get(document_length, "3-5 sentences")} long and written as if it's from an authoritative source document.

Format your response as:
Document 1:
[hypothetical document text]

Document 2:
[hypothetical document text]

Document 3:
[hypothetical document text]
"""

            response = await self.llm.acomplete(prompt)
            documents = self._parse_hyde_response(response.text)

            logger.debug(f"HyDE generated {len(documents)} hypothetical documents")
            return documents

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            raise QueryExpansionError(f"HyDE generation failed: {e}", original_query=query)

    def _parse_hyde_response(self, response: str) -> List[str]:
        """Parse HyDE response to extract hypothetical documents."""
        try:
            documents = []
            current_doc = ""

            lines = response.strip().split('\n')
            in_document = False

            for line in lines:
                line = line.strip()

                if re.match(r'^Document \d+:', line):
                    if current_doc.strip():
                        documents.append(current_doc.strip())
                    current_doc = ""
                    in_document = True
                elif in_document and line:
                    current_doc += line + " "
                elif not line and current_doc:
                    # Empty line, end current document
                    documents.append(current_doc.strip())
                    current_doc = ""
                    in_document = False

            # Add final document if exists
            if current_doc.strip():
                documents.append(current_doc.strip())

            return documents

        except Exception as e:
            logger.error(f"Failed to parse HyDE response: {e}")
            return []


class QueryExpansionService:
    """Main service for query expansion."""

    def __init__(self, settings: Settings, llm: Optional[LLM] = None):
        """
        Initialize query expansion service.

        Args:
            settings: Application settings
            llm: Language model instance
        """
        self.settings = settings

        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            try:
                self.llm = OpenAI(
                    api_key=settings.llm.openai_api_key,
                    model=settings.llm.openai_model,
                    temperature=0.1,
                )
            except Exception as e:
                if not settings.app.mock_llm_responses:
                    raise QueryExpansionError(f"Failed to initialize LLM: {e}")
                logger.warning("Using mock LLM responses for development")
                self.llm = None

        # Initialize expanders
        self.synonym_expander = SynonymExpander()
        if self.llm:
            self.llm_expander = LLMExpander(self.llm, settings)
            self.hyde_expander = HyDEExpander(self.llm)
        else:
            self.llm_expander = None
            self.hyde_expander = None

    async def expand_query(
        self,
        query: str,
        methods: Optional[List[ExpansionMethod]] = None,
        max_expansions: int = 3,
        context: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Expand query using specified methods.

        Args:
            query: Original query
            methods: Expansion methods to use
            max_expansions: Maximum expansions per method
            context: Additional context

        Returns:
            Dictionary of expansion results by method
        """
        try:
            if not self.settings.rag.enable_query_expansion:
                return {"original": [query]}

            # Use configured methods if not specified
            if methods is None:
                methods = [
                    ExpansionMethod(method)
                    for method in self.settings.rag.query_expansion_methods
                ]

            results = {"original": [query]}

            # Apply each expansion method
            expansion_tasks = []

            for method in methods:
                if method == ExpansionMethod.SYNONYM:
                    task = self._expand_with_synonyms(query, max_expansions)
                elif method == ExpansionMethod.LLM and self.llm_expander:
                    task = self._expand_with_llm(query, max_expansions, context)
                elif method == ExpansionMethod.HYDE and self.hyde_expander:
                    task = self._expand_with_hyde(query, max_expansions)
                elif method == ExpansionMethod.MULTI_QUERY and self.llm_expander:
                    task = self._expand_with_multi_query(query, max_expansions, context)
                else:
                    logger.warning(f"Expansion method {method} not available or not implemented")
                    continue

                expansion_tasks.append((method, task))

            # Execute expansion tasks
            for method, task in expansion_tasks:
                try:
                    if asyncio.iscoroutine(task):
                        expansions = await task
                    else:
                        expansions = task

                    if expansions:
                        results[method.value] = expansions

                except Exception as e:
                    logger.error(f"Expansion method {method} failed: {e}")
                    results[method.value] = []

            # Limit total expansions
            total_expansions = []
            for method_results in results.values():
                total_expansions.extend(method_results)

            max_total = self.settings.rag.max_expanded_queries
            if len(total_expansions) > max_total:
                # Keep original and best expansions
                results["truncated"] = True

            logger.info(f"Query expansion generated {len(total_expansions)} total expansions")
            return results

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            raise QueryExpansionError(f"Query expansion failed: {e}", original_query=query)

    def _expand_with_synonyms(self, query: str, max_expansions: int) -> List[str]:
        """Expand query with synonyms."""
        return self.synonym_expander.expand_query(
            query,
            max_synonyms_per_word=2,
        )[:max_expansions]

    async def _expand_with_llm(
        self,
        query: str,
        max_expansions: int,
        context: Optional[str],
    ) -> List[str]:
        """Expand query with LLM."""
        if not self.llm_expander:
            return []

        return await self.llm_expander.expand_query(
            query,
            expansion_type="related_queries",
            max_expansions=max_expansions,
            context=context,
        )

    async def _expand_with_hyde(self, query: str, max_expansions: int) -> List[str]:
        """Expand query with HyDE."""
        if not self.hyde_expander:
            return []

        documents = await self.hyde_expander.generate_hypothetical_documents(
            query,
            num_documents=max_expansions,
            document_length="medium",
        )
        return documents

    async def _expand_with_multi_query(
        self,
        query: str,
        max_expansions: int,
        context: Optional[str],
    ) -> List[str]:
        """Expand query with multiple reformulations."""
        if not self.llm_expander:
            return []

        return await self.llm_expander.expand_query(
            query,
            expansion_type="reformulations",
            max_expansions=max_expansions,
            context=context,
        )

    def get_expansion_stats(self) -> Dict[str, Any]:
        """Get query expansion statistics."""
        return {
            "enabled": self.settings.rag.enable_query_expansion,
            "available_methods": [
                method for method in self.settings.rag.query_expansion_methods
            ],
            "max_expansions": self.settings.rag.max_expanded_queries,
            "llm_available": self.llm is not None,
            "synonym_expander_available": True,
            "hyde_available": self.hyde_expander is not None,
        }