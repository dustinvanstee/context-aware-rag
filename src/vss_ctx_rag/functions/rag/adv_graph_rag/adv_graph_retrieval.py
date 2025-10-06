# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""adv_graph_retrieval.py: Contains retrieval utilities for Advanced Graph RAG"""

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.functions.rag.graph_rag.constants import (
    VECTOR_SEARCH_TOP_K,
    CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
)
from langchain_core.documents import Document
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import json
import re
from langchain_core.messages import HumanMessage


class AdvGraphRetrieval:
    def __init__(self, llm, graph: Neo4jGraphDB, top_k=None, max_retries=None):
        
        self.chat_llm = llm
        self.graph_db = graph
        self.top_k = top_k
        self.max_retries = max_retries if max_retries else 3
        self.vector_retriever = Neo4jVector.from_existing_index(
            embedding=self.graph_db.embeddings,
            index_name="vector",
            graph=self.graph_db.graph_db,
        ).as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.top_k or VECTOR_SEARCH_TOP_K,
                "score_threshold": CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
            },
        )
        self.doc_retriever = self.create_document_retriever_chain()


    def _format_relative_time(self, timestamp: float) -> str:
        """Format timestamp relative to current time in seconds"""
        if timestamp is None:
            return "unknown time"

        now = datetime.now(timezone.utc).timestamp()
        diff = now - timestamp
        return f"{int(diff)} seconds ago"

    def create_document_retriever_chain(self):
        with TimeMeasure(
            "context_manager/adv_graph_retrieval/create_document_retriever_chain",
            "green",
        ):
            graph = self.graph_db.graph_db

            # Time-aware retrievers
            retrieval_query_all_with_stream_id = """
WITH node, score
CALL {
    WITH node
    MATCH (node)-[:PART_OF]->(d:Document)
    RETURN d.stream_id AS stream_id
    LIMIT 1
}
RETURN node.text AS text,
       score,
       node {.*, embedding: Null} AS metadata,
       stream_id
"""

            retriever_all = Neo4jVector.from_existing_index(
                embedding=self.graph_db.embeddings,
                index_name="vector",
                retrieval_query=retrieval_query_all_with_stream_id,
                graph=graph,
            ).as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.top_k or VECTOR_SEARCH_TOP_K,
                    "score_threshold": CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
                },
            )

            return retriever_all

    async def get_all_entity_types(self) -> List[str]:
        """Query Neo4j to get all unique entity types"""
        try:
            query = """
            MATCH (n)
            WHERE NOT n:Document AND NOT n:Chunk
            RETURN DISTINCT labels(n) AS labels
            """
            result = await self.graph_db.arun_cypher_query(query)
            entity_types = set()
            for record in result:
                # Each record has labels as a list
                for label_list in record.values():
                    if isinstance(label_list, list):
                        entity_types.update(label_list)
            return list(entity_types)
        except Exception as e:
            logger.error(f"Error getting entity types: {e}")
            return []

    async def get_all_stream_ids(self) -> List[str]:
        """Query Neo4j to get all unique stream IDs"""
        try:
            query = """
            MATCH (d:Document)
            RETURN DISTINCT d.stream_id AS stream_id
            """
            result = await self.graph_db.arun_cypher_query(query)
            stream_ids = [record["stream_id"] for record in result if "stream_id" in record]
            return stream_ids
        except Exception as e:
            logger.error(f"Error getting stream IDs: {e}")
            return []

    async def retrieve_by_relationship(
        self, start_type: str, rel_type: str, end_type: str
    ) -> List[Dict]:
        """Retrieve records based on relationships between entities"""
        query = f"""
        MATCH (s:{start_type})-[r:{rel_type}]->(e:{end_type})
        MATCH (s)<-[:MENTIONS]-(c:Chunk)
        RETURN c AS n
        LIMIT 5
        """
        logger.info(f"Relationship query: {query}")
        result = await self.graph_db.arun_cypher_query(query)
        return result

    async def retrieve_temporal_context(
        self,
        start_time: float = None,
        end_time: float = None,
        stream_ids: List[str] = None,
    ) -> List[Dict]:
        """Retrieve records within a specific time range"""
        try:
            # Build time filter
            filters = []
            if start_time is not None:
                filters.append(f"c.start_time >= {start_time}")
            if end_time is not None:
                filters.append(f"c.start_time <= {end_time}")

            # Build stream ID filter
            if stream_ids:
                stream_filter = "d.stream_id IN [" + ", ".join([f"'{sid}'" for sid in stream_ids]) + "]"
                filters.append(stream_filter)

            # Combine filters
            where_clause = " AND ".join(filters) if filters else "TRUE"

            query = f"""
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)
            WHERE {where_clause}
            RETURN c AS n
            ORDER BY c.start_time DESC
            LIMIT {self.top_k if self.top_k else VECTOR_SEARCH_TOP_K}
            """

            logger.info(f"Temporal query: {query}")

            result = await self.graph_db.arun_cypher_query(query)
            logger.info(f"Retrieved {len(result)} temporal records")
            return result

        except Exception as e:
            logger.error(f"Error during temporal retrieval: {e}")
            return []

    async def retrieve_semantic_context(
        self,
        question: str,
        start_time: float = None,
        end_time: float = None,
        sort_by: str = None,
        stream_ids: List[str] = None,
    ) -> List[Dict]:
        """Semantic similarity search with optional time filtering"""
        try:
            logger.info(
                f"Performing semantic search for: {question} "
                f"[start_time={start_time}, end_time={end_time}, sort={sort_by}]"
            )

            # Use Langchain retriever with semantic search
            result_docs = await self.doc_retriever.ainvoke(question)

            # Apply time filters if needed
            if start_time or end_time:
                filtered_docs = []
                for doc in result_docs:
                    doc_start_time = doc.metadata.get("start_time")
                    if doc_start_time:
                        if start_time and doc_start_time < start_time:
                            continue
                        if end_time and doc_start_time > end_time:
                            continue
                    filtered_docs.append(doc)
                result_docs = filtered_docs
                logger.info(
                    f"Filtered to {len(result_docs)} docs within time range"
                )

            # Apply stream ID filters
            if stream_ids:
                filtered_docs = []
                for doc in result_docs:
                    doc_stream_id = doc.metadata.get("stream_id", "")
                    if doc_stream_id in stream_ids:
                        filtered_docs.append(doc)
                result_docs = filtered_docs
                logger.info(
                    f"Filtered to {len(result_docs)} docs matching stream IDs"
                )

            # Sort if needed
            if sort_by == "time" and result_docs:
                result_docs.sort(
                    key=lambda x: x.metadata.get("start_time", 0), reverse=True
                )
                logger.info("Sorted results by time")

            # Convert to expected format
            neo4j_format = []
            for doc in result_docs:
                neo4j_format.append(
                    {
                        "n": {
                            "text": doc.page_content,
                            "start_time": doc.metadata.get("start_time"),
                            "end_time": doc.metadata.get("end_time"),
                            "stream_id": doc.metadata.get("stream_id"),
                        }
                    }
                )

            logger.info(f"Returning {len(neo4j_format)} semantic results")
            return neo4j_format

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """Use LLM to analyze question and determine basic retrieval elements"""
        logger.info(f"Analyzing question: {question}")
        prompt = f"""Analyze this question and identify key elements for graph database retrieval.
        Question: {question}

        Identify and return as JSON:
        1. Entity types mentioned. Available entity types: {await self.get_all_entity_types()}
        2. Relationships of interest
        3. Location references
        4. Stream IDs mentioned. Available stream_ids: {await self.get_all_stream_ids()}
        5. Retrieval strategy (similarity, temporal)
            a. similarity: If the question needs to find similar content, return the retrieval strategy as similarity
            b. temporal: If the question is about a specific time range or time-based filtering, return the strategy as temporal

        Example question: "Between 30 seconds and 5 minutes ago, has the dog found the ball?"
        Example response:
        {{\
            "entity_types": ["Dog", "Ball"],\
            "relationships": ["DROPPED", "PICKED_UP"],\
            "location_references": ["backyard"],\
            "stream_ids": [],\
            "retrieval_strategy": "temporal"\
        }}\

        Example question with stream: "Summarize channel 3 over the last 5 minutes."
        Example response:
        {{\
            "entity_types": [],\
            "relationships": [],\
            "location_references": [],\
            "stream_ids": ["fm-radio-ch3"],\
            "retrieval_strategy": "temporal"\
        }}\

        Example question without time filtering: "What topics were discussed about dogs?"
        Example response:
        {{\
            "entity_types": ["Dog"],\
            "relationships": [],\
            "location_references": [],\
            "stream_ids": [],\
            "retrieval_strategy": "similarity"\
        }}\

        Output only valid JSON. Do not include any other text.
        """

        response = await self.chat_llm.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def analyze_temporal_strategy(self, question: str) -> str:
        """Determine if question needs temporal retrieval and strategy"""
        prompt = f"""Analyze this question to determine what type of temporal retrieval is needed.
        Question: {question}

        Determine the temporal strategy:
        - "none": No time filtering needed
        - "relative": Relative time like "in the last hour", "5 minutes ago"
        - "absolute": Absolute time like "between 2pm and 3pm", "on January 15th"
        - "both": Both relative and absolute references

        Return JSON with just the strategy:
        {{"temporal_strategy": "none"|"relative"|"absolute"|"both"}}

        Examples:
        "What topics were discussed?" -> {{"temporal_strategy": "none"}}
        "What happened in the last hour?" -> {{"temporal_strategy": "relative"}}
        "What happened between 2pm and 3pm?" -> {{"temporal_strategy": "absolute"}}

        Output only valid JSON.
        """

        response = await self.chat_llm.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def analyze_temporal_times(self, question: str, temporal_strategy: str) -> str:
        """Extract specific temporal information from question"""
        strategy_guidance = {
            "relative": "Extract relative time expressions (e.g., 'last 5 minutes', '30 seconds ago'). Calculate based on current time.",
            "absolute": "Extract absolute time expressions (e.g., '2pm', 'January 15th'). Return actual timestamps.",
            "both": "Extract both relative and absolute time expressions.",
        }

        guidance = strategy_guidance.get(temporal_strategy, "Extract any time references.")

        prompt = f"""Analyze this question to extract temporal information.
        Question: {question}
        Temporal Strategy: {temporal_strategy}

        {guidance}

        Return JSON with time bounds:
        {{\
            "start_relative_seconds": <seconds ago for start time>,\
            "end_relative_seconds": <seconds ago for end time>,\
            "start_absolute": "<ISO timestamp if absolute>",\
            "end_absolute": "<ISO timestamp if absolute>"\
        }}

        Examples:
        "in the last 5 minutes" -> {{"start_relative_seconds": 300, "end_relative_seconds": 0}}
        "between 30 seconds and 5 minutes ago" -> {{"start_relative_seconds": 300, "end_relative_seconds": 30}}
        "after 2pm today" -> {{"start_absolute": "2024-01-15T14:00:00Z"}}

        Output only valid JSON.
        """

        response = await self.chat_llm.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _convert_temporal_times_to_timestamps(
        self, temporal_times: Dict, temporal_strategy: str
    ) -> Dict[str, float]:
        """Convert temporal time expressions to Unix timestamps"""
        now = datetime.now(timezone.utc).timestamp()
        result = {}

        if temporal_strategy in ["relative", "both"]:
            start_rel = temporal_times.get("start_relative_seconds")
            end_rel = temporal_times.get("end_relative_seconds")

            if start_rel is not None:
                result["start_time"] = now - start_rel
            if end_rel is not None:
                result["end_time"] = now - end_rel

        if temporal_strategy in ["absolute", "both"]:
            start_abs = temporal_times.get("start_absolute")
            end_abs = temporal_times.get("end_absolute")

            if start_abs:
                try:
                    dt = datetime.fromisoformat(start_abs.replace("Z", "+00:00"))
                    result["start_time"] = dt.timestamp()
                except Exception as e:
                    logger.error(f"Error parsing absolute start time: {e}")

            if end_abs:
                try:
                    dt = datetime.fromisoformat(end_abs.replace("Z", "+00:00"))
                    result["end_time"] = dt.timestamp()
                except Exception as e:
                    logger.error(f"Error parsing absolute end time: {e}")

        return result

    async def retrieve_relevant_context(self, question: str) -> tuple:
        """Main retrieval method that orchestrates different retrieval strategies
        
        Returns:
            tuple: (documents, retrieval_metadata)
                documents: List of Document objects
                retrieval_metadata: Dict with analysis, strategy, temporal_range info
        """
        with TimeMeasure(
            "context_manager/adv_graph_retrieval/retrieve_relevant_context", "blue"
        ):
            logger.info(f"Starting context retrieval for question: {question}")

            # Step 1: Basic question analysis
            analysis = await self._parse_json_with_retries(self.analyze_question, "basic analysis", question)

            if not analysis:
                logger.error("Failed to parse basic analysis, using defaults")
                analysis = {
                    "entity_types": [],
                    "relationships": [],
                    "location_references": [],
                    "stream_ids": [],
                    "retrieval_strategy": "similarity",
                }

            # Get basic parameters from analysis
            strategy = analysis.get("retrieval_strategy", "")
            stream_ids = analysis.get("stream_ids", [])
            logger.info(f"Using retrieval strategy: {strategy}")

            # Step 2 & 3: Temporal analysis
            start_time = None
            end_time = None
            temporal_strategy = "none"

            # Step 2: Determine temporal strategy type
            temporal_strategy_analysis = await self._parse_json_with_retries(self.analyze_temporal_strategy, "temporal strategy", question)

            if temporal_strategy_analysis:
                temporal_strategy = temporal_strategy_analysis.get("temporal_strategy", "none")
                logger.info(f"Using temporal strategy: {temporal_strategy}")

                # Step 3: Determine specific times if not "none"
                if temporal_strategy != "none":
                    temporal_times = await self._parse_json_with_retries(self.analyze_temporal_times, "temporal times", question, temporal_strategy)

                    if temporal_times:
                        # Convert to actual timestamps
                        timestamps = self._convert_temporal_times_to_timestamps(temporal_times, temporal_strategy)
                        start_time = timestamps.get("start_time")
                        end_time = timestamps.get("end_time")
                        logger.info(f"Temporal range: {start_time} to {end_time}")

            # Collect context from retrieval strategies
            contexts = []

            if strategy == "temporal":
                temporal_data = await self.retrieve_temporal_context(
                    start_time, end_time, stream_ids
                )
                logger.info(f"Temporal Contexts...")
                if temporal_data:
                    contexts.extend(temporal_data)
                    logger.info(f"Retrieved {len(temporal_data)} temporal records")
                else:
                    logger.info("No temporal data found in that time range")
                    # Return empty with metadata
                    retrieval_metadata = {
                        "retrieval_strategy": strategy,
                        "temporal_strategy": temporal_strategy,
                        "temporal_range": {
                            "start_time": start_time,
                            "end_time": end_time
                        },
                        "stream_ids": stream_ids,
                        "entity_types": analysis.get("entity_types", [])
                    }
                    return None, retrieval_metadata
            else:  # semantic retrieval
                # Semantic similarity retrieval
                semantic_data = await self.retrieve_semantic_context(
                    question,
                    start_time=start_time,
                    end_time=end_time,
                    sort_by=analysis.get("sort_by", "score"),
                    stream_ids=stream_ids,
                )
                logger.info(f"Semantic Contexts...")
                if semantic_data:
                    contexts.extend(semantic_data)

            logger.info(f"Contexts: {contexts}")

            # Relationship-based retrieval
            relationships = analysis.get("relationships", [])
            for rel in relationships:
                if isinstance(rel, str):
                    # If relationship is specified without types, skip
                    continue
                start_type = rel.get("from")
                end_type = rel.get("to")
                rel_type = rel.get("type")
                if all([start_type, end_type, rel_type]):
                    rel_data = await self.retrieve_by_relationship(
                        start_type, rel_type, end_type
                    )
                    logger.info(f"Relationship Data: {rel_data}")
                    if rel_data:
                        contexts.extend(rel_data)
                        logger.info(
                            f"Retrieved {len(rel_data)} records for "
                            f"relationship {rel_type}"
                        )

            # Convert to Documents
            documents = []
            for ctx in contexts:
                # Convert Neo4j results to Document format
                # Check if ctx has expected structure
                if isinstance(ctx, dict) and "n" in ctx:
                    if "text" in ctx["n"]:
                        doc = Document(
                            page_content=str(ctx["n"].get("text", "")),
                            metadata={
                                "start_time": ctx.get("n", {}).get("start_time", ""),
                                "end_time": ctx.get("n", {}).get("end_time", ""),
                                "stream_id": ctx.get("n", {}).get("stream_id", ""),
                            },
                        )
                        documents.append(doc)

            # Build retrieval metadata
            retrieval_metadata = {
                "retrieval_strategy": strategy,
                "temporal_strategy": temporal_strategy,
                "temporal_range": {
                    "start_time": start_time,
                    "end_time": end_time
                },
                "stream_ids": stream_ids,
                "entity_types": analysis.get("entity_types", []),
                "num_documents_retrieved": len(documents)
            }

            logger.info(f"Returning {len(documents)} documents with metadata: {retrieval_metadata}")
            return documents, retrieval_metadata

    async def _parse_json_with_retries(self, analysis_func, analysis_type: str, *args, **kwargs) -> Dict:
        """Helper method to retry analysis function calls and parse JSON responses"""
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                # Call the analysis function
                response = await analysis_func(*args, **kwargs)

                # Parse JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                logger.info(f"{analysis_type} response (attempt {retry_count + 1}): {response}")

                if json_start >= 0 and json_end > json_start:
                    result = json.loads(response[json_start:json_end])
                    logger.info(f"Successfully parsed {analysis_type} JSON on attempt {retry_count + 1}")
                    return result
                else:
                    raise json.JSONDecodeError(f"No JSON found in {analysis_type} response", response, 0)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {analysis_type} JSON response (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.info(f"Retrying {analysis_type} analysis (attempt {retry_count + 1}/{self.max_retries})")
                else:
                    logger.error(f"Max retries ({self.max_retries}) reached for {analysis_type}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error in {analysis_type} analysis (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached for {analysis_type}")
                    return None

        return None