# app/system.py
import os
import json
from typing import Any, Optional

import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq


class LLMSystem:
    """Unified system for connecting to any LLM and processing support tickets."""
    
    def __init__(self, provider: str = "groq", model: str = "gpt-oss-120b"):
        """
        Initialize the LLM system.
        
        Args:
            provider: LLM provider ("openai", "anthropic", or "groq")
            model: Model name to use
        """
        self.provider = provider
        self.model = model
        self.client = self._init_client()
        
    def _init_client(self):
        """Initialize the appropriate LLM client based on provider."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            return OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            return Anthropic(api_key=api_key)
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set in environment")
            return Groq(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make a call to the configured LLM."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            return response.content[0].text
        elif self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        return ""
    
    # ========== Retrieval ==========
    def retriever(self, query: str, top_k: int = 5) -> list:
        """Retrieve relevant documents for a query."""
        # Use FAISS vector search if ingestion is available
        if hasattr(self, 'ingestion') and self.ingestion and self.ingestion.index is not None:
            try:
                results = self.ingestion.search(query, top_k)
                return [(r["metadata"]["path"], r["text"]) for r in results]
            except Exception as e:
                print(f"FAISS search failed, falling back to keyword search: {e}")
        
        # Fallback to keyword-based retrieval
        from pathlib import Path
        import faiss
        import numpy as np
        
        data_dir = Path("data")
        results = []
        
        # Search through all data directories
        for product_dir in data_dir.iterdir():
            if not product_dir.is_dir():
                continue
            index_file = product_dir / "index.md"
            if not index_file.exists():
                continue
                
            # Read index to find relevant docs
            index_content = index_file.read_text(encoding="utf-8")
            
            # Simple keyword-based retrieval
            query_lower = query.lower()
            if any(kw in index_content.lower() for kw in query_lower.split()):
                # Get all md files in this product area
                for md_file in product_dir.rglob("*.md"):
                    if md_file.name == "index.md":
                        continue
                    content = md_file.read_text(encoding="utf-8")
                    if any(kw in content.lower() for kw in query_lower.split()):
                        results.append((str(md_file), content[:500]))
                        if len(results) >= top_k:
                            break
                if len(results) >= top_k:
                    break
        
        return results[:top_k]
    
    # ========== Reranking ==========
    def reranker(self, query: str, candidates: list) -> list:
        """Rerank retrieved documents by relevance."""
        if not candidates:
            return []
        
        # Simple relevance scoring based on keyword overlap
        query_terms = set(query.lower().split())
        scored = []
        
        for doc_path, content in candidates:
            content_terms = set(content.lower().split())
            overlap = len(query_terms & content_terms)
            scored.append((doc_path, content, overlap))
        
        # Sort by relevance score
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(path, text) for path, text, _ in scored[:5]]
    
    # ========== Confidence ==========
    def confidence(self, docs: list) -> float:
        """Calculate confidence score based on retrieved docs."""
        if not docs:
            return 0.0
        
        # Base confidence on number of relevant docs found
        base_confidence = min(len(docs) / 5, 1.0) * 0.7
        
        # Boost if docs contain substantial content
        content_boost = sum(1 for _, text in docs if len(text) > 200) / len(docs) * 0.3
        
        return min(base_confidence + content_boost, 1.0)
    
    # ========== Risk Assessment ==========
    def risk(self, issue: str) -> str:
        """Assess risk level of the issue."""
        risk_keywords = {
            "high": ["fraud", "unauthorized", "payment failed", "money deducted", 
                     "account hacked", "breach", "stolen", "compromised"],
            "medium": ["refund", "cancel", "error", "bug", "issue", "problem"],
            "low": ["question", "how to", "where", "what is", "help"]
        }
        
        issue_lower = issue.lower()
        
        for level in ["high", "medium", "low"]:
            if any(kw in issue_lower for kw in risk_keywords[level]):
                return level
        return "low"
    
    # ========== Decision ==========
    def decide(self, confidence: float, risk: str) -> str:
        """Decide whether to auto-reply or escalate."""
        threshold = 0.65
        
        if confidence >= threshold and risk != "high":
            return "replied"
        return "escalated"
    
    # ========== Classification ==========
    def classify(self, issue: str) -> str:
        """Classify the request type into the allowed taxonomy."""
        system_prompt = """Classify the support ticket into one of these categories:
- product issue
- feature_request
- bug
- invalid

If the ticket is about a real product problem or operational issue, return `product issue`.
If it is a request for new behavior or enhancement, return `feature_request`.
If it is a report of malfunction or error, return `bug`.
If it is not a valid support request or cannot be answered, return `invalid`.

Respond with only the category name exactly as shown."""
        
        result = self._call_llm(system_prompt, issue)
        return self._normalize_request_type(result, issue)

    def _normalize_request_type(self, result: str, issue: str) -> str:
        """Normalize classifier output to one of the allowed values."""
        allowed = {
            "product issue": "product issue",
            "product_issue": "product issue",
            "feature_request": "feature_request",
            "feature request": "feature_request",
            "bug": "bug",
            "invalid": "invalid"
        }
        normalized = result.strip().lower()
        if normalized in allowed:
            return allowed[normalized]
        normalized = normalized.replace("-", " ").replace("_", " ")
        if normalized in allowed:
            return allowed[normalized]
        # Heuristic fallback based on issue text
        issue_lower = issue.lower()
        if any(word in issue_lower for word in ["bug", "error", "crash", "fail", "failure", "not working", "issue"]):
            return "bug"
        if any(word in issue_lower for word in ["feature", "enhancement", "add", "support for", "ability to", "request"]):
            return "feature_request"
        if any(word in issue_lower for word in ["invalid", "spam", "not a support", "question", "how do i", "where is", "what is"]):
            return "invalid"
        return "product issue"
    
    # ========== Product Area Detection ==========
    def detect_area(self, docs: list) -> str:
        """Detect the product area from documents."""
        if not docs:
            return "unknown"
        
        # Use the immediate parent folder of the retrieved file as the product area
        areas = []
        for doc_path, _ in docs:
            parent_folder = os.path.basename(os.path.dirname(doc_path))
            if parent_folder:
                areas.append(parent_folder)
        
        if areas:
            return max(set(areas), key=areas.count)
        return "general"
    
    # ========== Response Generation ==========
    def generate(self, issue: str, context: str) -> str:
        """Generate a response to the support ticket."""
        system_prompt = """You are a helpful support agent. Based on the provided context 
from knowledge base articles, generate a helpful and professional response to the 
user's issue. If the context doesn't contain enough information to fully answer 
the question, acknowledge the limitation and provide what help you can."""
        
        user_prompt = f"""User Issue:
{issue}

Relevant Knowledge Base:
{context}

Generate a helpful support response:"""
        
        return self._call_llm(system_prompt, user_prompt)


# Factory function for creating system instances
def create_system(provider: str = "openai", model: str = None) -> LLMSystem:
    """Create an LLM system instance."""
    if model is None:
        model = "gpt-4o" if provider == "openai" else "claude-sonnet-4-20250514"
    return LLMSystem(provider=provider, model=model)