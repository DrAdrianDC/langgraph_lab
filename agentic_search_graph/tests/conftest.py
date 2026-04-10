"""
Pytest configuration: patch API clients before any module import so that
tests can import from src.langgraph_agent without real API keys.
"""
import os
from unittest.mock import MagicMock, patch

# Provide dummy keys before anything else runs.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

# Start patches at collection time so the module-level ChatGroq / TavilySearch
# instantiation in langgraph_agent.py succeeds without real credentials.
patch("langchain_groq.ChatGroq", return_value=MagicMock()).start()
patch("langchain_tavily.TavilySearch", return_value=MagicMock()).start()
