"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file for all tests."""
    # Find the project root (where .env is located)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)

    # Ensure required environment variables are available for tests
    # You can add fallback values or skip tests if keys are missing
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        pytest.skip(f"Missing required environment variables: {missing_keys}")
