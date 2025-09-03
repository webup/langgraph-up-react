"""Configuration for evaluation tests."""

import pytest


def pytest_configure(config):
    """Configure pytest for evaluation tests."""
    # Register custom markers
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "evaluation: marks tests as evaluation tests"
    )


@pytest.fixture(scope="session")
def evaluation_config():
    """Fixture providing evaluation configuration."""
    return {
        "timeout": 300,  # 5 minute timeout for evaluation tests
        "retry_count": 2,  # Retry failed evaluations
    }


@pytest.fixture(scope="function")
def evaluation_thread_id(request):
    """Generate unique thread ID for evaluation tests."""
    test_name = request.node.name
    return f"eval_test_{test_name}_{id(request)}"