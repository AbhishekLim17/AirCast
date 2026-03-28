import os


# Provide deterministic dummy env so modules that validate config can import in tests.
os.environ.setdefault("WAQI_API_TOKEN", "test-token")
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("HF_TOKEN", "hf_test_token")
os.environ.setdefault("HF_USERNAME", "test-user")


def pytest_sessionstart(session):
    # Keep hook for future shared test bootstrap needs.
    return None
