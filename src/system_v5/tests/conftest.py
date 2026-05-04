import os
import sys

# Ensure the repository `src` directory is on sys.path so tests can import the package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: tests that require an LLM backend")
