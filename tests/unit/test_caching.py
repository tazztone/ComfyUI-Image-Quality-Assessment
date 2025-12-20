"""
Unit tests for model caching and hashing.
Tests utils/iqa_core.py without requiring ComfyUI server.
"""

import pytest
import sys
import importlib.util
from pathlib import Path

# Add custom node root to path BEFORE any project imports
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

# Load the iqa_core module directly using importlib to bypass package __init__.py
iqa_core_path = custom_node_root / "utils" / "iqa_core.py"
spec = importlib.util.spec_from_file_location("iqa_core_module", iqa_core_path)
iqa_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iqa_core)

# Import what we need from the loaded module
ModelCache = iqa_core.ModelCache
get_hash = iqa_core.get_hash


@pytest.mark.unit
class TestModelCache:
    """Tests for the ModelCache class in iqa_core."""

    def test_basic_cache(self):
        cache = ModelCache(max_size=3)
        cache.put("key1", "model1")
        cache.put("key2", "model2")
        assert cache.get("key1") == "model1"
        assert cache.get("key2") == "model2"

    def test_cache_miss(self):
        cache = ModelCache(max_size=3)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = ModelCache(max_size=2)
        cache.put("key1", "model1")
        cache.put("key2", "model2")
        cache.put("key3", "model3")  # This should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") == "model2"
        assert cache.get("key3") == "model3"

    def test_access_updates_lru(self):
        cache = ModelCache(max_size=2)
        cache.put("key1", "model1")
        cache.put("key2", "model2")
        cache.get("key1")  # Access key1, making key2 LRU
        cache.put("key3", "model3")  # Should evict key2
        assert cache.get("key1") == "model1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "model3"


@pytest.mark.unit
class TestGetHash:
    """Tests for the get_hash function in iqa_core."""

    def test_string_hash(self):
        result = get_hash("test_string")
        assert result is not None
        assert len(result) == 64  # SHA256 hex digest

    def test_deterministic(self):
        hash1 = get_hash({"a": 1, "b": 2})
        hash2 = get_hash({"b": 2, "a": 1})  # Same dict, different order
        assert hash1 == hash2
