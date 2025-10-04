#!/usr/bin/env python3
"""Test script to verify config parsing fix."""

import os
import sys

# Test with comma-separated values (like in .env.example)
os.environ["ALLOWED_EXTENSIONS"] = ".pdf,.docx,.txt"
os.environ["CORS_ORIGINS"] = "http://localhost:3000,http://127.0.0.1:3000"
os.environ["SUPPORTED_LANGUAGES"] = "en,zh"
os.environ["QUERY_EXPANSION_METHODS"] = "llm,synonym"

# Ensure required env vars are set
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

try:
    from src.core.config import get_settings

    settings = get_settings()

    print("✅ Config loaded successfully!")
    print(f"\nAllowed extensions: {settings.app.allowed_extensions}")
    print(f"CORS origins: {settings.app.cors_origins}")
    print(f"Supported languages: {settings.rag.supported_languages}")
    print(f"Query expansion methods: {settings.rag.query_expansion_methods}")

    # Verify parsing worked correctly
    assert settings.app.allowed_extensions == [".pdf", ".docx", ".txt"]
    assert settings.app.cors_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]
    assert settings.rag.supported_languages == ["en", "zh"]
    assert settings.rag.query_expansion_methods == ["llm", "synonym"]

    print("\n✅ All validations passed!")
    sys.exit(0)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
