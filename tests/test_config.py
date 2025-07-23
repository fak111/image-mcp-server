import pytest
from pydantic import ValidationError
from src.image_mcp.config import Settings


def test_settings_with_api_key(mock_api_key):
    """Test settings initialization with API key."""
    settings = Settings()
    assert settings.internvl_api_key == "test_api_key_12345"
    assert settings.internvl_api_endpoint == "https://chat.intern-ai.org.cn/api/v1/chat/completions"
    assert settings.default_model == "internvl3-latest"
    assert settings.default_temperature == 0.7
    assert settings.default_top_p == 0.9
    assert settings.max_tokens == 2048
    assert settings.max_image_size_mb == 10
    assert settings.compression_threshold_mb == 5
    assert settings.max_images_per_request == 5
    assert settings.max_image_dimension == 2048


def test_settings_without_api_key(monkeypatch):
    """Test settings initialization without API key should fail."""
    # Clear all related environment variables
    monkeypatch.delenv("INTERNVL_API_KEY", raising=False)
    
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    # Check that the error is about the missing API key
    errors = exc_info.value.errors()
    assert any("internvl_api_key" in str(error) for error in errors)


def test_settings_custom_values():
    """Test settings with custom environment variables."""
    import os
    
    # Temporarily set environment variables for this test
    old_values = {}
    test_env = {
        "INTERNVL_API_KEY": "custom_key",
        "DEFAULT_MODEL": "custom-model", 
        "DEFAULT_TEMPERATURE": "0.5",
        "MAX_TOKENS": "1024"
    }
    
    # Save old values and set new ones
    for key, value in test_env.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        settings = Settings()
        assert settings.internvl_api_key == "custom_key"
        assert settings.default_model == "custom-model"
        assert settings.default_temperature == 0.5
        assert settings.max_tokens == 1024
    finally:
        # Restore old values
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value