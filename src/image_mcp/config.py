from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    internvl_api_key: str = Field(default="")
    internvl_api_endpoint: str = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
    default_model: str = "internvl3-latest"
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    max_tokens: int = 2048
    max_image_size_mb: int = 10
    compression_threshold_mb: int = 5
    max_images_per_request: int = 5
    max_image_dimension: int = 2048


settings = Settings()
