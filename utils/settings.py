from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cache

class Settings(BaseSettings):
    github_split: str
    survey_csv: str

    model_config = SettingsConfigDict(env_file=".env")








@cache
def get_settings() -> Settings:
    return Settings()