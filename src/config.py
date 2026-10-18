"""Application configuration loaded from environment variables."""

from functools import cached_property

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralised config backed by ``os.environ`` (via python-dotenv)."""

    @cached_property
    def cloud_api_key(self) -> str:
        key = self._get_required("CLOUD_API_KEY")
        if not key or key == "...":
            raise RuntimeError(
                "CLOUD_API_KEY is missing or unset. "
                "Copy .env.example to .env and fill in your key."
            )
        return key

    @cached_property
    def cloud_base_url(self) -> str:
        return self._get_required("CLOUD_BASE_URL")

    @cached_property
    def embedding_model(self) -> str:
        return self._get_required("EMBEDDING_MODEL")

    @staticmethod
    def _get_required(name: str) -> str:
        import os

        value = os.environ.get(name, "")
        if not value:
            raise RuntimeError(f"{name} environment variable is required.")
        return value


config = Config()
