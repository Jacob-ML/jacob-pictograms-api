from os import getenv

from openai import OpenAI
from dotenv import load_dotenv


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY")
)


def get_embedding(text: str) -> list[float]:
    return client.embeddings.create(
        input=text, model=getenv("EMBEDDING_MODEL"), dimensions=256
    )

