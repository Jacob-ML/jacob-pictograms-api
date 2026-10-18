from os import getenv

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=getenv("CLOUD_BASE_URL"), api_key=getenv("CLOUD_API_KEY")
)


def get_embedding(text: str) -> list[float]:
    return client.embeddings.create(
        input=text, model=getenv("EMBEDDING_MODEL"), dimensions=256
    )
