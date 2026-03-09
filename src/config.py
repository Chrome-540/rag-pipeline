from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "rag-index"

    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 10

    class Config:
        env_file = ".env"


settings = Settings()
