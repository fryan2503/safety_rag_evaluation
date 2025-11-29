import os
from dotenv import load_dotenv


class EnvironmentConfig:
    def __init__(self):
        load_dotenv(override=True)
        self.VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
        self.ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")