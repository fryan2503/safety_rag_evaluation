import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
client = OpenAI()

batch_id = {
  "file_id": "file-64GNFgTACfqgBk4kRJjpGg",
  "batch_id": "batch_690e88e8fa1c8190a6d1d6f79f3740e9",
  "status": "validating"
}

batch = client.batches.retrieve(batch_id["batch_id"])
print(json.dumps(batch.model_dump(), indent=2))