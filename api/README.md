# SIGHT FastAPI app

A small FastAPI service that combines an OpenAI vector store search with a model Q&A workflow.

## Overview

- POST `/query`: Run a synchronous model call that returns the final answer and a `conversation_id`.
- POST `/stream`: Stream token-level generation via Server-Sent Events (SSE). Each event is a JSON object; during generation `{"type":"token","token":"<delta>"}` is emitted, and a final `{"type":"done","answer":"...","conversation_id":"..."}` is sent.
- GET `/info`: Returns the current `CONFIG` as JSON (useful for debugging).

## Environment variables

The application expects the following environment variables (typically in a `.env` file):

- `MODEL` - model or deployment name to use for generation
- `MAX_TOKENS` - maximum tokens to generate (integer)
- `REASONING_EFFORT` - reasoning effort configuration
- `EMBED_MODEL` - embedding model id
- `TOP_K` - number of search results to retrieve (integer)
- `MAX_CHARS_PER_CONTENT` - truncation for retrieved contents (integer)
- `SYSTEM_INSTRUCTIONS` - instructions injected into every prompt (required; used as the initial system message when creating conversations)
- `OPENAI_API_KEY` - OpenAI API key used by the OpenAI client
- `OPENAI_VECTOR_STORE_ID` - identifier of the OpenAI vector store to search (required for `/query` and `/stream`)

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app with uvicorn:

```bash
uvicorn app:app --reload --port 8000
```

For local development you can optionally use a `.env` file and uncomment the `dotenv` lines in `app.py` to load it automatically.

## Examples

Synchronous POST (curl):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What safety checks should I perform before jogging a UR5e robot?"}'
```

Streaming POST (curl) — use `-N`/`--no-buffer` to see SSE messages:

```bash
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What safety checks should I perform before jogging a UR5e robot?"}'
```

You can also use a browser or an SSE-capable client library to consume `/stream` and process incremental tokens.

---

## Deploy to Google Cloud Run 🚀

The following steps show how to build a Docker image locally, push it to **Artifact Registry**, and deploy the container to **Cloud Run** using the project `sight-480418` and the `us-east5` region. The commands below are PowerShell-friendly and are ready to copy/paste; replace secret values where noted.

### 1) Prepare & test locally

- Build locally. Start Docker Engine and run the following command:

```powershell
docker build -t sight:latest .
```


### 2) Set project, login, and enable required APIs

```powershell
# Log in and set the project
gcloud auth login
gcloud config set project sight-480418

#Run the following command in case you receive a quota warning
gcloud auth application-default set-quota-project sight-480418

# Enable required APIs (do that only once)
gcloud services enable artifactregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
```

### 3) Create an Artifact Registry repo (us-east5)

```powershell
gcloud artifacts repositories create sight-repo --repository-format=docker --location=us-east5 --description="Docker repo for SIGHT"
```

> If the repo already exists you’ll see an error you can safely ignore.

### 4) Configure Docker to use Artifact Registry

```powershell
gcloud auth configure-docker us-east5-docker.pkg.dev --quiet
```

### 5) Tag and push the image

This will fully qualify the Artifact Registry path:

```powershell
# Tag locally
docker tag sight:latest us-east5-docker.pkg.dev/sight-480418/sight-repo/sight:latest

# Push to Artifact Registry
docker push us-east5-docker.pkg.dev/sight-480418/sight-repo/sight:latest
```


### 6) Deploy to Cloud Run 

This deploys a managed Cloud Run service named `sight` using the pushed image. This is OK, but a safer approach is to use Secret Manager for OpenAI API.

```powershell
gcloud run deploy sight `
  --image us-east5-docker.pkg.dev/sight-480418/sight-repo/sight:latest `
  --platform managed `
  --region us-east5 `
  --allow-unauthenticated `
  --port 8080 `
  --memory 2Gi `
  --cpu 1 `
  --env-vars-file env.yaml
```

Note: edit the `env.yaml` file to add API keys.

## Changing environment variables

This is how you can update the existing Cloud Run service's environment variables without a full redeploy:

```powershell
# Update environment variables (creates a new revision)
gcloud run services update sight --update-env-vars TOP_K=10 --region us-east5 --platform managed
```

