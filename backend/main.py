import os
import time
import json
import tempfile
from collections import defaultdict
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# ---------------------------------------------------------------------------
# Global BM25 retriever — pure Python, no embeddings, no PyTorch
# ---------------------------------------------------------------------------
retriever: BM25Retriever | None = None

# ---------------------------------------------------------------------------
# Groq REST API — OpenAI-compatible endpoint
# Models are fetched dynamically from Groq at startup so this never breaks
# ---------------------------------------------------------------------------
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_URL  = f"{GROQ_BASE}/chat/completions"

# Models to skip (audio/vision/moderation — not text chat models)
_SKIP_KEYWORDS = ["whisper", "distil-whisper", "guard", "vision", "tts", "playai"]

# Preferred model substrings in priority order
_PREFERRED = [
    "llama-3.3",
    "llama-3.1-70b",
    "llama-3.1-8b",
    "llama3-70b",
    "llama3-8b",
    "llama",
    "gemma",
    "mixtral",
]

# Resolved at runtime
_active_model: str | None = None
_fallback_models: list[str] = []

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter
# ---------------------------------------------------------------------------
MAX_REQUESTS_PER_WINDOW = 20
WINDOW_SECONDS = 60 * 60
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(ip: str):
    now = time.time()
    window_start = now - WINDOW_SECONDS
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > window_start]
    if len(_rate_limit_store[ip]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {MAX_REQUESTS_PER_WINDOW} evaluations per hour per user."
        )
    _rate_limit_store[ip].append(now)


async def fetch_available_models(api_key: str) -> list[str]:
    """Fetch all active text-chat models from Groq API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{GROQ_BASE}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
        if resp.status_code != 200:
            print(f"[groq] Could not fetch models: {resp.status_code} {resp.text[:200]}")
            return []

        all_models = [m["id"] for m in resp.json().get("data", [])]

        # Filter out non-text-chat models
        text_models = [
            m for m in all_models
            if not any(kw in m.lower() for kw in _SKIP_KEYWORDS)
        ]

        # Sort by preference
        def priority(model_id: str) -> int:
            for i, pref in enumerate(_PREFERRED):
                if pref in model_id.lower():
                    return i
            return len(_PREFERRED)

        text_models.sort(key=priority)
        print(f"[groq] Available text models: {text_models}")
        return text_models

    except Exception as e:
        print(f"[groq] Error fetching models: {e}")
        return []


async def ensure_models_loaded(api_key: str):
    """Populate _active_model and _fallback_models if not already set."""
    global _active_model, _fallback_models
    if _active_model:
        return
    models = await fetch_available_models(api_key)
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Could not fetch Groq model list. Check your GROQ_API_KEY in Render Environment settings."
        )
    _active_model = models[0]
    _fallback_models = models[1:]
    print(f"[groq] Primary model: {_active_model}, Fallbacks: {_fallback_models}")


async def call_groq(api_key: str, prompt: str) -> dict:
    """
    Call Groq REST API. Auto-detects available models on first call.
    Skips any model that returns 404 / decommissioned and tries the next.
    """
    global _active_model, _fallback_models

    await ensure_models_loaded(api_key)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    models_to_try = [_active_model] + list(_fallback_models)
    last_error: Exception | None = None

    for model in models_to_try:
        if not model:
            continue
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(GROQ_URL, json=body, headers=headers)

            print(f"[groq] model={model} status={resp.status_code}")

            # Model gone — skip to next and invalidate cache
            if resp.status_code == 404 or (
                resp.status_code == 400 and
                any(kw in resp.text for kw in ["decommissioned", "model_not_found", "does not exist"])
            ):
                print(f"[groq] {model} unavailable, trying next...")
                last_error = Exception(f"{model}: unavailable")
                if _active_model == model:
                    _active_model = _fallback_models[0] if _fallback_models else None
                    _fallback_models = _fallback_models[1:] if _fallback_models else []
                continue

            if resp.status_code == 401:
                raise HTTPException(
                    status_code=500,
                    detail="Groq API key is invalid — check GROQ_API_KEY in Render Environment settings."
                )

            if resp.status_code == 429:
                raw = resp.text
                if "daily" in raw.lower() or "exhausted" in raw.lower():
                    raise HTTPException(
                        status_code=429,
                        detail="Groq daily quota exhausted (~14,400 req/day on free tier). Try again tomorrow."
                    )
                print(f"[groq] Rate limited on {model}, waiting 15s...")
                time.sleep(15)
                continue

            if resp.status_code == 503:
                print(f"[groq] {model} overloaded, waiting 10s...")
                time.sleep(10)
                continue

            if resp.status_code != 200:
                print(f"[groq] {model} error {resp.status_code}: {resp.text[:200]}")
                last_error = Exception(f"{model}: HTTP {resp.status_code}")
                continue

            # ✅ Success — promote this model as primary
            _active_model = model
            print(f"[groq] ✅ success with model={model}")

            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()

            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]).strip()

            return json.loads(text)

        except HTTPException:
            raise
        except Exception as e:
            last_error = e
            print(f"[groq] exception on {model}: {e}")
            continue

    raise HTTPException(
        status_code=503,
        detail=(
            f"All Groq models failed. Last error: {last_error}. "
            "Visit /models endpoint to see what is available on your account."
        )
    )


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ActionRequest(BaseModel):
    action: str
    game_state: dict


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Bureaucracy RPG API is running!",
        "rate_limit": f"{MAX_REQUESTS_PER_WINDOW} evaluations/hour per user",
        "active_model": _active_model or "not yet resolved (will auto-detect on first request)",
    }


@app.get("/models")
async def list_models():
    """Lists all Groq text-chat models available for the configured API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not set in environment"}
    models = await fetch_available_models(api_key)
    return {"available_text_models": models, "active_model": _active_model}


@app.post("/upload")
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.remove(tmp_path)

        if not docs:
            raise HTTPException(status_code=400, detail="PDF appears to be empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        retriever = BM25Retriever.from_documents(chunks, k=3)
        return {"status": "success", "message": f"Processed {len(chunks)} chunks from {len(docs)} pages"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[upload] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload Error: {str(e)}")


@app.post("/evaluate")
@app.post("/evaluate/")
async def evaluate_action(req: ActionRequest, request: Request):
    global retriever

    if retriever is None:
        raise HTTPException(status_code=400, detail="Upload a PDF document first!")

    client_ip = request.headers.get("x-forwarded-for", request.client.host)
    check_rate_limit(client_ip)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY missing — add it in Render Environment settings."
        )

    docs = retriever.invoke(req.action)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a humorous compliance Game Master evaluating an employee's action against company policies.

DOCUMENT RULES CONTEXT:
{context}

CURRENT PLAYER STATS:
{req.game_state}

PLAYER ACTION:
{req.action}

Evaluate the action against the document rules. Respond ONLY with a valid JSON object with exactly these fields:
{{
  "narrative": "A humorous 2-3 sentence story consequence of the action, referencing the policy",
  "clause_cited": "The exact section/clause name from the document that applies",
  "compliance_change": <integer between -30 and +10>,
  "budget_change": <integer, negative means money spent>,
  "sanity_change": <integer between -20 and +5>
}}"""

    result = await call_groq(api_key, prompt)

    required = ["narrative", "clause_cited", "compliance_change", "budget_change", "sanity_change"]
    for field in required:
        if field not in result:
            raise HTTPException(status_code=500, detail=f"Incomplete AI response — missing field: {field}")

    return result