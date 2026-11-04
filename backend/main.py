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
# Groq REST API — OpenAI-compatible endpoint, no SDK needed
# Free tier: ~14,400 requests/day, 30 req/min
#
# Models listed in preference order — the app tries each one automatically.
# If a model is decommissioned or unavailable, it moves to the next.
# ---------------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODELS = [
    "llama-3.1-8b-instant",        # ✅ confirmed active on free tier
    "llama-3.1-70b-versatile",     # larger model, may need access
    "llama-3.3-70b-versatile",     # newest 70B
    "llama3-groq-8b-8192-tool-use-preview",  # preview fallback
]

# Cache which model worked so we don't retry unavailable ones every request
_working_model: str | None = None

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


async def call_groq(api_key: str, prompt: str) -> dict:
    """
    Call Groq REST API. Automatically tries each model in GROQ_MODELS
    until one succeeds, skipping decommissioned or unavailable ones.
    """
    global _working_model

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Try cached working model first, then fall through the full list
    models_to_try = []
    if _working_model:
        models_to_try.append(_working_model)
    for m in GROQ_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error: Exception | None = None

    for model in models_to_try:
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

            # Model not available on this account — skip to next
            if resp.status_code == 404:
                print(f"[groq] {model} → 404 not found, trying next model...")
                last_error = Exception(f"{model}: 404 not found")
                if model == _working_model:
                    _working_model = None  # invalidate cache
                continue

            # Decommissioned or bad model name — skip to next
            if resp.status_code == 400:
                raw = resp.text
                if "decommissioned" in raw or "model_not_found" in raw or "does not exist" in raw:
                    print(f"[groq] {model} → decommissioned/not found, trying next model...")
                    last_error = Exception(f"{model}: decommissioned")
                    if model == _working_model:
                        _working_model = None
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
                        detail="Groq daily quota exhausted. Free tier allows ~14,400 req/day. Try again tomorrow or visit console.groq.com."
                    )
                # Per-minute rate limit — wait briefly and retry same model
                print(f"[groq] {model} → rate limited, waiting 15s...")
                time.sleep(15)
                continue

            if resp.status_code == 503:
                print(f"[groq] {model} → 503 overloaded, waiting 10s...")
                time.sleep(10)
                continue

            if resp.status_code != 200:
                print(f"[groq] {model} → error {resp.status_code}: {resp.text[:200]}")
                last_error = Exception(f"{model}: HTTP {resp.status_code}")
                continue

            # ✅ Success — cache this model
            _working_model = model
            print(f"[groq] ✅ success with model={model}")

            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()

            # Strip markdown code fences if present
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
            f"All Groq models failed or are unavailable. Last error: {last_error}. "
            "Visit console.groq.com to check which models are active for your account."
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
        "models": GROQ_MODELS,
        "active_model": _working_model or "auto-detect on first request"
    }


@app.get("/models")
async def list_models():
    """Debug endpoint — lists all Groq models available for the configured API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not set in environment"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
        return {"status": resp.status_code, "body": resp.json()}
    except Exception as e:
        return {"error": str(e)}


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

    # Validate required fields
    required = ["narrative", "clause_cited", "compliance_change", "budget_change", "sanity_change"]
    for field in required:
        if field not in result:
            raise HTTPException(status_code=500, detail=f"Incomplete AI response — missing field: {field}")

    return result