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
# ---------------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"
GROQ_MODEL_FALLBACK = "llama3-8b-8192"  # smaller/faster fallback

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


async def call_groq(api_key: str, prompt: str, max_retries: int = 5) -> dict:
    """Call Groq REST API (OpenAI-compatible). Returns parsed JSON dict."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(max_retries):
        # Switch to smaller fallback model on last 2 attempts
        model = GROQ_MODEL_FALLBACK if attempt >= max_retries - 2 else GROQ_MODEL
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(GROQ_URL, json=body, headers=headers)

            print(f"[groq] model={model} status={resp.status_code} attempt={attempt+1}")

            if resp.status_code == 429:
                raw = resp.text
                print(f"[groq] 429 raw: {raw[:300]}")

                # Daily limit exhausted — no point retrying today
                if "daily" in raw.lower() or "exhausted" in raw.lower():
                    raise HTTPException(
                        status_code=429,
                        detail="Groq daily quota exhausted. Free tier allows ~14,400 req/day. Please try again tomorrow or get a paid plan at console.groq.com."
                    )

                # Per-minute rate limit — retry with backoff
                if attempt < max_retries - 1:
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                    print(f"[groq] 429 rate limit, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    last_error = Exception("429 RATE_LIMIT")
                    continue

                raise HTTPException(
                    status_code=429,
                    detail="Groq rate limit hit. Free tier allows ~30 req/min. Please wait a minute and try again."
                )

            if resp.status_code == 503:
                if attempt < max_retries - 1:
                    wait = min(5 * (2 ** attempt), 30)
                    print(f"[groq] 503 overloaded, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    last_error = Exception("503 UNAVAILABLE")
                    continue
                raise HTTPException(status_code=503, detail="Groq servers are overloaded right now. Please wait 30 seconds and try again.")

            if resp.status_code == 401:
                raise HTTPException(
                    status_code=500,
                    detail="Groq API key is invalid or missing — check GROQ_API_KEY in your environment variables."
                )

            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Groq API error {resp.status_code}: {resp.text[:300]}")

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
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    raise last_error


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
        "model": GROQ_MODEL
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