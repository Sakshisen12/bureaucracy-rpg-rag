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
# Gemini REST API — uses stable v1beta endpoint directly with httpx
# No Google SDK needed, works with any valid AI Studio API key
# ---------------------------------------------------------------------------
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_MODEL_FALLBACK = "gemini-1.5-flash"  # fallback if primary is overloaded
GEMINI_URL = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent"
GEMINI_URL_FALLBACK = f"{GEMINI_BASE}/models/{GEMINI_MODEL_FALLBACK}:generateContent"

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


async def call_gemini(api_key: str, prompt: str, max_retries: int = 5) -> dict:
    """Call Gemini REST API directly. Returns parsed JSON dict."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
            # Disable thinking mode — saves quota and avoids parsing thought chunks
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }

    last_error = None
    for attempt in range(max_retries):
        # Switch to fallback model on last 2 attempts if primary keeps failing
        url = GEMINI_URL_FALLBACK if attempt >= max_retries - 2 else GEMINI_URL
        model_used = GEMINI_MODEL_FALLBACK if attempt >= max_retries - 2 else GEMINI_MODEL
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    url,
                    json=body,
                    params={"key": api_key}
                )

            print(f"[gemini] model={model_used} status={resp.status_code} attempt={attempt+1}")

            if resp.status_code == 429:
                raw = resp.text
                print(f"[gemini] 429 raw: {raw[:300]}")

                # Check if daily quota is exhausted (no point retrying)
                if "quota" in raw.lower() or "exhausted" in raw.lower() or "day" in raw.lower():
                    raise HTTPException(
                        status_code=429,
                        detail="Daily API quota exhausted. The free tier allows limited requests per day. Please try again tomorrow or upgrade your API plan at aistudio.google.com."
                    )

                # Per-minute rate limit — retry with backoff
                if attempt < max_retries - 1:
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                    print(f"[gemini] 429 rate limit, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    last_error = Exception("429 RATE_LIMIT")
                    continue

                raise HTTPException(
                    status_code=429,
                    detail="Google API rate limit hit. Free tier allows ~15 requests/min. Please wait 1-2 minutes and try again."
                )
            if resp.status_code == 503:
                if attempt < max_retries - 1:
                    wait = min(5 * (2 ** attempt), 30)  # 5s, 10s, 20s, 30s, 30s
                    print(f"[gemini] 503 overloaded, retrying in {wait}s (attempt {attempt+1}/{max_retries}, model={model_used})")
                    time.sleep(wait)
                    last_error = Exception("503 UNAVAILABLE")
                    continue
                raise HTTPException(status_code=503, detail="Google AI servers are overloaded right now. This is a temporary Google issue — please wait 30 seconds and try again.")
            if resp.status_code == 404:
                detail = resp.text[:500]
                print(f"[gemini] 404 error: {detail}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API 404 — check your GOOGLE_API_KEY in Render. Raw: {detail[:300]}"
                )
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Gemini API error {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]

            # Thinking models return thought chunks first, then the real answer.
            # Skip any part marked with "thought": true and take the last remaining part.
            answer_parts = [p["text"] for p in parts if not p.get("thought", False)]
            text = (answer_parts[-1] if answer_parts else parts[-1]["text"]).strip()

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
        "model": GEMINI_MODEL
    }


@app.get("/models")
async def list_models():
    """Debug endpoint — lists all Gemini models available for the configured API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY not set in environment"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{GEMINI_BASE}/models",
                params={"key": api_key}
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

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY missing — add it in Render Environment settings."
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

    result = await call_gemini(api_key, prompt)

    # Validate required fields
    required = ["narrative", "clause_cited", "compliance_change", "budget_change", "sanity_change"]
    for field in required:
        if field not in result:
            raise HTTPException(status_code=500, detail=f"Incomplete AI response — missing field: {field}")

    return result