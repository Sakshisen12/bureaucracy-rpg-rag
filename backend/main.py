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
GEMINI_MODEL = "gemini-2.0-flash"  # updated model name
GEMINI_URL = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent"

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter
# ---------------------------------------------------------------------------
MAX_REQUESTS_PER_WINDOW = 10
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


async def call_gemini(api_key: str, prompt: str, max_retries: int = 3) -> dict:
    """Call Gemini REST API directly. Returns parsed JSON dict."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json"
        }
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    GEMINI_URL,
                    json=body,
                    params={"key": api_key}
                )

            print(f"[gemini] status={resp.status_code}")

            if resp.status_code == 429:
                raise HTTPException(status_code=429, detail="API quota exhausted. Please wait a while and try again.")
            if resp.status_code == 503:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[gemini] 503, retrying in {wait}s")
                    time.sleep(wait)
                    last_error = Exception("503 UNAVAILABLE")
                    continue
                raise HTTPException(status_code=503, detail="Google AI is overloaded. Try again in 10 seconds.")
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
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])

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