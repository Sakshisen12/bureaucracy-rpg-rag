import os
import time
import json
import tempfile
from collections import defaultdict
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# ---------------------------------------------------------------------------
# Global BM25 retriever — pure Python, no embeddings, no PyTorch
# ---------------------------------------------------------------------------
retriever: BM25Retriever | None = None

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter
# Limits each IP to MAX_REQUESTS_PER_WINDOW evaluate calls per time window.
# Resets after WINDOW_SECONDS. Stored in memory (resets on redeploy).
# ---------------------------------------------------------------------------
MAX_REQUESTS_PER_WINDOW = 10        # max /evaluate calls per IP
WINDOW_SECONDS = 60 * 60            # 1 hour window
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


# ---------------------------------------------------------------------------
# Call Gemini with retry logic for transient errors
# ---------------------------------------------------------------------------
def call_gemini_with_retry(model, prompt: str, max_retries: int = 3) -> dict:
    """Call Gemini directly and parse JSON response, with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            error_str = str(e)
            is_retryable = "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[evaluate] Transient error (attempt {attempt+1}), retrying in {wait}s: {e}")
                time.sleep(wait)
                last_error = e
            else:
                raise e
    raise last_error


app = FastAPI()

# CORS — allow_credentials MUST be False when allow_origins=["*"]
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
        "rate_limit": f"{MAX_REQUESTS_PER_WINDOW} evaluations/hour per user"
    }


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

        # BM25 — pure Python keyword retrieval, zero API calls
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

    # Rate limit by IP to protect API key quota
    client_ip = request.headers.get("x-forwarded-for", request.client.host)
    check_rate_limit(client_ip)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY missing — add it in Render Environment settings."
        )

    try:
        docs = retriever.invoke(req.action)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Configure Google Generative AI directly (no langchain wrapper)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
        )

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

        result = call_gemini_with_retry(model, prompt)

        # Validate required fields exist
        required = ["narrative", "clause_cited", "compliance_change", "budget_change", "sanity_change"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing field in AI response: {field}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e)
        print(f"[evaluate] ERROR TYPE: {type(e).__name__}")
        print(f"[evaluate] ERROR: {error_str}")

        if "503" in error_str or "UNAVAILABLE" in error_str:
            raise HTTPException(
                status_code=503,
                detail="Google AI is temporarily overloaded. Please wait 10 seconds and try again."
            )
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            raise HTTPException(
                status_code=429,
                detail="API quota exhausted. Please wait a while and try again."
            )
        raise HTTPException(status_code=500, detail=f"Evaluation Error: {error_str[:500]}")