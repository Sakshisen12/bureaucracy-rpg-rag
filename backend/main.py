import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------------------------------------------------------------------------
# Global BM25 retriever — pure Python keyword retrieval, zero local model,
# zero API calls during PDF upload. No PyTorch, no embeddings, no quota issues.
# ---------------------------------------------------------------------------
retriever: BM25Retriever | None = None

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


class GameTurnOutput(BaseModel):
    narrative: str = Field(description="Story consequence of player action")
    clause_cited: str = Field(description="Exact document rule/clause reference used")
    compliance_change: int = Field(description="Stat change (-30 to +10)")
    budget_change: int = Field(description="Budget expense delta in dollars")
    sanity_change: int = Field(description="Sanity change (-20 to +5)")


@app.get("/")
def read_root():
    return {"status": "online", "message": "Bureaucracy RPG API is running!"}


@app.post("/upload")
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever
    try:
        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load and chunk the PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.remove(tmp_path)

        if not docs:
            raise HTTPException(status_code=400, detail="PDF appears to be empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        # BM25 keyword retrieval — pure Python, no API, no embeddings, no PyTorch
        retriever = BM25Retriever.from_documents(chunks, k=3)

        return {"status": "success", "message": f"Processed {len(chunks)} chunks from {len(docs)} pages"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[upload] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload Error: {str(e)}")


@app.post("/evaluate")
@app.post("/evaluate/")
async def evaluate_action(req: ActionRequest):
    global retriever

    if retriever is None:
        raise HTTPException(status_code=400, detail="Upload a PDF document first!")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY missing — add it in Render Environment settings."
        )

    try:
        # Retrieve relevant policy chunks via BM25 keyword matching
        docs = retriever.invoke(req.action)
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0.3,
            google_api_key=api_key,
        )
        structured_llm = llm.with_structured_output(GameTurnOutput)

        prompt = f"""
        You are a humorous compliance Game Master evaluating an employee's action against company policies.

        DOCUMENT RULES CONTEXT:
        {context}

        CURRENT PLAYER STATS:
        {req.game_state}

        PLAYER ACTION:
        {req.action}

        Evaluate compliance against the document rules and return a JSON formatted result.
        """

        result = structured_llm.invoke(prompt)
        return result.model_dump()

    except Exception as e:
        print(f"[evaluate] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation Error: {str(e)}")