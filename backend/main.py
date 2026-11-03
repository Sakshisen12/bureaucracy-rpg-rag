import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# ---------------------------------------------------------------------------
# Global in-memory vector store (reset on each new PDF upload).
# Uses Google's embedding API — zero local model downloads, zero PyTorch,
# fits comfortably within Render's 512 MB free-tier RAM.
# ---------------------------------------------------------------------------
vectorstore: InMemoryVectorStore | None = None

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
    global vectorstore

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY missing — add it in Render Environment settings."
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.remove(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        # Google Generative AI Embeddings — API-based, no local model / no PyTorch
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

        return {"status": "success", "message": f"Processed {len(chunks)} chunks"}

    except Exception as e:
        print(f"[upload] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload Error: {str(e)}")


@app.post("/evaluate")
@app.post("/evaluate/")
async def evaluate_action(req: ActionRequest):
    global vectorstore

    if not vectorstore:
        raise HTTPException(status_code=400, detail="Upload a PDF document first!")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY missing — add it in Render Environment settings."
        )

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(req.action)
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
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