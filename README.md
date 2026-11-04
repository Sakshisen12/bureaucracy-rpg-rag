# ⚖️ Bureaucracy Survival RPG

> A turn-based RPG where your survival depends on how well you follow the rules — sourced directly from your own policy documents.

**Live Demo:** [bureaucracy-rpg-rag.vercel.app](https://bureaucracy-rpg-rag.vercel.app) &nbsp;|&nbsp; **Backend API:** [bureaucracy-rpg-rag.onrender.com](https://bureaucracy-rpg-rag.onrender.com)

---

## 🎮 What Is This?

**Bureaucracy Survival RPG** is a web game powered by **Retrieval-Augmented Generation (RAG)**. You upload any company policy PDF (HR handbook, expense policy, compliance guide, etc.), and the game's AI Game Master reads it in real-time to judge your in-game actions against the actual rules inside that document.

Every decision you make — expensing a client dinner, requesting equipment, skipping a form — is evaluated for **compliance**, **budget impact**, and **sanity cost** based on the document *you* provided.

---

## ✨ Features

- 📄 **Upload any policy PDF** — the game adapts to *your* rules
- 🤖 **AI-powered Game Master** — uses Groq (Llama) to narrate consequences and cite exact document clauses
- 🔍 **BM25 Retrieval** — pure-Python keyword-based document search, no embeddings or PyTorch required
- 📊 **Live Player HUD** — tracks Compliance %, Budget ($), and Sanity % in real-time
- ⚡ **Quick Action Presets** — one-click scenarios to jump straight into gameplay
- 🛡️ **Rate Limiting** — 20 evaluations per user per hour to protect API quota
- 🔁 **Auto Model Detection** — automatically fetches available Groq models at runtime; never breaks on model deprecations
- 💀 **Game Over States** — compliance or sanity hitting 0 ends the run

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, Vite, Tailwind CSS v4, Lucide Icons |
| **Backend** | FastAPI, Python 3.11 |
| **RAG / Retrieval** | LangChain + BM25Retriever (`rank_bm25`) |
| **PDF Parsing** | PyPDFLoader (`pypdf`) |
| **LLM** | Groq API — Llama 3.x (auto-detected, free tier) |
| **HTTP Client** | `httpx` — direct REST calls, no SDK needed |
| **Frontend Deploy** | Vercel |
| **Backend Deploy** | Render |

---

## 🏗️ Project Structure

```
bureaucracy-rpg-rag/
├── backend/
│   ├── main.py            # FastAPI app — /upload, /evaluate, /models endpoints
│   ├── requirements.txt   # Python dependencies
│   └── .python-version    # Python 3.11
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main game UI (upload, HUD, chat terminal)
│   │   ├── App.css        # Neobrutalist styling
│   │   └── main.jsx       # React entry point
│   ├── package.json
│   └── vite.config.js
├── render.yaml            # Render deployment config
└── .gitignore
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- A [Groq](https://console.groq.com) API key (free, no credit card needed)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/bureaucracy-rpg-rag.git
cd bureaucracy-rpg-rag
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the `backend/` folder:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free API key at [console.groq.com](https://console.groq.com) → API Keys → Create key (starts with `gsk_...`)

Start the backend:

```bash
uvicorn main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`.

### 3. Set up the frontend

```bash
cd ../frontend
npm install
```

In `frontend/src/App.jsx`, update the API base URL for local development:

```js
const API_BASE_URL = 'http://localhost:8000';
```

Start the frontend:

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 🌐 Deployment

### Backend → Render

The `render.yaml` at the root handles backend deployment automatically.

1. Push this repo to GitHub
2. Go to [Render](https://render.com) → **New Web Service** → connect your repo
3. Render will auto-detect `render.yaml` and configure the service
4. Add the environment variable `GROQ_API_KEY` in the Render dashboard under **Environment**

### Frontend → Vercel

1. Go to [Vercel](https://vercel.com) → **New Project** → import your repo
2. Set **Root Directory** to `frontend`
3. Vercel will auto-detect Vite and deploy
4. Make sure `API_BASE_URL` in `App.jsx` points to your Render backend URL

---

## 📡 API Reference

### `GET /`
Health check.

**Response:**
```json
{
  "status": "online",
  "message": "Bureaucracy RPG API is running!",
  "rate_limit": "20 evaluations/hour per user",
  "active_model": "llama-3.1-8b-instant"
}
```

### `GET /models`
Lists all available Groq text-chat models for the configured API key. Useful for debugging.

**Response:**
```json
{
  "available_text_models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
  "active_model": "llama-3.1-8b-instant"
}
```

### `POST /upload`
Upload and index a PDF policy document.

- **Body:** `multipart/form-data` with field `file` (PDF)
- **Response:** `{ "status": "success", "message": "Processed N chunks from M pages" }`

### `POST /evaluate`
Evaluate a player action against the loaded document.

**Request body:**
```json
{
  "action": "Expense a $400 client dinner",
  "game_state": {
    "compliance": 85,
    "budget": 420,
    "sanity": 70
  }
}
```

**Response:**
```json
{
  "narrative": "Your dinner receipt triggers Section 4.2...",
  "clause_cited": "Section 4.2 — Entertainment Expense Limit",
  "compliance_change": -15,
  "budget_change": -400,
  "sanity_change": -5
}
```

---

## ⚙️ How It Works

```
User uploads PDF
      ↓
PyPDFLoader parses the document
      ↓
RecursiveCharacterTextSplitter splits into ~500-token chunks
      ↓
BM25Retriever indexes chunks (no embeddings, no API calls)
      ↓
User submits an action
      ↓
BM25 retrieves the top 3 most relevant chunks
      ↓
Chunks + action + game state → Groq (Llama) prompt
      ↓
Groq returns structured JSON (narrative, clause, stat changes)
      ↓
Frontend updates HUD and renders the response in the terminal
```

---

## 🔒 Rate Limiting

The `/evaluate` endpoint is rate-limited to **20 requests per IP per hour** to protect the Groq API key quota. Exceeding the limit returns HTTP `429`. The limit resets every hour (stored in-memory; resets on redeploy).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT — feel free to use this for your own projects, hackathons, or compliance training (we're not responsible for any actual HR violations 😄).

---

*Built with ☕ and an unhealthy respect for bureaucracy.*
