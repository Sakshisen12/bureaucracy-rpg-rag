import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { 
  ShieldAlert, 
  DollarSign, 
  Smile, 
  Upload, 
  Send, 
  RotateCcw, 
  FileText, 
  CheckCircle2, 
  AlertCircle,
  Sparkles,
  ChevronRight,
  Terminal
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

export default function App() {
  const [gameState, setGameState] = useState({
    compliance: 100,
    budget: 500,
    sanity: 80
  });

  const [history, setHistory] = useState([
    {
      action: "System Ready",
      narrative: "Welcome to Bureaucracy Survival RPG! Upload your policy document to begin.",
      clause: "Section 1.0 - Ingestion Phase",
      statsDelta: null
    }
  ]);

  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(''); // 'uploading' | 'success' | 'error'
  const [userAction, setUserAction] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, loading]);

  const presetActions = [
    "Expense a $400 client dinner on a Sunday night",
    "Request an ergonomic chair without manager approval",
    "Work overtime without filing Form B in advance",
    "Take 3 days off during end-of-quarter freeze"
  ];

  const handleFileUpload = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setUploadStatus('uploading');
    setErrorMsg('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.status === 'success') {
        setUploadStatus('success');
      }
    } catch (err) {
      setUploadStatus('error');
      setErrorMsg(err.response?.data?.detail || 'Failed to upload and process PDF');
    }
  };

  const handleActionSubmit = async (actionToSubmit) => {
    const actionText = actionToSubmit || userAction;
    if (!actionText.trim()) return;

    if (uploadStatus !== 'success') {
      setErrorMsg('Please upload a policy PDF document first!');
      return;
    }

    setErrorMsg('');
    setLoading(true);

    try {
      const payload = {
        action: actionText,
        game_state: gameState
      };

      const response = await axios.post(`${API_BASE_URL}/evaluate`, payload);
      const data = response.data;

      setGameState(prev => ({
        compliance: Math.max(0, Math.min(100, prev.compliance + data.compliance_change)),
        budget: prev.budget + data.budget_change,
        sanity: Math.max(0, Math.min(100, prev.sanity + data.sanity_change))
      }));

      setHistory(prev => [
        ...prev,
        {
          action: actionText,
          narrative: data.narrative,
          clause: data.clause_cited,
          statsDelta: {
            compliance: data.compliance_change,
            budget: data.budget_change,
            sanity: data.sanity_change
          }
        }
      ]);

      setUserAction('');
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || 'Error evaluating compliance action.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setGameState({ compliance: 100, budget: 500, sanity: 80 });
    setHistory([
      {
        action: "System Reset",
        narrative: "Terminal state cleared.",
        clause: "Section 0.0 - Memory Cleared",
        statsDelta: null
      }
    ]);
    setErrorMsg('');
  };

  const isGameOver = gameState.compliance <= 0 || gameState.sanity <= 0;

  return (
    <div className="min-h-screen bg-[#FFFBEB] font-mono p-4 md:p-8 text-black selection:bg-[#FACC15]">
      <header className="neo-box p-6 mb-8 bg-[#FACC15] flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <h1 className="text-3xl md:text-5xl font-black uppercase tracking-tight flex items-center gap-3">
            ⚖️ BUREAUCRACY SURVIVAL RPG
          </h1>
          <p className="text-sm md:text-base font-bold mt-1 text-black/80">
            RAG Constraint-Checking Engine • React & FastAPI
          </p>
        </div>
        <button 
          onClick={handleReset}
          className="neo-btn bg-[#EC4899] text-white px-5 py-3 flex items-center gap-2 hover:bg-[#db2777]"
        >
          <RotateCcw className="w-5 h-5" />
          RESET GAME
        </button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Control Column */}
        <div className="lg:col-span-4 space-y-6">
          
          <div className="neo-box p-6 bg-white space-y-4">
            <h2 className="text-xl font-black uppercase flex items-center gap-2 border-b-4 border-black pb-2">
              <Upload className="w-5 h-5 text-[#A855F7]" />
              Document Upload
            </h2>

            <div>
              <label className="block font-black text-xs uppercase mb-1">Upload Policy PDF</label>
              <label className="neo-btn w-full bg-[#06B6D4] text-white p-3 flex items-center justify-center gap-2 cursor-pointer hover:bg-[#0891b2]">
                <Upload className="w-5 h-5" />
                {file ? file.name.substring(0, 18) + '...' : 'SELECT POLICY PDF'}
                <input type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>

            {uploadStatus === 'uploading' && (
              <div className="neo-box p-3 bg-[#FEF08A] font-bold text-xs flex items-center gap-2 animate-pulse">
                <Sparkles className="w-4 h-4" /> Processing PDF...
              </div>
            )}
            {uploadStatus === 'success' && (
              <div className="neo-box p-3 bg-[#84CC16] text-white font-bold text-xs flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" /> Policy PDF Active!
              </div>
            )}
            {uploadStatus === 'error' && (
              <div className="neo-box p-3 bg-[#EF4444] text-white font-bold text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4" /> Upload Failed! Check Backend.
              </div>
            )}
          </div>

          <div className="neo-box p-6 bg-[#A855F7] text-white space-y-5">
            <h2 className="text-xl font-black uppercase border-b-4 border-black pb-2 flex justify-between items-center">
              <span>PLAYER HUD</span>
              <span className="text-xs bg-black text-[#FACC15] px-2 py-1">LIVE</span>
            </h2>

            <div className="space-y-1">
              <div className="flex justify-between font-black text-sm">
                <span className="flex items-center gap-1"><ShieldAlert className="w-4 h-4" /> COMPLIANCE</span>
                <span>{gameState.compliance}%</span>
              </div>
              <div className="w-full neo-box h-5 bg-black/30 p-0.5">
                <div 
                  className="h-full bg-[#FACC15] transition-all duration-500" 
                  style={{ width: `${gameState.compliance}%` }}
                />
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-black text-sm">
                <span className="flex items-center gap-1"><DollarSign className="w-4 h-4" /> BUDGET</span>
                <span>${gameState.budget}</span>
              </div>
              <div className="w-full neo-box h-5 bg-black/30 p-0.5">
                <div 
                  className="h-full bg-[#06B6D4] transition-all duration-500" 
                  style={{ width: `${Math.min(100, (gameState.budget / 500) * 100)}%` }}
                />
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-black text-sm">
                <span className="flex items-center gap-1"><Smile className="w-4 h-4" /> SANITY</span>
                <span>{gameState.sanity}%</span>
              </div>
              <div className="w-full neo-box h-5 bg-black/30 p-0.5">
                <div 
                  className="h-full bg-[#EC4899] transition-all duration-500" 
                  style={{ width: `${gameState.sanity}%` }}
                />
              </div>
            </div>
          </div>

          <div className="neo-box p-6 bg-white space-y-3">
            <h3 className="font-black uppercase text-xs text-gray-500">Quick Actions:</h3>
            <div className="space-y-2">
              {presetActions.map((action, idx) => (
                <button
                  key={idx}
                  disabled={loading || isGameOver}
                  onClick={() => handleActionSubmit(action)}
                  className="w-full text-left neo-btn p-2 bg-[#F8FAFC] hover:bg-[#FACC15] text-xs font-bold flex items-center justify-between group disabled:opacity-50"
                >
                  <span className="truncate mr-2">{action}</span>
                  <ChevronRight className="w-4 h-4 shrink-0 group-hover:translate-x-1 transition-transform" />
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* Right Chat Column */}
        <div className="lg:col-span-8 flex flex-col space-y-6">
          
          {errorMsg && (
            <div className="neo-box p-4 bg-[#EF4444] text-white font-black text-sm flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5 shrink-0" />
                <span>{errorMsg}</span>
              </div>
              <button onClick={() => setErrorMsg('')} className="font-black text-lg">&times;</button>
            </div>
          )}

          <div className="neo-box p-6 bg-[#18181B] text-white flex-1 min-h-[500px] max-h-[650px] overflow-y-auto space-y-6">
            
            <div className="flex items-center gap-2 border-b-2 border-zinc-700 pb-3 text-zinc-400 font-bold text-xs uppercase">
              <Terminal className="w-4 h-4 text-[#FACC15]" />
              <span>Auditor Terminal Stream</span>
            </div>

            {history.map((turn, index) => (
              <div key={index} className="space-y-3">
                <div className="flex justify-end">
                  <div className="neo-box p-4 bg-[#06B6D4] text-black max-w-[85%] font-bold">
                    <p className="text-xs uppercase opacity-75 font-black mb-1">Your Action:</p>
                    <p>{turn.action}</p>
                  </div>
                </div>

                <div className="flex justify-start">
                  <div className="neo-box p-5 bg-[#27272A] text-white max-w-[90%] space-y-3 border-zinc-600">
                    <div className="flex items-center justify-between gap-2 border-b border-zinc-700 pb-2">
                      <span className="font-black text-xs text-[#FACC15] uppercase">
                        🤖 Compliance Response
                      </span>
                      {turn.statsDelta && (
                        <div className="flex gap-2 text-[10px] font-black">
                          <span className={`px-1.5 py-0.5 ${turn.statsDelta.compliance >= 0 ? 'bg-[#84CC16] text-black' : 'bg-[#EF4444] text-white'}`}>
                            COMP: {turn.statsDelta.compliance > 0 ? `+${turn.statsDelta.compliance}` : turn.statsDelta.compliance}
                          </span>
                          <span className={`px-1.5 py-0.5 ${turn.statsDelta.budget >= 0 ? 'bg-[#06B6D4] text-black' : 'bg-[#EF4444] text-white'}`}>
                            BUDGET: {turn.statsDelta.budget >= 0 ? `+$${turn.statsDelta.budget}` : `-$${Math.abs(turn.statsDelta.budget)}`}
                          </span>
                        </div>
                      )}
                    </div>

                    <p className="text-sm font-medium leading-relaxed">{turn.narrative}</p>

                    {turn.clause && (
                      <div className="neo-box p-3 bg-[#3F3F46] text-[#FACC15] font-bold text-xs flex items-start gap-2">
                        <FileText className="w-4 h-4 shrink-0 mt-0.5" />
                        <div>
                          <span className="block text-[10px] uppercase text-zinc-400 font-black">Citation:</span>
                          <span>{turn.clause}</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="neo-box p-4 bg-[#27272A] text-[#FACC15] font-bold text-xs flex items-center gap-3">
                  <Sparkles className="w-5 h-5 animate-spin" />
                  <span>Auditing move against rules context...</span>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {isGameOver ? (
            <div className="neo-box p-6 bg-[#EF4444] text-white font-black text-center text-lg uppercase">
              💀 Game Over! Click 'RESET GAME' to restart.
            </div>
          ) : (
            <form 
              onSubmit={(e) => { e.preventDefault(); handleActionSubmit(); }} 
              className="flex gap-3"
            >
              <input
                type="text"
                placeholder="Type your action..."
                value={userAction}
                onChange={(e) => setUserAction(e.target.value)}
                disabled={loading}
                className="flex-1 neo-box p-4 font-bold bg-white text-black focus:outline-none focus:bg-[#FEF08A] disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={loading || !userAction.trim()}
                className="neo-btn bg-[#84CC16] text-black px-6 py-4 flex items-center gap-2 hover:bg-[#65a30d] disabled:opacity-50"
              >
                <Send className="w-5 h-5" />
                <span className="hidden md:inline">SUBMIT</span>
              </button>
            </form>
          )}

        </div>

      </div>
    </div>
  );
}