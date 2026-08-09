import { useState } from 'react'
import { api, type SearchResponse, type SynthesizeResponse } from './api'
import { Search, Loader2, AlertTriangle, CheckCircle, ThumbsUp, ThumbsDown, Upload, Check } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import clsx from 'clsx'

function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<"hybrid"|"dense">("hybrid")
  const [loading, setLoading] = useState(false)
  const [searchRes, setSearchRes] = useState<SearchResponse | null>(null)
  
  const [synLoading, setSynLoading] = useState(false)
  const [synRes, setSynRes] = useState<SynthesizeResponse | null>(null)

  const [docText, setDocText] = useState('')
  const [uploadLoading, setUploadLoading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!query) return
    setLoading(true)
    setSynRes(null)
    try {
      const res = await api.search(query, mode)
      setSearchRes(res)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!docText) return
    setUploadLoading(true)
    setUploadSuccess(false)
    try {
      await api.addDocument(docText)
      setUploadSuccess(true)
      setDocText('')
      setTimeout(() => setUploadSuccess(false), 3000)
    } catch (e) {
      console.error(e)
    } finally {
      setUploadLoading(false)
    }
  }

  const handleSynthesize = async () => {
    if (!query || !searchRes) return
    setSynLoading(true)
    try {
      const passages = searchRes.results.map(r => r.text)
      const res = await api.synthesize(query, passages)
      setSynRes(res)
    } catch (e) {
      console.error(e)
    } finally {
      setSynLoading(false)
    }
  }

  const handleFeedback = async (docId: string, feedback: "relevant" | "not_relevant") => {
    try {
      await api.submitFeedback(query, docId, feedback)
      // Feedback submitted visually can be indicated by state change, keeping simple here
    } catch(e) {
      console.error(e)
    }
  }

  return (
    <div className="min-h-screen p-8 max-w-4xl mx-auto flex flex-col gap-8 font-sans">
      <header className="text-center mt-8">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">Zero-LLM Relevance-Aware Retrieval</h1>
        <p className="text-slate-400 text-lg">Fast CPU-only Hybrid Retrieval with Confidence Validation</p>
      </header>

      <form onSubmit={handleSearch} className="flex gap-4">
        <input 
          className="flex-1 bg-slate-800 border border-slate-700 rounded-xl p-4 outline-none focus:border-blue-500 transition-colors shadow-lg"
          placeholder="Ask a medical question..."
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <select 
          className="bg-slate-800 border border-slate-700 rounded-xl p-4 outline-none shadow-lg cursor-pointer hover:border-slate-600 transition-colors"
          value={mode}
          onChange={e => setMode(e.target.value as any)}
        >
          <option value="hybrid">Hybrid</option>
          <option value="dense">Dense</option>
        </select>
        <button type="submit" disabled={loading} className="bg-blue-600 hover:bg-blue-500 p-4 rounded-xl text-white flex items-center gap-2 shadow-lg transition-colors font-medium">
          {loading ? <Loader2 className="animate-spin" /> : <Search size={20} />} Search
        </button>
      </form>

      <form onSubmit={handleUpload} className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 shadow-lg flex flex-col gap-3">
        <textarea 
          className="bg-slate-900 border border-slate-700 rounded-lg p-3 outline-none focus:border-indigo-500 transition-colors text-slate-200 resize-y min-h-[80px]"
          placeholder="Paste medical document or notes here to add to the knowledge base..."
          value={docText}
          onChange={e => setDocText(e.target.value)}
        />
        <div className="flex justify-between items-center">
          <span className="text-xs text-slate-500 flex-1">This will instantly update the BM25 & FAISS indices.</span>
          <button type="submit" disabled={uploadLoading || !docText} className="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg text-white flex items-center gap-2 transition-colors disabled:opacity-50 text-sm font-medium">
            {uploadLoading ? <Loader2 className="animate-spin" size={16} /> : (uploadSuccess ? <Check className="text-green-400" size={16} /> : <Upload size={16} />)}
            {uploadSuccess ? "Added!" : "Add Document"}
          </button>
        </div>
      </form>


      {searchRes && (
        <div className="flex flex-col gap-6">
          <div className="flex justify-between items-center text-sm text-slate-400 px-2">
            <span>Retrieval Latency: <strong className="text-blue-400 font-mono text-base">{searchRes.latency_ms} ms</strong></span>
          </div>

          {!searchRes.confident && (
            <motion.div initial={{opacity:0, y:-10}} animate={{opacity:1, y:0}} className="bg-red-950/40 border border-red-500/30 p-4 rounded-xl text-red-300 flex items-center gap-3 shadow-sm">
              <AlertTriangle className="flex-shrink-0 text-red-400" />
              <span>No high-confidence match found for this query. The results below did not meet the relevance threshold.</span>
            </motion.div>
          )}

          <div className="flex flex-col gap-4">
            {searchRes.results.map((r, i) => (
              <motion.div 
                key={r.id} 
                initial={{opacity:0, y:10}} 
                animate={{opacity:1, y:0}} 
                transition={{delay: i * 0.05}}
                className="bg-slate-800/80 backdrop-blur-sm border border-slate-700/50 p-6 rounded-xl shadow-lg flex flex-col gap-4 hover:border-slate-600 transition-colors"
              >
                <div className="flex justify-between items-start">
                  <span className="text-xs text-slate-500 uppercase font-bold tracking-wider">Rank #{i+1}</span>
                  <div className="flex gap-2">
                    <span className={clsx("px-2.5 py-1 rounded-md text-xs font-bold uppercase shadow-sm", {
                      "bg-green-500/20 text-green-400 border border-green-500/20": r.confidence === "high",
                      "bg-yellow-500/20 text-yellow-400 border border-yellow-500/20": r.confidence === "medium",
                      "bg-red-500/20 text-red-400 border border-red-500/20": r.confidence === "low"
                    })}>
                      {r.confidence}
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-slate-700/50 border border-slate-600/50 text-slate-300 text-xs font-medium">Rel Score: {r.relevance_score.toFixed(2)}</span>
                  </div>
                </div>
                <p className="text-slate-200 leading-relaxed text-[15px]">{r.text}</p>
                
                <div className="flex justify-between items-center mt-2 border-t border-slate-700/50 pt-4">
                  <div className="text-xs text-slate-400 max-w-[70%]">
                    <span className="truncate block">Matched Concepts: {r.matched_concepts.length > 0 ? r.matched_concepts.join(", ") : "None"}</span>
                  </div>
                  <div className="flex gap-1.5">
                    <button onClick={() => handleFeedback(r.id, 'relevant')} title="Mark Relevant" className="p-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-400 hover:text-green-400 transition-colors">
                      <ThumbsUp size={16} />
                    </button>
                    <button onClick={() => handleFeedback(r.id, 'not_relevant')} title="Mark Not Relevant (Exclude in future)" className="p-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-400 hover:text-red-400 transition-colors">
                      <ThumbsDown size={16} />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-6 flex justify-center">
            <button onClick={handleSynthesize} disabled={synLoading || searchRes.results.length === 0} className="bg-indigo-600 hover:bg-indigo-500 px-8 py-3 rounded-xl text-white flex items-center gap-3 shadow-lg disabled:opacity-50 transition-colors font-medium">
              {synLoading ? <Loader2 className="animate-spin" size={20} /> : "Synthesize Answer"}
            </button>
          </div>

          <AnimatePresence>
            {synRes && (
              <motion.div 
                initial={{opacity:0, scale:0.98, y: 10}} 
                animate={{opacity:1, scale:1, y:0}} 
                className="mt-4 bg-slate-800/90 border border-slate-700 rounded-xl shadow-2xl overflow-hidden backdrop-blur-sm"
              >
                {!synRes.grounded && (
                  <div className="bg-orange-950/80 border-b border-orange-500/30 p-4 text-orange-300 flex items-center gap-3">
                    <AlertTriangle className="flex-shrink-0 text-orange-400" />
                    <span className="text-sm"><strong>⚠ Ungrounded Answer:</strong> {synRes.warning}</span>
                  </div>
                )}
                {synRes.grounded && (
                  <div className="bg-emerald-950/60 border-b border-emerald-500/30 p-4 text-emerald-400 flex items-center gap-3">
                    <CheckCircle className="flex-shrink-0" />
                    <span className="text-sm"><strong>Grounded Answer:</strong> Based strictly on retrieved evidence.</span>
                  </div>
                )}
                <div className="p-8 text-slate-200 whitespace-pre-wrap leading-relaxed text-[15px]">
                  {synRes.answer}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}

export default App
