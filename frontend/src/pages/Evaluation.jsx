import React, { useState } from 'react'
import axios from 'axios'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts'
import '../styles/evaluation.css'

const DEFAULT_QUESTIONS = [
  "What is a Python class?",
  "How does error handling work in Python?",
  "What is retrieval augmented generation?",
  "What is the attention mechanism in transformers?",
  "How do Python modules work?",
  "What are data structures in Python?",
  "How does inheritance work in Python?",
  "What is the difference between a list and a tuple?",
  "How does LangChain work?",
  "What are the key contributions of the RAG paper?",
]

export default function Evaluation({ ctx }) {
  const [questions, setQuestions] = useState(DEFAULT_QUESTIONS.join('\n'))
  const [results,   setResults]   = useState([])
  const [running,   setRunning]   = useState(false)

  const run = async () => {
    const qs = questions.split('\n').map(q => q.trim()).filter(Boolean)
    if (!qs.length) return
    setRunning(true)
    const { data } = await axios.post('/api/evaluate', { questions: qs })
    setResults(data.results)
    setRunning(false)
  }

  const avgDist  = results.length ? (results.reduce((a,r) => a+r.avg_distance,0)/results.length).toFixed(3) : '—'
  const avgChunks= results.length ? (results.reduce((a,r) => a+r.chunks_found,0)/results.length).toFixed(1) : '—'
  const avgLat   = results.length ? (results.reduce((a,r) => a+r.latency,0)/results.length).toFixed(2) : '—'
  const coverage = results.length ? Math.round(results.filter(r=>r.chunks_found>0).length/results.length*100) : 0

  const radarData = [
    { metric: 'Coverage',  value: coverage },
    { metric: 'Relevance', value: results.length ? Math.round((1-parseFloat(avgDist))*100) : 0 },
    { metric: 'Speed',     value: results.length ? Math.round(Math.max(0,100-(parseFloat(avgLat)*30))) : 0 },
    { metric: 'Depth',     value: results.length ? Math.min(100,Math.round(parseFloat(avgChunks)/5*100)) : 0 },
  ]

  return (
    <div className="eval-page">
      <div className="topbar">
        <span className="topbar-title">Evaluation</span>
        <div className="topbar-right">
          <button className="tbtn on" onClick={run} disabled={running}>
            {running ? 'Running...' : 'Run evaluation'}
          </button>
        </div>
      </div>
      <div className="eval-body">
        <div className="eval-left">
          <div className="eval-card">
            <div className="eval-card-title">Test questions</div>
            <textarea
              className="eval-textarea"
              value={questions}
              onChange={e => setQuestions(e.target.value)}
              rows={12}
              placeholder="One question per line..."
            />
          </div>
          {results.length > 0 && (
            <div className="results-table">
              <div className="rtable-header">
                <span>Question</span><span>Chunks</span><span>Dist</span><span>Lat</span>
              </div>
              {results.map((r,i) => (
                <div key={i} className="rtable-row">
                  <span className="rq">{r.question.slice(0,40)}...</span>
                  <span className={'rc ' + (r.chunks_found > 0 ? 'g' : 'r')}>{r.chunks_found}</span>
                  <span className={'rd ' + (r.avg_distance < 0.5 ? 'g' : r.avg_distance < 0.7 ? 'a' : 'r')}>{r.avg_distance.toFixed(3)}</span>
                  <span className="rl">{r.latency.toFixed(2)}s</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="eval-right">
          <div className="eval-card">
            <div className="eval-card-title">Summary metrics</div>
            <div className="metrics-grid">
              <div className="metric-box"><div className="metric-val">{coverage}%</div><div className="metric-lbl">Coverage</div></div>
              <div className="metric-box"><div className="metric-val">{avgDist}</div><div className="metric-lbl">Avg distance</div></div>
              <div className="metric-box"><div className="metric-val">{avgChunks}</div><div className="metric-lbl">Avg chunks</div></div>
              <div className="metric-box"><div className="metric-val">{avgLat}s</div><div className="metric-lbl">Avg latency</div></div>
            </div>
          </div>
          <div className="eval-card">
            <div className="eval-card-title">Quality radar</div>
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#252535"/>
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#8888a8' }}/>
                <Tooltip formatter={v => v+'%'}/>
                <Radar dataKey="value" stroke="#7c6fe0" fill="#7c6fe0" fillOpacity={0.25}/>
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
