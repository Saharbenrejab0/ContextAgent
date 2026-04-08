import React, { useState, useEffect } from 'react'
import axios from 'axios'
import '../styles/sidebar.css'

export default function Sidebar({ ctx }) {
  const [docs, setDocs]       = useState([])
  const [chunks, setChunks]   = useState(0)
  const [stats, setStats]     = useState(null)

  useEffect(() => {
    axios.get('/api/documents').then(r => {
      setDocs(r.data.documents)
      setChunks(r.data.total_chunks)
    })
  }, [])

  useEffect(() => {
    if (!ctx.sessionId) return
    const interval = setInterval(() => {
      axios.get(`/api/stats/${ctx.sessionId}`).then(r => setStats(r.data))
    }, 3000)
    return () => clearInterval(interval)
  }, [ctx.sessionId])

  const badge = ext => {
    if (ext === 'pdf') return <span className="badge badge-pdf">PDF</span>
    if (ext === 'md')  return <span className="badge badge-md">MD</span>
    return <span className="badge badge-txt">TXT</span>
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">ContextAgent</div>
        <div className="sidebar-sub">Document assistant · v1</div>
        <div className={'status-row ' + (chunks > 0 ? 'ready' : 'empty')}>
          <span className="status-dot"/>
          <span>{chunks > 0 ? `${chunks} chunks ready` : 'No chunks indexed'}</span>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sec-label">Documents</div>
        <div className="doc-list">
          {docs.map(d => (
            <div key={d.name} className="doc-item">
              <div className="doc-row">
                <span className="doc-name" title={d.name}>{d.name}</span>
                {badge(d.type)}
              </div>
              <div className="doc-meta">{d.size_kb} KB</div>
            </div>
          ))}
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sec-label">Session</div>
        <div className="stats-grid">
          <div className="stat-card"><div className="stat-value purple">{stats?.turns ?? 0}</div><div className="stat-label">Turns</div></div>
          <div className="stat-card"><div className="stat-value">{stats ? (stats.total_tokens >= 1000 ? (stats.total_tokens/1000).toFixed(1)+'k' : stats.total_tokens) : 0}</div><div className="stat-label">Tokens</div></div>
          <div className="stat-card"><div className="stat-value">{chunks}</div><div className="stat-label">Chunks</div></div>
          <div className="stat-card"><div className="stat-value green">${stats?.total_cost?.toFixed(4) ?? '0.00'}</div><div className="stat-label">Cost</div></div>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sec-label">Settings</div>
        <div className="slider-group">
          {[
            { label: 'Top K', key: 'topK', min: 1, max: 10 },
            { label: 'Memory', key: 'memoryWindow', min: 1, max: 12 },
          ].map(s => (
            <div key={s.key} className="slider-row">
              <div className="slider-header">
                <span className="slider-label">{s.label}</span>
                <span className="slider-val">{ctx.settings[s.key]}</span>
              </div>
              <input type="range" min={s.min} max={s.max} step="1"
                value={ctx.settings[s.key]}
                onChange={e => ctx.setSettings(p => ({...p, [s.key]: +e.target.value}))}/>
            </div>
          ))}
          <div className="slider-row">
            <div className="slider-header">
              <span className="slider-label">Temperature</span>
              <span className="slider-val">{ctx.settings.temperature.toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05"
              value={ctx.settings.temperature}
              onChange={e => ctx.setSettings(p => ({...p, temperature: +e.target.value}))}/>
          </div>
        </div>
      </div>
    </aside>
  )
}
