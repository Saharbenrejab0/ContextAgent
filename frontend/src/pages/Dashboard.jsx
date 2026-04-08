import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis } from 'recharts'
import '../styles/dashboard.css'

const TT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#1a1a24', border: '1px solid #252535', borderRadius: 8, padding: '8px 12px', fontSize: 12, color: '#e2e2f0' }}>
      {payload.map((p,i) => <div key={i}>{p.name}: <b>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</b></div>)}
    </div>
  )
}

export default function Dashboard({ ctx }) {
  const [stats, setStats] = useState(null)
  const [docs,  setDocs]  = useState([])

  useEffect(() => {
    axios.get(`/api/stats/${ctx.sessionId}`).then(r => setStats(r.data))
    axios.get('/api/documents').then(r => setDocs(r.data.documents))
    const iv = setInterval(() => {
      axios.get(`/api/stats/${ctx.sessionId}`).then(r => setStats(r.data))
    }, 5000)
    return () => clearInterval(iv)
  }, [])

  const tokenData = stats?.token_history?.map((t, i) => ({ turn: i+1, tokens: t })) || []
  const distData  = stats?.distance_hist?.map((d, i) => ({ i: i+1, distance: +d.toFixed(3) })) || []
  const docData   = docs.map(d => ({ name: d.name.slice(0,12), size: d.size_kb }))

  const cards = [
    { label: 'Total tokens',  value: stats?.total_tokens?.toLocaleString() ?? '0',     cls: '' },
    { label: 'Total cost',    value: '$'+(stats?.total_cost?.toFixed(5) ?? '0.00000'),  cls: 'green' },
    { label: 'Turns',         value: stats?.turns ?? 0,                                  cls: 'purple' },
    { label: 'Avg latency',   value: (stats?.avg_latency ?? 0)+'s',                     cls: 'amber' },
    { label: 'Avg distance',  value: stats?.avg_distance?.toFixed(3) ?? '—',            cls: '' },
    { label: 'Chunks indexed',value: stats?.total_chunks ?? 0,                           cls: '' },
  ]

  return (
    <div className="dash-page">
      <div className="topbar"><span className="topbar-title">Dashboard</span></div>
      <div className="dash-body">
        <div className="cards-row">
          {cards.map(c => (
            <div key={c.label} className="stat-card">
              <div className={'stat-value '+c.cls}>{c.value}</div>
              <div className="stat-label">{c.label}</div>
            </div>
          ))}
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-title">Tokens per turn</div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={tokenData}>
                <XAxis dataKey="turn" stroke="#505070" tick={{ fontSize: 10 }}/>
                <YAxis stroke="#505070" tick={{ fontSize: 10 }}/>
                <Tooltip content={<TT/>}/>
                <Line type="monotone" dataKey="tokens" stroke="#7c6fe0" strokeWidth={2} dot={{ r: 3, fill: '#7c6fe0' }}/>
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-title">Retrieval distances</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={distData}>
                <XAxis dataKey="i" stroke="#505070" tick={{ fontSize: 10 }}/>
                <YAxis domain={[0,1]} stroke="#505070" tick={{ fontSize: 10 }}/>
                <Tooltip content={<TT/>}/>
                <Bar dataKey="distance" fill="#2dd4a0" radius={[3,3,0,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-title">Documents by size (KB)</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={docData} layout="vertical">
                <XAxis type="number" stroke="#505070" tick={{ fontSize: 10 }}/>
                <YAxis dataKey="name" type="category" stroke="#505070" tick={{ fontSize: 10 }} width={80}/>
                <Tooltip content={<TT/>}/>
                <Bar dataKey="size" fill="#a09be8" radius={[0,3,3,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-title">System health</div>
            <ResponsiveContainer width="100%" height={180}>
              <RadarChart data={[
                { metric: 'Retrieval',  value: stats ? Math.round((1-stats.avg_distance)*100) : 0 },
                { metric: 'Speed',      value: stats ? Math.round(Math.max(0, 100-(stats.avg_latency*20))) : 0 },
                { metric: 'Coverage',   value: stats ? Math.min(100, Math.round(stats.total_chunks/7)) : 0 },
                { metric: 'Usage',      value: stats ? Math.min(100, stats.turns*10) : 0 },
                { metric: 'Economy',    value: stats ? Math.round(Math.max(0,100-(stats.total_cost*10000))) : 100 },
              ]}>
                <PolarGrid stroke="#252535"/>
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: '#8888a8' }}/>
                <Radar dataKey="value" stroke="#7c6fe0" fill="#7c6fe0" fillOpacity={0.2}/>
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
