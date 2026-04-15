import React, { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Chat from './pages/Chat'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Evaluation from './pages/Evaluation'
import Sidebar from './components/Sidebar'
import './styles/app.css'

const NAV = [
  { path: '/',          icon: 'chat',  label: 'Chat' },
  { path: '/dashboard', icon: 'dash',  label: 'Dashboard' },
  { path: '/documents', icon: 'docs',  label: 'Documents' },
  { path: '/eval',      icon: 'eval',  label: 'Evaluation' },
]

function NavIcon({ type }) {
  if (type === 'chat') return <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M3 5h12M3 9h12M3 13h8" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>
  if (type === 'dash') return <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="3" y="3" width="5" height="5" rx="1.5" stroke="currentColor" strokeWidth="1.3"/><rect x="10" y="3" width="5" height="5" rx="1.5" stroke="currentColor" strokeWidth="1.3"/><rect x="3" y="10" width="5" height="5" rx="1.5" stroke="currentColor" strokeWidth="1.3"/><rect x="10" y="10" width="5" height="5" rx="1.5" stroke="currentColor" strokeWidth="1.3"/></svg>
  if (type === 'docs') return <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M4 4h10a1 1 0 011 1v8a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.3"/><path d="M7 8l2 2 4-4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>
  if (type === 'eval') return <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M3 14l3-3 3 3 3-5 3 5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/></svg>
  return null
}

function NavBar({ theme, setTheme }) {
  return (
    <nav className="navbar">
      <div className="nav-logo">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <circle cx="9" cy="9" r="6" stroke="#a09be8" strokeWidth="1.2"/>
          <path d="M5 9h8M9 5v8" stroke="#7c6fe0" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      </div>
      {NAV.map(n => (
        <NavLink key={n.path} to={n.path} end={n.path === '/'} className={({isActive}) => 'nav-item' + (isActive ? ' active' : '')}>
          <NavIcon type={n.icon}/>
          <span className="nav-tooltip">{n.label}</span>
        </NavLink>
      ))}
      <div style={{ flex: 1 }}/>
      <button className="theme-toggle" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} title="Toggle theme">
        {theme === 'dark' ? '☀️' : '🌙'}
      </button>
    </nav>
  )
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => 'sess_' + Math.random().toString(36).slice(2))
  const [settings, setSettings]   = useState({ topK: 5, temperature: 0.1, memoryWindow: 6 })
  const [debugMode, setDebugMode] = useState(true)
  const [theme, setTheme]         = useState('dark')
  const [sessions, setSessions]   = useState([])
  const [messages, setMessages]   = useState([])

  // apply theme to root
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  // load session list on mount
  useEffect(() => {
    fetch('/api/sessions')
      .then(r => r.json())
      .then(d => setSessions(d.sessions || []))
  }, [])

  const refreshSessions = () => {
    fetch('/api/sessions')
      .then(r => r.json())
      .then(d => setSessions(d.sessions || []))
  }

  const loadSession = async (sid) => {
    const res  = await fetch(`/api/sessions/${sid}/messages`)
    const data = await res.json()
    const msgs = (data.messages || []).map(m => ({
      role:    m.role,
      content: m.content,
      sources: m.sources ? m.sources.split(',').filter(Boolean) : [],
      tokens:  m.tokens || 0,
      latency: 0,
    }))
    setSessionId(sid)
    setMessages(msgs)
  }

  const ctx = {
    sessionId, setSessionId,
    settings, setSettings,
    debugMode, setDebugMode,
    theme,
    sessions, refreshSessions,
    loadSession,
    messages, setMessages,
  }

  return (
    <BrowserRouter>
      <div className={'app-root ' + theme} style={{ display: 'flex', height: '100vh', width: '100%' }}>
        <NavBar theme={theme} setTheme={setTheme}/>
        <Sidebar ctx={ctx}/>
        <main style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <Routes>
            <Route path="/"          element={<Chat ctx={ctx}/>}/>
            <Route path="/dashboard" element={<Dashboard ctx={ctx}/>}/>
            <Route path="/documents" element={<Documents/>}/>
            <Route path="/eval"      element={<Evaluation ctx={ctx}/>}/>
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}