import React, { useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useNavigate } from 'react-router-dom'
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

function NavBar() {
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
    </nav>
  )
}

export default function App() {
  const [sessionId] = useState(() => 'sess_' + Math.random().toString(36).slice(2))
  const [settings, setSettings] = useState({ topK: 5, temperature: 0.1, memoryWindow: 6 })
  const [debugMode, setDebugMode] = useState(true)

  const ctx = { sessionId, settings, setSettings, debugMode, setDebugMode }

  return (
    <BrowserRouter>
      <div style={{ display: 'flex', height: '100vh', width: '100%' }}>
        <NavBar/>
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
