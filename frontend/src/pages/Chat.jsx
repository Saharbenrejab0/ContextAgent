import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import '../styles/chat.css'

function ChunkPanel({ chunks }) {
  if (!chunks.length) return (
    <div className="chunk-empty">No query yet — chunks will appear here</div>
  )
  return (
    <div className="chunk-list">
      {chunks.map((c, i) => {
        const scoreClass = c.distance < 0.45 ? 'hi' : c.distance < 0.65 ? 'md' : 'lo'
        const pct = Math.round(Math.min(c.distance * 100, 100))
        return (
          <div key={i} className="chunk-card">
            <div className="chunk-top">
              <span className={'score score-' + scoreClass}>{c.distance.toFixed(3)}</span>
              <span className="chunk-src">{c.source} · #{c.chunk_index}</span>
            </div>
            <div className="chunk-txt">{c.text.slice(0, 240)}...</div>
            <div className="chunk-bar-bg">
              <div className={'chunk-bar-fill ' + scoreClass} style={{ width: pct + '%' }}/>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default function Chat({ ctx }) {
  const [messages,  setMessages]  = useState([])
  const [input,     setInput]     = useState('')
  const [loading,   setLoading]   = useState(false)
  const [lastChunks,setChunks]    = useState([])
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return

    setInput('')
    setMessages(m => [...m, { role: 'user', content: q }])
    setLoading(true)

    try {
      const res = await fetch('/api/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id:    ctx.sessionId,
          question:      q,
          top_k:         ctx.settings.topK,
          temperature:   ctx.settings.temperature,
          memory_window: ctx.settings.memoryWindow,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        setMessages(m => [...m, { role: 'error', content: err.detail || 'API error' }])
        setLoading(false)
        return
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let   buffer  = ''
      let   msgIdx  = -1

      setMessages(m => {
        msgIdx = m.length
        return [...m, { role: 'assistant', content: '', sources: [], tokens: 0, latency: 0 }]
      })

      const startTime = Date.now()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (!data) continue

          if (data.startsWith('[DONE]')) {
            try {
              const meta    = JSON.parse(data.slice(6))
              const latency = ((Date.now() - startTime) / 1000).toFixed(2)
              setMessages(m => m.map((msg, i) =>
                i === msgIdx
                  ? { ...msg, sources: meta.sources || [], tokens: meta.tokens?.total || 0, latency }
                  : msg
              ))
              if (meta.chunks) setChunks(meta.chunks)
            } catch(e) {
              console.error('Failed to parse DONE metadata:', e)
            }
          } else {
            try {
              const token = JSON.parse(data)
              setMessages(m => m.map((msg, i) =>
                i === msgIdx ? { ...msg, content: msg.content + token } : msg
              ))
            } catch {
              setMessages(m => m.map((msg, i) =>
                i === msgIdx ? { ...msg, content: msg.content + data } : msg
              ))
            }
          }
        }
      }
    } catch (e) {
      setMessages(m => [...m, { role: 'error', content: 'Stream error: ' + e.message }])
    }

    setLoading(false)
  }

  const reset = async () => {
    await fetch('/api/reset', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: ctx.sessionId }),
    })
    setMessages([])
    setChunks([])
  }

  return (
    <div className="chat-page">
      <div className="topbar">
        <span className="topbar-title">Chat</span>
        <div className="topbar-right">
          <button
            className={'tbtn' + (ctx.debugMode ? ' on' : '')}
            onClick={() => ctx.setDebugMode(d => !d)}
          >
            Debug
          </button>
          <button className="tbtn" onClick={reset}>Reset</button>
        </div>
      </div>

      <div className="chat-body">
        <div className="chat-main">
          <div className="chat-msgs">
            {messages.length === 0 && (
              <div className="chat-empty">
                <div className="chat-empty-icon">◈</div>
                <div className="chat-empty-title">Ask your documents</div>
                <div className="chat-empty-sub">
                  Answers grounded in your content — always cited
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={'msg-row ' + m.role}>
                <div className={'avatar av-' + (m.role === 'user' ? 'u' : 'a')}>
                  {m.role === 'user' ? 'S' : 'CA'}
                </div>
                <div className="msg-content">
                  <div className={'bubble bubble-' + m.role}>
                    {m.role === 'assistant'
                      ? <ReactMarkdown>{m.content || '...'}</ReactMarkdown>
                      : m.content
                    }
                  </div>

                  {m.role === 'assistant' && m.tokens > 0 && (
                    <div className="msg-meta">
                      {m.sources?.map(s => (
                        <span key={s} className="chip">{s}</span>
                      ))}
                      <span className="meta-txt">
                        {m.tokens} tokens · {m.latency}s
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && messages[messages.length - 1]?.role !== 'assistant' && (
              <div className="msg-row assistant">
                <div className="avatar av-a">CA</div>
                <div className="bubble bubble-assistant">
                  <div className="typing">
                    <span/><span/><span/>
                  </div>
                </div>
              </div>
            )}

            <div ref={bottomRef}/>
          </div>

          <div className="input-area">
            <div className="input-wrap">
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
                placeholder="Ask a question about your documents..."
                disabled={loading}
              />
            </div>
            <button className="send-btn" onClick={send} disabled={loading}>
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                <path
                  d="M2 7.5h11M8 2.5l5 5-5 5"
                  stroke="white"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>

        {ctx.debugMode && (
          <div className="debug-panel">
            <div className="debug-header">
              <span className="debug-title">Retrieved chunks</span>
              <span className="debug-sub">{lastChunks.length} results</span>
            </div>
            <ChunkPanel chunks={lastChunks}/>
          </div>
        )}
      </div>
    </div>
  )
}