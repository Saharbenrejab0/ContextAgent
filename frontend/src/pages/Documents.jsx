import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import { useDropzone } from 'react-dropzone'
import '../styles/documents.css'

export default function Documents() {
  const [docs,     setDocs]     = useState([])
  const [chunks,   setChunks]   = useState(0)
  const [uploading,setUploading]= useState(false)
  const [progress, setProgress] = useState('')
  const [reingesting, setReing] = useState(false)

  const load = () => axios.get('/api/documents').then(r => {
    setDocs(r.data.documents); setChunks(r.data.total_chunks)
  })

  useEffect(() => { load() }, [])

  const onDrop = useCallback(async files => {
    if (!files.length) return
    setUploading(true)
    for (const file of files) {
      setProgress(`Ingesting ${file.name}...`)
      const fd = new FormData()
      fd.append('file', file)
      try {
        await axios.post('/api/ingest', fd)
      } catch(e) {
        setProgress('Error: ' + (e.response?.data?.detail || e.message))
      }
    }
    setProgress('')
    setUploading(false)
    load()
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'application/pdf': [], 'text/plain': [], 'text/markdown': [] }
  })

  const reingest = async () => {
    setReing(true)
    await axios.post('/api/reingest')
    setReing(false)
    load()
  }

  const badge = ext => {
    if (ext === 'pdf') return <span className="badge badge-pdf">PDF</span>
    if (ext === 'md')  return <span className="badge badge-md">MD</span>
    return <span className="badge badge-txt">TXT</span>
  }

  return (
    <div className="docs-page">
      <div className="topbar">
        <span className="topbar-title">Documents</span>
        <div className="topbar-right">
          <button className="tbtn" onClick={reingest} disabled={reingesting}>
            {reingesting ? 'Re-ingesting...' : 'Re-ingest all'}
          </button>
        </div>
      </div>

      <div className="docs-body">
        <div {...getRootProps()} className={'dropzone' + (isDragActive ? ' active' : '')}>
          <input {...getInputProps()}/>
          <div className="drop-icon">↑</div>
          <div className="drop-title">{isDragActive ? 'Drop files here' : 'Drag & drop documents'}</div>
          <div className="drop-sub">PDF, TXT, MD — or click to browse</div>
          {uploading && <div className="drop-progress">{progress}</div>}
        </div>

        <div className="docs-stats">
          <div className="stat-card"><div className="stat-value">{docs.length}</div><div className="stat-label">Documents</div></div>
          <div className="stat-card"><div className="stat-value purple">{chunks}</div><div className="stat-label">Total chunks</div></div>
        </div>

        <div className="docs-table">
          <div className="table-header">
            <span>Name</span><span>Type</span><span>Size</span>
          </div>
          {docs.map(d => (
            <div key={d.name} className="table-row">
              <span className="td-name">{d.name}</span>
              <span>{badge(d.type)}</span>
              <span className="td-size">{d.size_kb} KB</span>
            </div>
          ))}
          {docs.length === 0 && <div className="table-empty">No documents found in docs/</div>}
        </div>
      </div>
    </div>
  )
}
