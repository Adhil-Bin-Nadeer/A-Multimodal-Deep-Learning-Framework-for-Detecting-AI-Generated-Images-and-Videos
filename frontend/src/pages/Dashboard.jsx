import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'

const INITIAL_STEPS = [
  { id: 'step1', icon: '🔒', label: '1. C2PA Metadata',       status: 'waiting' },
  { id: 'step2', icon: '🌊', label: '2. SynthID Watermark',   status: 'waiting' },
  { id: 'step3', icon: '🧠', label: '3. Deep Learning (ResNet)', status: 'waiting' },
]

function stepClass(status) {
  switch (status) {
    case 'processing': return 'step-processing'
    case 'complete':   return 'step-completed'
    case 'flagged':    return 'step-flagged'
    default:           return 'step-waiting'
  }
}

function stepLabel(status) {
  switch (status) {
    case 'processing': return 'Scanning...'
    case 'complete':   return 'Verified'
    case 'flagged':    return 'Flagged'
    default:           return 'Waiting'
  }
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms))
}

export default function Dashboard() {
  const navigate    = useNavigate()
  const fileInputRef= useRef(null)
  const consoleRef  = useRef(null)

  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadLabel,  setUploadLabel]  = useState('Drag & Drop File Here')
  const [fileInfo,     setFileInfo]     = useState('(JPG, PNG, WEBP)')
  const [scanning,     setScanning]     = useState(false)
  const [isDragOver,   setIsDragOver]   = useState(false)
  const [steps,        setSteps]        = useState(INITIAL_STEPS)
  const [logs,         setLogs]         = useState(['> System ready.', '> Awaiting file input...'])

  // ── helpers ──────────────────────────────────────────────
  const appendLog = useCallback((msg) => {
    setLogs(prev => {
      const next = [...prev, `> ${msg}`]
      setTimeout(() => {
        if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight
      }, 0)
      return next
    })
  }, [])

  const setStep = useCallback((id, status) => {
    setSteps(prev => prev.map(s => s.id === id ? { ...s, status } : s))
  }, [])

  // ── file selection ────────────────────────────────────────
  const selectFile = useCallback((file) => {
    setSelectedFile(file)
    setUploadLabel(file.name)
    setFileInfo(`${(file.size / 1024).toFixed(1)} KB – Click to change`)
    appendLog(`File selected: ${file.name}`)
  }, [appendLog])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) selectFile(file)
  }, [selectFile])

  const handleInputChange = (e) => {
    if (e.target.files[0]) selectFile(e.target.files[0])
  }

  // ── scan ─────────────────────────────────────────────────
  async function startScan() {
    if (!selectedFile || scanning) return
    setScanning(true)
    setUploadLabel(`Analyzing: ${selectedFile.name}...`)
    setFileInfo('Processing...')

    appendLog(`File received: ${selectedFile.name}`)
    appendLog(`File type: ${selectedFile.type || 'Unknown'}`)
    appendLog('Initiating 3-layer forensic analysis...')

    setStep('step1', 'processing')
    appendLog('[LAYER 1] Extracting C2PA provenance data...')

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('/api/analyze', { method: 'POST', body: formData })
      const result   = await response.json()

      if (!result.success) {
        appendLog(`❌ Error: ${result.error}`)
        setScanning(false)
        return
      }

      // Layer 1 – C2PA
      const c2pa = result.layers.c2pa
      setStep('step1', 'complete')
      if (c2pa?.c2pa_present) {
        appendLog(`[LAYER 1] C2PA metadata FOUND! Issuer: ${c2pa.issuer || 'Unknown'}`)
        appendLog(`[LAYER 1] AI Generated Flag: ${c2pa.ai_generated ? 'YES' : 'NO'}`)
      } else {
        appendLog('[LAYER 1] No C2PA signature found. Proceeding to next layer...')
      }

      // Layer 2 – SynthID
      setStep('step2', 'processing')
      appendLog('[LAYER 2] SynthID check...')
      await delay(500)
      const synthid = result.layers.synthid
      if (synthid?.status === 'skipped') appendLog(`[LAYER 2] ${synthid.reason}`)
      setStep('step2', 'complete')

      // Layer 3 – AI Model
      setStep('step3', 'processing')
      appendLog('[LAYER 3] Running AI detection model...')
      await delay(500)
      const aiModel = result.layers.ai_model
      if (aiModel?.status === 'complete') {
        setStep('step3', 'complete')
        appendLog(`[LAYER 3] Result: ${aiModel.label} (${aiModel.confidence.toFixed(1)}% confidence)`)
      } else if (aiModel?.status === 'skipped') {
        setStep('step3', 'complete')
        appendLog(`[LAYER 3] Skipped – ${aiModel.reason}`)
      } else {
        appendLog(`[LAYER 3] ${aiModel?.error || 'Model unavailable'}`)
      }

      appendLog('')
      appendLog(`=== FINAL VERDICT: ${result.final_verdict} ===`)
      appendLog(`Confidence: ${result.confidence.toFixed(1)}%`)

      sessionStorage.setItem('analysisResult', JSON.stringify(result))
      appendLog('Generating forensic report...')
      await delay(1000)
      navigate('/report')

    } catch (err) {
      appendLog(`❌ Network error: ${err.message}`)
      setScanning(false)
    }
  }

  // ── render ────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-4xl text-center">
        <h1 className="gradient-text text-4xl font-extrabold mb-2 tracking-tight">
          Scanner Dashboard
        </h1>
        <p className="text-slate-400 mb-6 tracking-wide">
          Initialize forensic analysis protocols
        </p>

        {/* Upload zone */}
        <div
          className={`glass-card h-48 flex flex-col items-center justify-center mb-6 cursor-pointer
                      border-2 border-dashed transition-all duration-300
                      ${isDragOver
                        ? 'border-neon-blue bg-neon-blue/5'
                        : 'border-white/10 hover:border-neon-blue hover:bg-neon-blue/5'}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setIsDragOver(true) }}
          onDragEnter={e => { e.preventDefault(); setIsDragOver(true) }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <span className="text-4xl mb-2">⬆️</span>
          <h3 className="font-semibold text-base">{uploadLabel}</h3>
          <p className="text-slate-400 text-xs mt-1">{fileInfo}</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp"
            className="hidden"
            onChange={handleInputChange}
          />
        </div>

        {/* Scan button */}
        <button
          className="btn-neon px-12 py-3 text-base mb-8"
          disabled={!selectedFile || scanning}
          onClick={startScan}
        >
          {scanning ? 'Scanning...' : 'Start Scan'}
        </button>

        {/* Progress steps */}
        <h4 className="text-left mb-3 text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Scanning Progress
        </h4>
        <div className="grid grid-cols-3 gap-4 mb-6">
          {steps.map(step => (
            <div
              key={step.id}
              className={`glass-card p-5 text-center border transition-all duration-300 ${stepClass(step.status)}`}
            >
              <span className="block text-3xl mb-2">{step.icon}</span>
              <div className="text-sm font-medium">{step.label}</div>
              <div className="text-xs mt-1 opacity-80">{stepLabel(step.status)}</div>
            </div>
          ))}
        </div>

        {/* Console */}
        <div ref={consoleRef} className="console-log">
          {logs.map((line, i) => (
            <div key={i} dangerouslySetInnerHTML={{ __html: line }} />
          ))}
        </div>
      </div>
    </div>
  )
}
