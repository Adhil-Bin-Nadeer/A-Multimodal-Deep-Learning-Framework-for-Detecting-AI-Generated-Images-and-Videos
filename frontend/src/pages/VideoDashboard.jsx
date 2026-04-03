import { useCallback, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

const INITIAL_STEPS = [
  { id: 'step1', icon: '🎬', label: '1. Frame Sampling', status: 'waiting' },
  { id: 'step2', icon: '🧠', label: '2. Video AI Analysis', status: 'waiting' },
]

function stepClass(status) {
  switch (status) {
    case 'processing': return 'step-processing'
    case 'complete': return 'step-completed'
    case 'flagged': return 'step-flagged'
    default: return 'step-waiting'
  }
}

function stepLabel(status) {
  switch (status) {
    case 'processing': return 'Running...'
    case 'complete': return 'Complete'
    case 'flagged': return 'Flagged'
    default: return 'Waiting'
  }
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function parseVideoLabel(result) {
  const rawLabel = result?.label ? String(result.label).toUpperCase() : ''
  if (rawLabel === 'FAKE' || rawLabel === 'REAL') {
    return rawLabel
  }

  const text = String(result?.result || '')
  const match = text.match(/(REAL|FAKE)/i)
  return match ? match[1].toUpperCase() : 'UNKNOWN'
}

export default function VideoDashboard() {
  const navigate = useNavigate()
  const fileInputRef = useRef(null)
  const consoleRef = useRef(null)

  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadLabel, setUploadLabel] = useState('Drag & Drop Video Here')
  const [fileInfo, setFileInfo] = useState('(MP4, AVI, MOV)')
  const [scanning, setScanning] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [steps, setSteps] = useState(INITIAL_STEPS)
  const [logs, setLogs] = useState(['> System ready.', '> Awaiting video input...'])

  const appendLog = useCallback((message) => {
    setLogs(prev => {
      const next = [...prev, `> ${message}`]
      setTimeout(() => {
        if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight
      }, 0)
      return next
    })
  }, [])

  const setStep = useCallback((id, status) => {
    setSteps(prev => prev.map(step => (step.id === id ? { ...step, status } : step)))
  }, [])

  const selectFile = useCallback((file) => {
    setSelectedFile(file)
    setUploadLabel(file.name)
    setFileInfo(`${(file.size / 1024 / 1024).toFixed(1)} MB - Click to change`)
    appendLog(`Video selected: ${file.name}`)
  }, [appendLog])

  const handleDrop = useCallback((event) => {
    event.preventDefault()
    setIsDragOver(false)
    const file = event.dataTransfer.files[0]
    if (file) selectFile(file)
  }, [selectFile])

  const handleInputChange = (event) => {
    if (event.target.files[0]) selectFile(event.target.files[0])
  }

  async function startScan() {
    if (!selectedFile || scanning) return

    setScanning(true)
    setUploadLabel(`Analyzing: ${selectedFile.name}...`)
    setFileInfo('Processing...')

    appendLog(`Video received: ${selectedFile.name}`)
    appendLog(`File type: ${selectedFile.type || 'Unknown'}`)

    setStep('step1', 'processing')
    appendLog('[STEP 1] Sampling frames from uploaded video...')

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('/api/analyze_video', { method: 'POST', body: formData })
      const result = await response.json()

      if (!result.success) {
        appendLog(`Error: ${result.error || 'Video analysis failed'}`)
        setScanning(false)
        return
      }

      const framesAnalyzed = result?.explainability?.metrics?.frames_analyzed
      setStep('step1', 'complete')
      appendLog(`[STEP 1] Frame sampling complete${Number.isFinite(framesAnalyzed) ? ` (${framesAnalyzed} frames analyzed)` : ''}.`)

      setStep('step2', 'processing')
      appendLog('[STEP 2] Running deepfake video detector...')
      await delay(400)

      const label = parseVideoLabel(result)
      const confidence = Number.isFinite(result?.confidence) ? Number(result.confidence) : 0.0
      const isAI = label === 'FAKE'

      setStep('step2', isAI ? 'flagged' : 'complete')
      appendLog(`[STEP 2] Result: ${label} (${confidence.toFixed(1)}% confidence)`)

      appendLog('')
      appendLog(`=== FINAL VERDICT: ${isAI ? 'AI Generated Video' : 'Authentic Video'} ===`)
      appendLog(`Confidence: ${confidence.toFixed(1)}%`)

      sessionStorage.setItem('videoAnalysisResult', JSON.stringify({
        ...result,
        filename: selectedFile.name,
        parsed_label: label,
      }))

      await delay(700)
      navigate('/video-report')
    } catch (error) {
      appendLog(`Network error: ${error.message}`)
      setScanning(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-4xl text-center">
        <h1 className="gradient-text text-4xl font-extrabold mb-2 tracking-tight">
          Video Scanner Dashboard
        </h1>
        <p className="text-slate-400 mb-6 tracking-wide">
          Upload a clip for deepfake video analysis
        </p>

        <div
          className={`glass-card h-48 flex flex-col items-center justify-center mb-6 cursor-pointer
                      border-2 border-dashed transition-all duration-300
                      ${isDragOver
                        ? 'border-neon-blue bg-neon-blue/5'
                        : 'border-white/10 hover:border-neon-blue hover:bg-neon-blue/5'}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={event => { event.preventDefault(); setIsDragOver(true) }}
          onDragEnter={event => { event.preventDefault(); setIsDragOver(true) }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <span className="text-4xl mb-2">🎞️</span>
          <h3 className="font-semibold text-base">{uploadLabel}</h3>
          <p className="text-slate-400 text-xs mt-1">{fileInfo}</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.avi,.mov"
            className="hidden"
            onChange={handleInputChange}
          />
        </div>

        <div className="flex items-center justify-center gap-3 mb-8">
          <button
            className="btn-neon px-12 py-3 text-base"
            disabled={!selectedFile || scanning}
            onClick={startScan}
          >
            {scanning ? 'Scanning...' : 'Start Scan'}
          </button>
          <button
            className="btn-neon px-8 py-3 text-base"
            onClick={() => navigate('/')}
          >
            Back
          </button>
        </div>

        <h4 className="text-left mb-3 text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Scanning Progress
        </h4>
        <div className="grid grid-cols-2 gap-4 mb-6">
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

        <div ref={consoleRef} className="console-log h-40">
          {logs.map((line, index) => (
            <div key={index}>{line}</div>
          ))}
        </div>
      </div>
    </div>
  )
}
