import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

function probabilityLabel(confidence) {
  if (confidence > 80) return 'Very High'
  if (confidence > 60) return 'High'
  if (confidence > 40) return 'Medium'
  return 'Low'
}

function parseLabel(result) {
  const rawLabel = result?.parsed_label || result?.label
  const normalized = rawLabel ? String(rawLabel).toUpperCase() : ''
  if (normalized === 'FAKE' || normalized === 'REAL') {
    return normalized
  }

  const text = String(result?.result || '')
  const match = text.match(/(REAL|FAKE)/i)
  return match ? match[1].toUpperCase() : 'UNKNOWN'
}

function parseConfidence(result) {
  const value = Number(result?.confidence)
  if (Number.isFinite(value)) {
    return value
  }

  const text = String(result?.result || '')
  const match = text.match(/([\d.]+)%/)
  return match ? Number(match[1]) : 0.0
}

function ScoreCircle({ confidence, isAI }) {
  const color = isAI ? '#ff2a2a' : '#00ff88'
  return (
    <div
      className="score-circle mx-auto"
      style={{
        background: `conic-gradient(${color} ${confidence}%, transparent 0)`,
        boxShadow: `0 0 20px ${color}33`,
      }}
    >
      <div className="score-inner" style={{ color }}>
        {confidence.toFixed(1)}%
      </div>
    </div>
  )
}

export default function VideoReport() {
  const navigate = useNavigate()
  const [result, setResult] = useState(null)

  useEffect(() => {
    const json = sessionStorage.getItem('videoAnalysisResult')
    if (!json) return
    setResult(JSON.parse(json))
    sessionStorage.removeItem('videoAnalysisResult')
  }, [])

  const parsed = useMemo(() => {
    if (!result) return null

    const label = parseLabel(result)
    const confidence = parseConfidence(result)
    const isAI = label === 'FAKE'
    const source = result.source || 'Video AI Detector'
    const framesAnalyzed = result?.explainability?.metrics?.frames_analyzed

    return {
      label,
      confidence,
      isAI,
      source,
      framesAnalyzed,
      forensicSummary: result.forensic_summary,
    }
  }, [result])

  if (!result || !parsed) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4 text-slate-400">
        <p className="text-lg">No video analysis data found.</p>
        <button className="btn-neon" onClick={() => navigate('/video-dashboard')}>Scan Video</button>
      </div>
    )
  }

  const verdictBorder = parsed.isAI ? 'border-neon-red' : 'border-neon-green'
  const verdictBg = parsed.isAI ? 'bg-neon-red/10' : 'bg-neon-green/10'
  const verdictText = parsed.isAI ? 'text-neon-red' : 'text-neon-green'
  const verdictTitle = parsed.isAI ? 'VERDICT: AI-GENERATED VIDEO' : 'VERDICT: AUTHENTIC VIDEO'

  const logLines = ['[FORENSIC SUMMARY]']
  if (parsed.forensicSummary?.points?.length) {
    parsed.forensicSummary.points.forEach(point => logLines.push(`- ${point}`))
    if (parsed.forensicSummary.conclusion) {
      logLines.push(`[CONCLUSION] ${parsed.forensicSummary.conclusion}`)
    }
  } else {
    logLines.push(`- Detection method: ${parsed.source}`)
    if (Number.isFinite(parsed.framesAnalyzed)) {
      logLines.push(`- Frames analyzed: ${parsed.framesAnalyzed}`)
    }
    if (result.result) {
      logLines.push(`- Model output: ${result.result}`)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-4xl">
        <div className="text-center mb-6">
          <h1 className="gradient-text text-4xl font-extrabold tracking-tight mb-1">
            Video Scan Report
          </h1>
          <p className="text-slate-400">
            File: <span className="text-white font-medium">{result.filename || 'Unknown'}</span>
          </p>
        </div>

        <div className={`glass-card border ${verdictBorder} ${verdictBg} ${verdictText}
                         p-4 text-center mb-8 shadow-[0_0_20px_rgba(0,0,0,0.3)]`}>
          <div className="text-2xl font-extrabold tracking-widest">{verdictTitle}</div>
          <div className="text-sm mt-1 opacity-80">Source: {parsed.source}</div>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-8">
          <div className="glass-card p-4 text-center">
            <span className="block text-xs text-slate-400 mb-1 uppercase tracking-wider">Label</span>
            <span className={`text-lg font-bold ${parsed.isAI ? 'text-neon-red' : 'text-neon-green'}`}>
              {parsed.label}
            </span>
          </div>
          <div className="glass-card p-4 text-center">
            <span className="block text-xs text-slate-400 mb-1 uppercase tracking-wider">Frames</span>
            <span className="text-lg font-bold text-neon-blue">
              {Number.isFinite(parsed.framesAnalyzed) ? parsed.framesAnalyzed : 'N/A'}
            </span>
          </div>
          <div className="glass-card p-4 text-center">
            <span className="block text-xs text-slate-400 mb-1 uppercase tracking-wider">Method</span>
            <span className="text-lg font-bold text-neon-blue">{parsed.source}</span>
          </div>
        </div>

        <div className="glass-card p-8 mb-8">
          <div className="grid grid-cols-[auto_1fr] gap-6 items-start">
            <div className="flex flex-col items-center gap-3">
              <p className="text-xs text-slate-400 text-center">Final Confidence Score</p>
              <ScoreCircle confidence={parsed.confidence} isAI={parsed.isAI} />
              <p className={`font-bold text-sm ${parsed.isAI ? 'text-neon-red' : 'text-neon-green'}`}>
                Probability: {probabilityLabel(parsed.confidence)}
              </p>
            </div>

            <div className="bg-black/40 rounded-lg p-4 h-48 overflow-y-auto font-mono text-sm text-slate-300 leading-relaxed">
              {logLines.map((line, index) => <div key={index}>{line}</div>)}
            </div>
          </div>
        </div>

        <div className="text-center flex items-center justify-center gap-3">
          <button className="btn-neon" onClick={() => navigate('/video-dashboard')}>
            Scan Another Video
          </button>
          <button className="btn-neon" onClick={() => navigate('/')}>
            Home
          </button>
        </div>
      </div>
    </div>
  )
}
