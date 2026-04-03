import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

// ── helpers ──────────────────────────────────────────────────────────────────
function formatMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/^### (.+)$/gm,   '<h4 style="color:#00f2ff;margin-top:1rem">$1</h4>')
    .replace(/^## (.+)$/gm,    '<h3 style="color:#00f2ff;margin-top:1.5rem">$1</h3>')
    .replace(/^# (.+)$/gm,     '<h2 style="color:#00f2ff;margin-top:1.5rem">$1</h2>')
    .replace(/^- (.+)$/gm,     '• $1')
    .replace(/\n/g,            '<br />')
}

function probabilityLabel(confidence) {
  if (confidence > 80) return 'Very High'
  if (confidence > 60) return 'High'
  if (confidence > 40) return 'Medium'
  return 'Low'
}

// ── LayerBadge ───────────────────────────────────────────────────────────────
function LayerBadge({ label, value, color }) {
  const colorMap = {
    green: 'text-neon-green',
    red:   'text-neon-red',
    gray:  'text-slate-500',
  }
  return (
    <div className="glass-card p-4 text-center">
      <span className="block text-xs text-slate-400 mb-1 uppercase tracking-wider">{label}</span>
      <span className={`text-lg font-bold ${colorMap[color] || 'text-white'}`}>{value}</span>
    </div>
  )
}

// ── ScoreCircle ───────────────────────────────────────────────────────────────
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
      <div
        className="score-inner"
        style={{ color }}
      >
        {confidence.toFixed(1)}%
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function Report() {
  const navigate = useNavigate()

  const [result,          setResult]          = useState(null)
  const [reportHtml,      setReportHtml]       = useState('')
  const [reportVisible,   setReportVisible]    = useState(false)
  const [generating,      setGenerating]       = useState(false)
  const [reportBtnLabel,  setReportBtnLabel]   = useState('Generate Detailed Report')
  const [rawJson,         setRawJson]          = useState(null)

  // ── Load from sessionStorage ───────────────────────────────────────────────
  useEffect(() => {
    const json = sessionStorage.getItem('analysisResult')
    if (!json) return
    setRawJson(json)
    setResult(JSON.parse(json))
    sessionStorage.removeItem('analysisResult')
  }, [])

  if (!result) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4 text-slate-400">
        <p className="text-lg">No analysis data found.</p>
        <button className="btn-neon" onClick={() => navigate('/')}>Go Home</button>
      </div>
    )
  }

  const { is_ai_generated: isAI, confidence, layers, filename, final_verdict } = result
  const c2pa    = layers?.c2pa
  const synthid = layers?.synthid
  const aiModel = layers?.ai_model
  const forensicSummary = result.forensic_summary

  // ── Layer 1 badge ──────────────────────────────────────────────────────────
  let l1Value = 'Not Found', l1Color = 'gray'
  if (c2pa?.c2pa_present) {
    l1Value = c2pa.ai_generated ? 'AI Declared' : 'Verified'
    l1Color = c2pa.ai_generated ? 'red' : 'green'
  }

  // ── Layer 2 badge ──────────────────────────────────────────────────────────
  let l2Value = 'N/A', l2Color = 'gray'
  if (synthid?.status === 'complete') {
    l2Value = synthid.is_watermarked
      ? `Detected (${synthid.confidence.toFixed(1)}% detector confidence)`
      : `Not Detected (${synthid.confidence.toFixed(1)}% detector confidence)`
    l2Color = synthid.is_watermarked ? 'red' : 'green'
  } else if (synthid?.status === 'skipped') {
    l2Value = 'Skipped'; l2Color = 'gray'
  } else if (synthid?.status === 'unavailable') {
    l2Value = 'Unavailable'; l2Color = 'gray'
  } else if (synthid?.status === 'error') {
    l2Value = 'Error'; l2Color = 'red'
  }

  // ── Layer 3 badge ──────────────────────────────────────────────────────────
  let l3Value = 'Error', l3Color = 'gray'
  if (aiModel?.status === 'complete') {
    l3Value = `${aiModel.confidence.toFixed(1)}% ${aiModel.label}`
    l3Color = aiModel.label === 'AI Image' ? 'red' : 'green'
  } else if (aiModel?.status === 'skipped') {
    l3Value = 'Skipped'; l3Color = 'gray'
  }

  // ── Build log ──────────────────────────────────────────────────────────────
  let logLines = ['[FORENSIC SUMMARY]']
  if (forensicSummary?.points?.length) {
    forensicSummary.points.forEach(item => {
      logLines.push(`- ${item}`)
    })
    if (forensicSummary.conclusion) {
      logLines.push(`[CONCLUSION] ${forensicSummary.conclusion}`)
    }
  } else {
    logLines.push(`[CONCLUSION] Final verdict: ${final_verdict} with ${confidence.toFixed(1)}% confidence.`)
  }

  // ── Verdict banner colours ─────────────────────────────────────────────────
  const verdictBorder = isAI ? 'border-neon-red'   : 'border-neon-green'
  const verdictBg     = isAI ? 'bg-neon-red/10'    : 'bg-neon-green/10'
  const verdictText   = isAI ? 'text-neon-red'      : 'text-neon-green'
  const verdictTitle  = isAI ? 'VERDICT: AI-GENERATED' : 'VERDICT: AUTHENTIC'
  const verdictReason = c2pa?.c2pa_present && c2pa?.ai_generated
    ? 'Source: C2PA signed manifest declaration of AI generation'
    : synthid?.status === 'complete' && synthid?.is_watermarked
      ? 'Source: SynthID watermark detection'
    : c2pa?.c2pa_present
      ? 'Source: C2PA provenance metadata + AI detection model analysis'
      : 'Source: AI detection model ensemble analysis'

  // ── Generate forensic report ───────────────────────────────────────────────
  async function generateReport() {
    if (generating) return
    setGenerating(true)
    setReportBtnLabel('Generating...')
    setReportVisible(true)
    setReportHtml('Generating detailed forensic report...')

    try {
      const response = await fetch('/api/forensic-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: rawJson,
      })
      const report = await response.json()
      if (report.success) {
        setReportHtml(formatMarkdown(report.enhanced_report))
        setReportBtnLabel('Report Generated')
      } else {
        setReportHtml(`<span class="text-neon-red">Error: ${report.error}</span>`)
        setReportBtnLabel('Error')
      }
    } catch (err) {
      setReportHtml(`Network Error: ${err.message}`)
      setReportBtnLabel('Error')
    } finally {
      setGenerating(false)
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="gradient-text text-4xl font-extrabold tracking-tight mb-1">
            Scan Report &amp; Forensics
          </h1>
          <p className="text-slate-400">
            File: <span className="text-white font-medium">{filename || 'Unknown'}</span>
          </p>
        </div>

        {/* Verdict banner */}
        <div className={`glass-card border ${verdictBorder} ${verdictBg} ${verdictText}
                         p-4 text-center mb-8 shadow-[0_0_20px_rgba(0,0,0,0.3)]`}>
          <div className="text-2xl font-extrabold tracking-widest">{verdictTitle}</div>
          <div className="text-sm mt-1 opacity-80">{verdictReason}</div>
        </div>

        {/* 3-layer breakdown */}
        <h3 className="font-semibold text-base mb-3 uppercase tracking-wider text-slate-300">
          3-Layer Results Breakdown
        </h3>
        <div className="grid grid-cols-3 gap-4 mb-8">
          <LayerBadge label="Layer 1: C2PA"    value={l1Value} color={l1Color} />
          <LayerBadge label="Layer 2: SynthID" value={l2Value} color={l2Color} />
          <LayerBadge label="Layer 3: AI Model" value={l3Value} color={l3Color} />
        </div>

        {/* Forensic log */}
        <h3 className="font-semibold text-base mb-3 uppercase tracking-wider text-slate-300">
          Forensic Log &amp; Detailed Analysis
        </h3>
        <div className="glass-card p-8 mb-8">
          <div className="grid grid-cols-[auto_1fr] gap-6 items-start">
            {/* Score circle */}
            <div className="flex flex-col items-center gap-3">
              <p className="text-xs text-slate-400 text-center">Final Confidence Score</p>
              <ScoreCircle confidence={confidence} isAI={isAI} />
              <p
                className={`font-bold text-sm ${isAI ? 'text-neon-red' : 'text-neon-green'}`}
              >
                Probability: {probabilityLabel(confidence)}
              </p>
            </div>

            {/* Detailed log */}
            <div
              className="bg-black/40 rounded-lg p-4 h-48 overflow-y-auto
                         font-mono text-sm text-slate-300 leading-relaxed"
            >
              {logLines.map((line, i) => <div key={i}>{line}</div>)}
            </div>
          </div>
        </div>

        {/* AI forensic report */}
        <h3 className="font-semibold text-base mb-3 uppercase tracking-wider text-slate-300">
          Forensic Report
        </h3>
        <div className="glass-card p-6 mb-8">
          <button
            className="btn-neon mb-4"
            onClick={generateReport}
            disabled={generating || reportBtnLabel === 'Report Generated'}
          >
            {reportBtnLabel}
          </button>

          {reportVisible && (
            <div
              className="font-mono text-sm text-slate-300 leading-relaxed max-h-96
                         overflow-y-auto text-left"
              dangerouslySetInnerHTML={{ __html: reportHtml }}
            />
          )}
        </div>

        {/* Back button */}
        <div className="text-center">
          <button className="btn-neon" onClick={() => navigate('/')}>
            Scan Another File
          </button>
        </div>
      </div>
    </div>
  )
}
