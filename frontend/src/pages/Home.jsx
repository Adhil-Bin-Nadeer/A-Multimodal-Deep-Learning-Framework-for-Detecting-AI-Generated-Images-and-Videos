import { useNavigate } from 'react-router-dom'

export default function Home() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      {/* Main content */}
      <div className="w-full max-w-4xl text-center animate-fade-in-up">
        <h1 className="gradient-text text-5xl font-extrabold mb-2 tracking-tight">
          AI Generated Image and Video Detector
        </h1>
        <p className="text-slate-400 text-lg tracking-widest mb-12">
          Multi-Layered AI Detection &amp; Forensic Analysis
        </p>

        {/* Cards */}
        <div className="flex flex-wrap justify-center gap-8 mb-16">
          {/* Scan Image */}
          <div
            className="glass-card w-60 h-72 flex flex-col items-center justify-center cursor-pointer
                       transition-all duration-300 hover:-translate-y-1.5
                       hover:border-neon-blue hover:shadow-[0_0_20px_rgba(0,242,255,0.2)]
                       group"
            onClick={() => navigate('/dashboard')}
          >
            <span className="text-6xl mb-6">🖼️</span>
            <h3 className="text-lg font-semibold mb-4">Scan Image</h3>
            <button
              className="btn-neon group-hover:bg-neon-blue group-hover:text-black
                         group-hover:shadow-[0_0_20px_#00f2ff]"
            >
              Start Scan
            </button>
          </div>

          {/* Scan Video */}
          <div
            className="glass-card w-60 h-72 flex flex-col items-center justify-center
                       transition-all duration-300 hover:-translate-y-1.5
                       hover:border-neon-blue hover:shadow-[0_0_20px_rgba(0,242,255,0.2)]
                       group cursor-pointer"
            onClick={() => navigate('/video-dashboard')}
          >
            <span className="text-6xl mb-6">🎞️</span>
            <h3 className="text-lg font-semibold mb-4">Scan Video</h3>
            <button
              className="btn-neon group-hover:bg-neon-blue group-hover:text-black
                         group-hover:shadow-[0_0_20px_#00f2ff]"
            >
              Start Scan
            </button>
          </div>
        </div>
      </div>

      {/* System status bar */}
      <div className="fixed bottom-5 text-sm text-slate-400">
        System Status: Online&nbsp;
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-neon-green
                         shadow-[0_0_8px_#00ff88] ml-1 align-middle" />
      </div>
    </div>
  )
}
