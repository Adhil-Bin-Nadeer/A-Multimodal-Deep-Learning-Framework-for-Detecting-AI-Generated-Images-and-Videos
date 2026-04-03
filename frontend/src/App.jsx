import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Report from './pages/Report.jsx'
import VideoDashboard from './pages/VideoDashboard.jsx'
import VideoReport from './pages/VideoReport.jsx'

export default function App() {
  return (
    <Routes>
      <Route path="/"               element={<Home />} />
      <Route path="/dashboard"      element={<Dashboard />} />
      <Route path="/report"         element={<Report />} />
      <Route path="/video-dashboard" element={<VideoDashboard />} />
      <Route path="/video-report"    element={<VideoReport />} />
    </Routes>
  )
}
