import React from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Scan from './pages/Scan'
import Model from './pages/Model'

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-gray-100">
      <header className="p-4 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <svg className="w-10 h-10" viewBox="0 0 24 24" fill="none" aria-hidden>
            <rect width="24" height="24" rx="6" className="fill-brand-700"/>
          </svg>
          <div>
            <div className="text-xl font-bold">PhishGuard Pro</div>
            <div className="text-sm text-gray-400">Hackathon demo • Synthetic data only</div>
          </div>
        </div>
        <nav className="flex gap-4">
          <Link to="/" className="text-gray-300 hover:text-white">Dashboard</Link>
          <Link to="/scan" className="text-gray-300 hover:text-white">Scan URL</Link>
          <Link to="/model" className="text-gray-300 hover:text-white">Model</Link>
        </nav>
      </header>

      <main className="p-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/scan" element={<Scan />} />
          <Route path="/model" element={<Model />} />
        </Routes>
      </main>
    </div>
  )
}