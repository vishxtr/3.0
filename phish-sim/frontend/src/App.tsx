// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Shield, Activity, AlertTriangle, Settings } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Analysis from './pages/Analysis'
import Simulator from './pages/Simulator'
import SettingsPage from './pages/Settings'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900">
        {/* Navigation */}
        <nav className="bg-gray-800 border-b border-gray-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <Shield className="h-8 w-8 text-primary-500 mr-3" />
                <h1 className="text-xl font-bold text-white">Phish-Sim</h1>
                <span className="ml-3 text-sm text-gray-400">T001 - Project Scaffolding</span>
              </div>
              <div className="flex items-center space-x-4">
                <a href="/" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium flex items-center">
                  <Activity className="h-4 w-4 mr-2" />
                  Dashboard
                </a>
                <a href="/analysis" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium flex items-center">
                  <AlertTriangle className="h-4 w-4 mr-2" />
                  Analysis
                </a>
                <a href="/simulator" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">
                  Simulator
                </a>
                <a href="/settings" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium flex items-center">
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </a>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App