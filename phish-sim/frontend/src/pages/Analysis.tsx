// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React, { useState } from 'react'
import { Search, AlertTriangle, CheckCircle, Clock } from 'lucide-react'

const Analysis: React.FC = () => {
  const [input, setInput] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)

  const handleAnalyze = async () => {
    if (!input.trim()) return
    
    setIsAnalyzing(true)
    // Simulate API call
    setTimeout(() => {
      setResult({
        score: 0.3,
        decision: 'benign',
        confidence: 0.7,
        processing_time_ms: 45
      })
      setIsAnalyzing(false)
    }, 1000)
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <h1 className="text-2xl font-bold text-white mb-6">Phishing Analysis</h1>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Enter URL or text to analyze
            </label>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Enter a suspicious URL or text content..."
              className="input-field w-full h-32 resize-none"
            />
          </div>
          
          <button
            onClick={handleAnalyze}
            disabled={!input.trim() || isAnalyzing}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {isAnalyzing ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="h-4 w-4 mr-2" />
                Analyze
              </>
            )}
          </button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2 className="text-xl font-bold text-white mb-4">Analysis Result</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center mb-2">
                {result.decision === 'benign' ? (
                  <CheckCircle className="h-5 w-5 text-success-500 mr-2" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-danger-500 mr-2" />
                )}
                <span className="font-medium text-white">Decision</span>
              </div>
              <p className="text-2xl font-bold text-white capitalize">{result.decision}</p>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center mb-2">
                <AlertTriangle className="h-5 w-5 text-primary-500 mr-2" />
                <span className="font-medium text-white">Threat Score</span>
              </div>
              <p className="text-2xl font-bold text-white">{(result.score * 100).toFixed(1)}%</p>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center mb-2">
                <Clock className="h-5 w-5 text-gray-400 mr-2" />
                <span className="font-medium text-white">Processing Time</span>
              </div>
              <p className="text-2xl font-bold text-white">{result.processing_time_ms}ms</p>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="font-medium text-white mb-2">Analysis Details</h3>
            <p className="text-gray-300 text-sm">
              This is a placeholder analysis result from T001. The actual ML pipeline will be implemented in future tasks.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default Analysis