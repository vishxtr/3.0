// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react'
import { Play, Shield, AlertTriangle } from 'lucide-react'

const Simulator: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="card">
        <h1 className="text-2xl font-bold text-white mb-6">Phishing Simulator</h1>
        <p className="text-gray-300 mb-6">
          Simulate phishing attacks and test detection capabilities. This feature will be implemented in future tasks.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-700 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <Play className="h-6 w-6 text-primary-500 mr-3" />
              <h3 className="text-lg font-semibold text-white">Red Team Simulation</h3>
            </div>
            <p className="text-gray-300 text-sm mb-4">
              Launch simulated phishing campaigns to test detection systems.
            </p>
            <button className="btn-primary w-full" disabled>
              Coming Soon (T010)
            </button>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <Shield className="h-6 w-6 text-success-500 mr-3" />
              <h3 className="text-lg font-semibold text-white">Defense Testing</h3>
            </div>
            <p className="text-gray-300 text-sm mb-4">
              Test defensive mechanisms against various attack vectors.
            </p>
            <button className="btn-primary w-full" disabled>
              Coming Soon (T010)
            </button>
          </div>
        </div>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-bold text-white mb-4">Simulation Status</h2>
        <div className="text-gray-400 text-center py-8">
          <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-gray-600" />
          <p>Simulator not yet implemented. This will be available in T010.</p>
        </div>
      </div>
    </div>
  )
}

export default Simulator