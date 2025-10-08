// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react'
import { Settings as SettingsIcon, Database, Cpu, Shield } from 'lucide-react'

const SettingsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="card">
        <h1 className="text-2xl font-bold text-white mb-6 flex items-center">
          <SettingsIcon className="h-6 w-6 mr-3" />
          System Settings
        </h1>
        
        <div className="space-y-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <Cpu className="h-5 w-5 text-primary-500 mr-2" />
              <h3 className="text-lg font-semibold text-white">ML Pipeline</h3>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">Model Status:</span>
                <span className="text-gray-500">Not Initialized</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Detection Threshold:</span>
                <span className="text-gray-500">0.5</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Max Processing Time:</span>
                <span className="text-gray-500">100ms</span>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <Database className="h-5 w-5 text-success-500 mr-2" />
              <h3 className="text-lg font-semibold text-white">Database</h3>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">Type:</span>
                <span className="text-gray-500">SQLite</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Status:</span>
                <span className="text-success-500">Connected</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Records:</span>
                <span className="text-gray-500">0</span>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <Shield className="h-5 w-5 text-danger-500 mr-2" />
              <h3 className="text-lg font-semibold text-white">Security</h3>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">Simulation Mode:</span>
                <span className="text-success-500">Enabled</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Real Threats:</span>
                <span className="text-danger-500">Disabled</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Data Privacy:</span>
                <span className="text-success-500">Protected</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-bold text-white mb-4">System Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Version:</span>
            <span className="text-white ml-2">1.0.0</span>
          </div>
          <div>
            <span className="text-gray-400">Task:</span>
            <span className="text-white ml-2">T001 - Project Scaffolding</span>
          </div>
          <div>
            <span className="text-gray-400">Environment:</span>
            <span className="text-white ml-2">Development</span>
          </div>
          <div>
            <span className="text-gray-400">License:</span>
            <span className="text-white ml-2">MIT</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SettingsPage