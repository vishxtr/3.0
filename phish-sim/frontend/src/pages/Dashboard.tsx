// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react'
import { Shield, Activity, AlertTriangle, Clock } from 'lucide-react'

const Dashboard: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-lg p-6 text-white">
        <h1 className="text-3xl font-bold mb-2">Phish-Sim Dashboard</h1>
        <p className="text-primary-100">
          Real-Time AI/ML-Based Phishing Detection & Prevention — Web Simulation
        </p>
        <div className="mt-4 flex items-center space-x-4 text-sm">
          <div className="flex items-center">
            <Activity className="h-4 w-4 mr-2" />
            Status: Running
          </div>
          <div className="flex items-center">
            <Clock className="h-4 w-4 mr-2" />
            Task: T001 - Project Scaffolding
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center">
            <Shield className="h-8 w-8 text-success-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Total Scans</p>
              <p className="text-2xl font-bold text-white">0</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <AlertTriangle className="h-8 w-8 text-danger-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Threats Detected</p>
              <p className="text-2xl font-bold text-white">0</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <Activity className="h-8 w-8 text-primary-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Avg Response Time</p>
              <p className="text-2xl font-bold text-white">0ms</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <Shield className="h-8 w-8 text-gray-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Accuracy</p>
              <p className="text-2xl font-bold text-white">0%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="card">
        <h2 className="text-xl font-bold text-white mb-4">Recent Activity</h2>
        <div className="text-gray-400 text-center py-8">
          <Shield className="h-12 w-12 mx-auto mb-4 text-gray-600" />
          <p>No activity yet. Start by analyzing a URL or text content.</p>
        </div>
      </div>

      {/* System Status */}
      <div className="card">
        <h2 className="text-xl font-bold text-white mb-4">System Status</h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Backend API</span>
            <span className="text-success-500 font-medium">✓ Healthy</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-300">ML Pipeline</span>
            <span className="text-gray-500 font-medium">⏳ Initializing</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Database</span>
            <span className="text-success-500 font-medium">✓ Connected</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard