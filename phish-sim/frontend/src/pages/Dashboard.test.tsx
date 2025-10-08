// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Dashboard from './Dashboard'

describe('Dashboard', () => {
  it('renders dashboard components', () => {
    render(<Dashboard />)
    
    expect(screen.getByText('Phish-Sim Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Total Scans')).toBeInTheDocument()
    expect(screen.getByText('Threats Detected')).toBeInTheDocument()
    expect(screen.getByText('Avg Response Time')).toBeInTheDocument()
    expect(screen.getByText('Accuracy')).toBeInTheDocument()
  })

  it('shows system status', () => {
    render(<Dashboard />)
    
    expect(screen.getByText('System Status')).toBeInTheDocument()
    expect(screen.getByText('Backend API')).toBeInTheDocument()
    expect(screen.getByText('ML Pipeline')).toBeInTheDocument()
    expect(screen.getByText('Database')).toBeInTheDocument()
  })

  it('displays recent activity section', () => {
    render(<Dashboard />)
    
    expect(screen.getByText('Recent Activity')).toBeInTheDocument()
    expect(screen.getByText('No activity yet. Start by analyzing a URL or text content.')).toBeInTheDocument()
  })
})