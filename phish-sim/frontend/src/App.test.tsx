// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import App from './App'

describe('App', () => {
  it('renders the main navigation', () => {
    render(<App />)
    
    expect(screen.getByText('Phish-Sim')).toBeInTheDocument()
    expect(screen.getByText('T001 - Project Scaffolding')).toBeInTheDocument()
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Analysis')).toBeInTheDocument()
    expect(screen.getByText('Simulator')).toBeInTheDocument()
    expect(screen.getByText('Settings')).toBeInTheDocument()
  })

  it('renders the dashboard by default', () => {
    render(<App />)
    
    expect(screen.getByText('Phish-Sim Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Real-Time AI/ML-Based Phishing Detection & Prevention — Web Simulation')).toBeInTheDocument()
  })
})