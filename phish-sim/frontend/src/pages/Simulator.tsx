// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React, { useState, useEffect } from 'react';
import { Play, Shield, AlertTriangle, Pause, Square, BarChart3, Target, Zap, Clock, CheckCircle } from 'lucide-react';
import { useBatchAnalysis } from '../hooks/useApi';
import { useWebSocketConnection } from '../hooks/useWebSocket';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import StatusBadge from '../components/ui/StatusBadge';
import { SimulationConfig, SimulationResult, AnalysisResponse } from '../types';

const Simulator: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentSimulation, setCurrentSimulation] = useState<SimulationResult | null>(null);
  const [simulationHistory, setSimulationHistory] = useState<SimulationResult[]>([]);
  const [config, setConfig] = useState<SimulationConfig>({
    name: 'Default Simulation',
    description: 'Test phishing detection capabilities',
    attack_type: 'phishing_email',
    target_count: 10,
    duration_minutes: 5,
    intensity: 'medium',
  });

  const { analyzeBatch, isLoading, error, results } = useBatchAnalysis();
  const { isConnected: wsConnected } = useWebSocketConnection();

  // Sample test data for simulation
  const samplePhishingData = {
    phishing_email: [
      'Urgent: Your account will be suspended in 24 hours. Click here to verify: http://fake-bank.com/verify',
      'Congratulations! You won $1000. Claim your prize now: http://scam-lottery.com/claim',
      'Security Alert: Unusual login detected. Secure your account: http://phishing-security.net/login',
      'Your package delivery failed. Reschedule here: http://fake-delivery.com/reschedule',
      'Tax refund available. Click to claim: http://irs-scam.gov/refund',
    ],
    malicious_url: [
      'http://fake-paypal-security.com/login',
      'http://microsoft-update-scam.net/download',
      'http://amazon-account-verify.com/confirm',
      'http://apple-id-locked.com/unlock',
      'http://google-security-alert.com/verify',
    ],
    social_engineering: [
      'Hi, this is IT support. We need to verify your password for security reasons.',
      'Your computer has been infected. Download our antivirus software immediately.',
      'We detected suspicious activity on your account. Please provide your SSN for verification.',
      'You have been selected for a special offer. Just provide your credit card details.',
      'Your subscription is expiring. Click here to renew and avoid service interruption.',
    ],
  };

  const getSampleData = (type: string, count: number): string[] => {
    const data = samplePhishingData[type as keyof typeof samplePhishingData] || [];
    return Array.from({ length: count }, (_, i) => data[i % data.length]);
  };

  const startSimulation = async () => {
    if (isRunning) return;

    setIsRunning(true);
    const simulationId = `sim_${Date.now()}`;
    
    const newSimulation: SimulationResult = {
      id: simulationId,
      config,
      start_time: new Date(),
      status: 'running',
      results: {
        total_attacks: 0,
        detected: 0,
        missed: 0,
        false_positives: 0,
        detection_rate: 0,
      },
    };

    setCurrentSimulation(newSimulation);

    try {
      // Generate test data based on configuration
      const testData = getSampleData(config.attack_type, config.target_count);
      
      // Create analysis requests
      const requests = testData.map((content, index) => ({
        content,
        content_type: config.attack_type === 'malicious_url' ? 'url' as const : 'email' as const,
        user_id: `sim_${simulationId}_${index}`,
        session_id: simulationId,
      }));

      // Run batch analysis
      const analysisResults = await analyzeBatch(requests);
      
      // Calculate results
      const totalAttacks = analysisResults.length;
      const detected = analysisResults.filter(r => r.prediction === 'phish').length;
      const missed = totalAttacks - detected;
      const falsePositives = 0; // In a real scenario, this would be calculated differently
      const detectionRate = totalAttacks > 0 ? (detected / totalAttacks) * 100 : 0;

      // Update simulation with results
      const completedSimulation: SimulationResult = {
        ...newSimulation,
        end_time: new Date(),
        status: 'completed',
        results: {
          total_attacks: totalAttacks,
          detected,
          missed,
          false_positives: falsePositives,
          detection_rate: detectionRate,
        },
      };

      setCurrentSimulation(completedSimulation);
      setSimulationHistory(prev => [completedSimulation, ...prev.slice(0, 9)]); // Keep last 10
      
    } catch (err) {
      console.error('Simulation failed:', err);
      if (currentSimulation) {
        setCurrentSimulation({
          ...currentSimulation,
          end_time: new Date(),
          status: 'failed',
        });
      }
    } finally {
      setIsRunning(false);
    }
  };

  const stopSimulation = () => {
    setIsRunning(false);
    if (currentSimulation) {
      setCurrentSimulation({
        ...currentSimulation,
        end_time: new Date(),
        status: 'paused',
      });
    }
  };

  const clearHistory = () => {
    setSimulationHistory([]);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'text-blue-500';
      case 'completed':
        return 'text-green-500';
      case 'failed':
        return 'text-red-500';
      case 'paused':
        return 'text-yellow-500';
      default:
        return 'text-gray-500';
    }
  };

  const getDetectionRateColor = (rate: number) => {
    if (rate >= 90) return 'text-green-500';
    if (rate >= 70) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="space-y-6">
      {/* Simulation Configuration */}
      <Card title="Simulation Configuration" subtitle="Configure and run phishing detection tests">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <Input
              label="Simulation Name"
              value={config.name}
              onChange={(value) => setConfig(prev => ({ ...prev, name: value }))}
              placeholder="Enter simulation name"
            />
            
            <Input
              type="textarea"
              label="Description"
              value={config.description}
              onChange={(value) => setConfig(prev => ({ ...prev, description: value }))}
              placeholder="Enter simulation description"
            />

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Attack Type
              </label>
              <select
                value={config.attack_type}
                onChange={(e) => setConfig(prev => ({ ...prev, attack_type: e.target.value as any }))}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white w-full"
              >
                <option value="phishing_email">Phishing Email</option>
                <option value="malicious_url">Malicious URL</option>
                <option value="social_engineering">Social Engineering</option>
              </select>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Target Count
              </label>
              <input
                type="number"
                min="1"
                max="100"
                value={config.target_count}
                onChange={(e) => setConfig(prev => ({ ...prev, target_count: parseInt(e.target.value) }))}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Duration (minutes)
              </label>
              <input
                type="number"
                min="1"
                max="60"
                value={config.duration_minutes}
                onChange={(e) => setConfig(prev => ({ ...prev, duration_minutes: parseInt(e.target.value) }))}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Intensity
              </label>
              <select
                value={config.intensity}
                onChange={(e) => setConfig(prev => ({ ...prev, intensity: e.target.value as any }))}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white w-full"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-700">
          <div className="flex items-center space-x-4">
            {wsConnected && (
              <div className="flex items-center text-green-400 text-sm">
                <Zap className="h-4 w-4 mr-1" />
                Real-time enabled
              </div>
            )}
            <StatusBadge 
              status={isRunning ? 'healthy' : 'down'} 
              className={isRunning ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'}
            />
          </div>

          <div className="flex space-x-2">
            {isRunning ? (
              <Button
                variant="danger"
                onClick={stopSimulation}
                className="flex items-center"
              >
                <Square className="h-4 w-4 mr-2" />
                Stop
              </Button>
            ) : (
              <Button
                onClick={startSimulation}
                disabled={isLoading}
                loading={isLoading}
                className="flex items-center"
              >
                <Play className="h-4 w-4 mr-2" />
                Start Simulation
              </Button>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-4 bg-red-900/20 border border-red-500/50 rounded-lg p-4">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
              <div>
                <h4 className="text-red-400 font-medium">Simulation Failed</h4>
                <p className="text-red-300 text-sm mt-1">{error.message}</p>
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Current Simulation */}
      {currentSimulation && (
        <Card title="Current Simulation" subtitle={`${currentSimulation.config.name} - ${currentSimulation.status}`}>
          <div className="space-y-6">
            {/* Simulation Status */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex items-center">
                  <Clock className="h-5 w-5 text-gray-400 mr-2" />
                  <span className="text-gray-300">Started:</span>
                  <span className="text-white ml-2">
                    {currentSimulation.start_time.toLocaleTimeString()}
                  </span>
                </div>
                {currentSimulation.end_time && (
                  <div className="flex items-center">
                    <CheckCircle className="h-5 w-5 text-gray-400 mr-2" />
                    <span className="text-gray-300">Ended:</span>
                    <span className="text-white ml-2">
                      {currentSimulation.end_time.toLocaleTimeString()}
                    </span>
                  </div>
                )}
              </div>
              <StatusBadge 
                status={currentSimulation.status === 'running' ? 'healthy' : 
                       currentSimulation.status === 'completed' ? 'healthy' : 'down'} 
              />
            </div>

            {/* Results */}
            {currentSimulation.status === 'completed' && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <Target className="h-5 w-5 text-blue-500 mr-2" />
                    <span className="font-medium text-white">Total Attacks</span>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {currentSimulation.results.total_attacks}
                  </p>
                </div>

                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <Shield className="h-5 w-5 text-green-500 mr-2" />
                    <span className="font-medium text-white">Detected</span>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {currentSimulation.results.detected}
                  </p>
                </div>

                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
                    <span className="font-medium text-white">Missed</span>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {currentSimulation.results.missed}
                  </p>
                </div>

                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <BarChart3 className="h-5 w-5 text-purple-500 mr-2" />
                    <span className="font-medium text-white">Detection Rate</span>
                  </div>
                  <p className={`text-2xl font-bold ${getDetectionRateColor(currentSimulation.results.detection_rate)}`}>
                    {currentSimulation.results.detection_rate.toFixed(1)}%
                  </p>
                </div>
              </div>
            )}

            {/* Running Status */}
            {currentSimulation.status === 'running' && (
              <div className="flex items-center justify-center py-8">
                <LoadingSpinner size="lg" text="Running simulation..." />
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Simulation History */}
      {simulationHistory.length > 0 && (
        <Card 
          title="Simulation History" 
          subtitle={`${simulationHistory.length} completed simulations`}
          actions={
            <Button
              variant="secondary"
              size="sm"
              onClick={clearHistory}
            >
              Clear History
            </Button>
          }
        >
          <div className="space-y-3">
            {simulationHistory.map((simulation) => (
              <div key={simulation.id} className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-4">
                  <div>
                    <h4 className="text-white font-medium">{simulation.config.name}</h4>
                    <p className="text-gray-400 text-sm">
                      {simulation.config.attack_type} • {simulation.config.target_count} targets • {simulation.config.intensity} intensity
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="flex items-center space-x-4">
                    <div className="text-center">
                      <p className="text-white font-bold">{simulation.results.detection_rate.toFixed(1)}%</p>
                      <p className="text-gray-400 text-xs">Detection Rate</p>
                    </div>
                    <div className="text-center">
                      <p className="text-white font-bold">{simulation.results.detected}/{simulation.results.total_attacks}</p>
                      <p className="text-gray-400 text-xs">Detected</p>
                    </div>
                    <StatusBadge 
                      status={simulation.status === 'completed' ? 'healthy' : 'down'} 
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Quick Test */}
      <Card title="Quick Test" subtitle="Test individual samples without full simulation">
        <div className="space-y-4">
          <p className="text-gray-300 text-sm">
            Use this section to quickly test individual phishing samples against the detection system.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button
              variant="secondary"
              onClick={() => {
                const sample = samplePhishingData.phishing_email[0];
                // This would trigger a single analysis
                console.log('Testing sample:', sample);
              }}
              className="flex items-center justify-center"
            >
              <AlertTriangle className="h-4 w-4 mr-2" />
              Test Phishing Email
            </Button>
            
            <Button
              variant="secondary"
              onClick={() => {
                const sample = samplePhishingData.malicious_url[0];
                console.log('Testing sample:', sample);
              }}
              className="flex items-center justify-center"
            >
              <Target className="h-4 w-4 mr-2" />
              Test Malicious URL
            </Button>
            
            <Button
              variant="secondary"
              onClick={() => {
                const sample = samplePhishingData.social_engineering[0];
                console.log('Testing sample:', sample);
              }}
              className="flex items-center justify-center"
            >
              <Shield className="h-4 w-4 mr-2" />
              Test Social Engineering
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Simulator;