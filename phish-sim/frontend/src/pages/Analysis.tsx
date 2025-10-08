// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React, { useState, useEffect } from 'react';
import { Search, AlertTriangle, CheckCircle, Clock, Copy, ExternalLink, RefreshCw, Zap } from 'lucide-react';
import { useAnalysis, useAnalysisHistory, useWebSocketConnection, useAnalysisSubscription } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import { AnalysisRequest, AnalysisResponse } from '../types';

const Analysis: React.FC = () => {
  const [input, setInput] = useState('');
  const [contentType, setContentType] = useState<'url' | 'email' | 'text'>('text');
  const [forceReanalyze, setForceReanalyze] = useState(false);
  const [currentResult, setCurrentResult] = useState<AnalysisResponse | null>(null);
  const [requestId, setRequestId] = useState<string | null>(null);

  const { analyze, isLoading, error, data, isSuccess, reset } = useAnalysis();
  const { history, addToHistory, getHistoryByType, getHistoryByPrediction } = useAnalysisHistory();
  const { isConnected: wsConnected } = useWebSocketConnection();
  const { result: wsResult, loadingState: wsLoadingState } = useAnalysisSubscription(requestId);

  // Update current result when analysis completes
  useEffect(() => {
    if (data) {
      setCurrentResult(data);
      addToHistory(data);
      setRequestId(null); // Clear subscription
    }
  }, [data, addToHistory]);

  // Update current result when WebSocket result arrives
  useEffect(() => {
    if (wsResult) {
      setCurrentResult(wsResult);
      addToHistory(wsResult);
    }
  }, [wsResult, addToHistory]);

  const handleAnalyze = async () => {
    if (!input.trim()) return;

    reset();
    setCurrentResult(null);

    const request: AnalysisRequest = {
      content: input.trim(),
      content_type: contentType,
      force_reanalyze: forceReanalyze,
    };

    try {
      const result = await analyze(request);
      setRequestId(result.request_id);
    } catch (err) {
      console.error('Analysis failed:', err);
    }
  };

  const handleCopyResult = () => {
    if (currentResult) {
      const resultText = `Analysis Result:
Content: ${currentResult.content}
Type: ${currentResult.content_type}
Prediction: ${currentResult.prediction}
Confidence: ${(currentResult.confidence * 100).toFixed(1)}%
Processing Time: ${currentResult.processing_time_ms}ms
Timestamp: ${currentResult.timestamp}`;
      
      navigator.clipboard.writeText(resultText);
    }
  };

  const handleClearHistory = () => {
    setCurrentResult(null);
    setInput('');
    reset();
  };

  const detectContentType = (text: string): 'url' | 'email' | 'text' => {
    if (text.match(/^https?:\/\/.+/i)) return 'url';
    if (text.includes('@') && text.includes('.')) return 'email';
    return 'text';
  };

  const handleInputChange = (value: string) => {
    setInput(value);
    if (value.trim()) {
      setContentType(detectContentType(value));
    }
  };

  const getPredictionColor = (prediction: string) => {
    switch (prediction) {
      case 'phish':
        return 'text-red-500';
      case 'suspicious':
        return 'text-yellow-500';
      case 'benign':
        return 'text-green-500';
      default:
        return 'text-gray-500';
    }
  };

  const getPredictionIcon = (prediction: string) => {
    switch (prediction) {
      case 'phish':
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
      case 'suspicious':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case 'benign':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-500';
    if (confidence >= 0.6) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="space-y-6">
      {/* Analysis Input */}
      <Card title="Phishing Analysis" subtitle="Analyze URLs, emails, and text content for phishing threats">
        <div className="space-y-4">
          <div>
            <Input
              type="textarea"
              label="Content to Analyze"
              placeholder="Enter a suspicious URL, email, or text content..."
              value={input}
              onChange={handleInputChange}
              className="h-32"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-300">Type:</label>
                <select
                  value={contentType}
                  onChange={(e) => setContentType(e.target.value as any)}
                  className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white text-sm"
                >
                  <option value="text">Text</option>
                  <option value="url">URL</option>
                  <option value="email">Email</option>
                </select>
              </div>

              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={forceReanalyze}
                  onChange={(e) => setForceReanalyze(e.target.checked)}
                  className="rounded border-gray-600 bg-gray-700 text-primary-500"
                />
                <span className="text-sm text-gray-300">Force Re-analyze</span>
              </label>
            </div>

            <div className="flex items-center space-x-2">
              {wsConnected && (
                <div className="flex items-center text-green-400 text-sm">
                  <Zap className="h-4 w-4 mr-1" />
                  Real-time
                </div>
              )}
              <Button
                onClick={handleAnalyze}
                disabled={!input.trim() || isLoading}
                loading={isLoading}
                className="flex items-center"
              >
                <Search className="h-4 w-4 mr-2" />
                Analyze
              </Button>
            </div>
          </div>

          {error && (
            <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4">
              <div className="flex items-center">
                <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
                <div>
                  <h4 className="text-red-400 font-medium">Analysis Failed</h4>
                  <p className="text-red-300 text-sm mt-1">{error.message}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Analysis Result */}
      {currentResult && (
        <Card 
          title="Analysis Result" 
          actions={
            <div className="flex space-x-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCopyResult}
              >
                <Copy className="h-4 w-4 mr-2" />
                Copy
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleClearHistory}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Clear
              </Button>
            </div>
          }
        >
          <div className="space-y-6">
            {/* Main Result Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  {getPredictionIcon(currentResult.prediction)}
                  <span className="font-medium text-white ml-2">Prediction</span>
                </div>
                <p className={`text-2xl font-bold capitalize ${getPredictionColor(currentResult.prediction)}`}>
                  {currentResult.prediction}
                </p>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <AlertTriangle className="h-5 w-5 text-blue-500 mr-2" />
                  <span className="font-medium text-white">Confidence</span>
                </div>
                <p className={`text-2xl font-bold ${getConfidenceColor(currentResult.confidence)}`}>
                  {(currentResult.confidence * 100).toFixed(1)}%
                </p>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <Clock className="h-5 w-5 text-gray-400 mr-2" />
                  <span className="font-medium text-white">Processing Time</span>
                </div>
                <p className="text-2xl font-bold text-white">
                  {currentResult.processing_time_ms}ms
                </p>
              </div>
            </div>

            {/* Detailed Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-white mb-3">Analysis Details</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Request ID:</span>
                    <span className="text-white font-mono">{currentResult.request_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Content Type:</span>
                    <span className="text-white capitalize">{currentResult.content_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Cached:</span>
                    <span className={currentResult.cached ? 'text-green-400' : 'text-yellow-400'}>
                      {currentResult.cached ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Timestamp:</span>
                    <span className="text-white">
                      {new Date(currentResult.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-white mb-3">Explanation</h4>
                <div className="space-y-2 text-sm">
                  {currentResult.explanation.nlp && (
                    <div>
                      <span className="text-gray-400">NLP Analysis:</span>
                      <p className="text-white mt-1">{currentResult.explanation.nlp.reason}</p>
                    </div>
                  )}
                  {currentResult.explanation.visual && (
                    <div>
                      <span className="text-gray-400">Visual Analysis:</span>
                      <p className="text-white mt-1">{currentResult.explanation.visual.reason}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Original Content */}
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="font-medium text-white mb-3">Original Content</h4>
              <div className="bg-gray-800 rounded p-3 text-sm text-gray-300 font-mono break-all">
                {currentResult.content}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Analysis History */}
      {history.length > 0 && (
        <Card title="Analysis History" subtitle={`${history.length} recent analyses`}>
          <div className="space-y-3">
            {history.slice(0, 10).map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-3">
                  {getPredictionIcon(item.prediction)}
                  <div>
                    <p className="text-white text-sm font-medium">
                      {item.content.substring(0, 60)}...
                    </p>
                    <p className="text-gray-400 text-xs">
                      {item.content_type} • {item.prediction} • {(item.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-gray-400 text-xs">
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </p>
                  <p className="text-gray-500 text-xs">
                    {item.processing_time_ms}ms
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* WebSocket Status */}
      {wsLoadingState === 'loading' && (
        <Card title="Real-time Analysis">
          <div className="flex items-center justify-center py-8">
            <LoadingSpinner size="lg" text="Waiting for real-time analysis results..." />
          </div>
        </Card>
      )}
    </div>
  );
};

export default Analysis;