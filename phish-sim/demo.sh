#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

echo "=========================================="
echo "Phish-Sim T001 Demo Script"
echo "=========================================="
echo ""

echo "📁 Project Structure:"
tree -I 'node_modules|__pycache__|*.pyc|.git' -L 3
echo ""

echo "🚀 Quick Start Commands:"
echo "1. Install dependencies: make deps"
echo "2. Run tests: make test"
echo "3. Start services: make up"
echo "4. Access frontend: http://localhost:3000"
echo "5. Access backend API: http://localhost:8000"
echo "6. API Documentation: http://localhost:8000/docs"
echo ""

echo "🔧 Available Make Commands:"
make help
echo ""

echo "📊 Backend API Endpoints:"
echo "- GET  /              - Root endpoint"
echo "- GET  /health        - Health check"
echo "- POST /analyze       - Analyze content for phishing"
echo "- GET  /metrics       - System metrics"
echo ""

echo "🎨 Frontend Pages:"
echo "- /                  - Dashboard"
echo "- /analysis          - Phishing Analysis"
echo "- /simulator         - Phishing Simulator"
echo "- /settings          - System Settings"
echo ""

echo "✅ T001 Implementation Complete!"
echo "Next Task: T002 - Data ingestion & synthetic dataset generator"