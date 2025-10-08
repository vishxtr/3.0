#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

/**
 * Frontend Test Runner
 * Runs all frontend tests and generates a comprehensive report
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🧪 Running Frontend Tests for Phish-Sim');
console.log('=====================================\n');

// Test configuration
const testConfig = {
  testDir: path.join(__dirname, '..'),
  coverageDir: path.join(__dirname, '..', 'coverage'),
  reportDir: path.join(__dirname, '..', 'test-reports'),
};

// Ensure directories exist
[testConfig.coverageDir, testConfig.reportDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Test suites to run
const testSuites = [
  {
    name: 'Component Tests',
    pattern: 'src/tests/components/**/*.test.{ts,tsx}',
    description: 'UI component functionality and rendering tests'
  },
  {
    name: 'Hook Tests',
    pattern: 'src/tests/hooks/**/*.test.{ts,tsx}',
    description: 'Custom React hooks and state management tests'
  },
  {
    name: 'Integration Tests',
    pattern: 'src/tests/**/*.test.{ts,tsx}',
    description: 'Full application integration and routing tests'
  }
];

// Run tests
async function runTests() {
  const results = {
    total: 0,
    passed: 0,
    failed: 0,
    suites: []
  };

  console.log('📋 Test Suites:');
  testSuites.forEach((suite, index) => {
    console.log(`   ${index + 1}. ${suite.name} - ${suite.description}`);
  });
  console.log('');

  for (const suite of testSuites) {
    console.log(`🔍 Running ${suite.name}...`);
    
    try {
      // Run vitest for the specific pattern
      const command = `npx vitest run ${suite.pattern} --reporter=verbose --coverage`;
      
      console.log(`   Command: ${command}`);
      
      const output = execSync(command, { 
        cwd: testConfig.testDir,
        encoding: 'utf8',
        stdio: 'pipe'
      });

      console.log(`   ✅ ${suite.name} completed successfully`);
      
      // Parse results (simplified)
      const lines = output.split('\n');
      const testLine = lines.find(line => line.includes('Tests:'));
      if (testLine) {
        const match = testLine.match(/(\d+) passed|(\d+) failed/);
        if (match) {
          const passed = parseInt(match[1]) || 0;
          const failed = parseInt(match[2]) || 0;
          results.passed += passed;
          results.failed += failed;
          results.total += passed + failed;
        }
      }

      results.suites.push({
        name: suite.name,
        status: 'passed',
        output: output
      });

    } catch (error) {
      console.log(`   ❌ ${suite.name} failed`);
      console.log(`   Error: ${error.message}`);
      
      results.suites.push({
        name: suite.name,
        status: 'failed',
        error: error.message
      });
    }
    
    console.log('');
  }

  // Generate summary
  console.log('📊 Test Summary');
  console.log('===============');
  console.log(`Total Tests: ${results.total}`);
  console.log(`Passed: ${results.passed} ✅`);
  console.log(`Failed: ${results.failed} ❌`);
  console.log(`Success Rate: ${results.total > 0 ? ((results.passed / results.total) * 100).toFixed(1) : 0}%`);
  console.log('');

  // Generate detailed report
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      total: results.total,
      passed: results.passed,
      failed: results.failed,
      successRate: results.total > 0 ? (results.passed / results.total) * 100 : 0
    },
    suites: results.suites,
    environment: {
      node: process.version,
      platform: process.platform,
      arch: process.arch
    }
  };

  // Save report
  const reportPath = path.join(testConfig.reportDir, `test-report-${Date.now()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`📄 Detailed report saved to: ${reportPath}`);

  // Coverage report
  const coveragePath = path.join(testConfig.coverageDir, 'index.html');
  if (fs.existsSync(coveragePath)) {
    console.log(`📈 Coverage report available at: ${coveragePath}`);
  }

  // Exit with appropriate code
  if (results.failed > 0) {
    console.log('\n❌ Some tests failed. Please review the output above.');
    process.exit(1);
  } else {
    console.log('\n🎉 All tests passed successfully!');
    process.exit(0);
  }
}

// Run the tests
runTests().catch(error => {
  console.error('💥 Test runner failed:', error);
  process.exit(1);
});