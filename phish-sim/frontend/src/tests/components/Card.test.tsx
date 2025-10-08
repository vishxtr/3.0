// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react';
import { render, screen } from '@testing-library/react';
import { Shield } from 'lucide-react';
import Card from '../../components/ui/Card';

describe('Card', () => {
  test('renders with basic content', () => {
    render(
      <Card>
        <p>Card content</p>
      </Card>
    );

    expect(screen.getByText('Card content')).toBeInTheDocument();
    expect(screen.getByText('Card content').closest('div')).toHaveClass('bg-gray-800', 'rounded-lg', 'border', 'border-gray-700');
  });

  test('renders with title', () => {
    render(
      <Card title="Test Title">
        <p>Card content</p>
      </Card>
    );

    expect(screen.getByText('Test Title')).toBeInTheDocument();
    expect(screen.getByText('Test Title')).toHaveClass('text-lg', 'font-semibold', 'text-white');
  });

  test('renders with subtitle', () => {
    render(
      <Card title="Test Title" subtitle="Test Subtitle">
        <p>Card content</p>
      </Card>
    );

    expect(screen.getByText('Test Subtitle')).toBeInTheDocument();
    expect(screen.getByText('Test Subtitle')).toHaveClass('text-sm', 'text-gray-400');
  });

  test('renders with icon', () => {
    render(
      <Card title="Test Title" icon={<Shield className="h-6 w-6" />}>
        <p>Card content</p>
      </Card>
    );

    const icon = screen.getByRole('img', { hidden: true }); // SVG icons are hidden from screen readers by default
    expect(icon).toBeInTheDocument();
    expect(icon).toHaveClass('h-6', 'w-6');
  });

  test('renders with actions', () => {
    render(
      <Card 
        title="Test Title" 
        actions={<button>Action Button</button>}
      >
        <p>Card content</p>
      </Card>
    );

    expect(screen.getByText('Action Button')).toBeInTheDocument();
  });

  test('applies custom className', () => {
    render(
      <Card className="custom-class">
        <p>Card content</p>
      </Card>
    );

    expect(screen.getByText('Card content').closest('div')).toHaveClass('custom-class');
  });

  test('renders all props together', () => {
    render(
      <Card 
        title="Complete Card"
        subtitle="With all features"
        icon={<Shield className="h-6 w-6" />}
        actions={<button>Action</button>}
        className="custom-class"
      >
        <p>Complete content</p>
      </Card>
    );

    expect(screen.getByText('Complete Card')).toBeInTheDocument();
    expect(screen.getByText('With all features')).toBeInTheDocument();
    expect(screen.getByText('Action')).toBeInTheDocument();
    expect(screen.getByText('Complete content')).toBeInTheDocument();
    expect(screen.getByText('Complete content').closest('div')).toHaveClass('custom-class');
  });
});