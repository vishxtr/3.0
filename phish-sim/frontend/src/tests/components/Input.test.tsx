// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Input from '../../components/ui/Input';

describe('Input', () => {
  test('renders text input by default', () => {
    render(<Input placeholder="Enter text" />);
    
    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('type', 'text');
  });

  test('renders different input types', () => {
    const { rerender } = render(<Input type="email" placeholder="Email" />);
    expect(screen.getByPlaceholderText('Email')).toHaveAttribute('type', 'email');

    rerender(<Input type="password" placeholder="Password" />);
    expect(screen.getByPlaceholderText('Password')).toHaveAttribute('type', 'password');

    rerender(<Input type="url" placeholder="URL" />);
    expect(screen.getByPlaceholderText('URL')).toHaveAttribute('type', 'url');
  });

  test('renders textarea when type is textarea', () => {
    render(<Input type="textarea" placeholder="Enter text" />);
    
    const textarea = screen.getByPlaceholderText('Enter text');
    expect(textarea.tagName).toBe('TEXTAREA');
    expect(textarea).toHaveClass('resize-none');
  });

  test('renders with label', () => {
    render(<Input label="Test Label" placeholder="Enter text" />);
    
    expect(screen.getByText('Test Label')).toBeInTheDocument();
    expect(screen.getByText('Test Label')).toHaveClass('block', 'text-sm', 'font-medium', 'text-gray-300');
  });

  test('shows required indicator when required', () => {
    render(<Input label="Required Field" required placeholder="Enter text" />);
    
    const requiredIndicator = screen.getByText('*');
    expect(requiredIndicator).toBeInTheDocument();
    expect(requiredIndicator).toHaveClass('text-red-500');
  });

  test('handles value changes', () => {
    const handleChange = jest.fn();
    render(<Input onChange={handleChange} placeholder="Enter text" />);
    
    const input = screen.getByPlaceholderText('Enter text');
    fireEvent.change(input, { target: { value: 'test value' } });
    
    expect(handleChange).toHaveBeenCalledWith('test value');
  });

  test('shows error message', () => {
    render(<Input error="This field is required" placeholder="Enter text" />);
    
    expect(screen.getByText('This field is required')).toBeInTheDocument();
    expect(screen.getByText('This field is required')).toHaveClass('text-sm', 'text-red-500');
  });

  test('applies error styling when error is present', () => {
    render(<Input error="Error message" placeholder="Enter text" />);
    
    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toHaveClass('border-red-500', 'focus:ring-red-500', 'focus:border-red-500');
  });

  test('is disabled when disabled prop is true', () => {
    render(<Input disabled placeholder="Enter text" />);
    
    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toBeDisabled();
    expect(input).toHaveClass('disabled:opacity-50', 'disabled:cursor-not-allowed');
  });

  test('applies custom className', () => {
    render(<Input className="custom-class" placeholder="Enter text" />);
    
    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toHaveClass('custom-class');
  });

  test('renders with controlled value', () => {
    render(<Input value="controlled value" placeholder="Enter text" />);
    
    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toHaveValue('controlled value');
  });
});