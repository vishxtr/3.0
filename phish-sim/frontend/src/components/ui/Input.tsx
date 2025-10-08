// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react';
import { InputProps } from '../../types';
import { clsx } from 'clsx';

const Input: React.FC<InputProps> = ({
  type = 'text',
  placeholder,
  value,
  onChange,
  disabled = false,
  error,
  label,
  required = false,
  className,
  ...props
}) => {
  const baseClasses = 'block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variantClasses = {
    text: 'bg-gray-700 border-gray-600 text-white',
    email: 'bg-gray-700 border-gray-600 text-white',
    password: 'bg-gray-700 border-gray-600 text-white',
    url: 'bg-gray-700 border-gray-600 text-white',
    textarea: 'bg-gray-700 border-gray-600 text-white resize-none',
  };

  const errorClasses = error ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '';

  const inputElement = type === 'textarea' ? (
    <textarea
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      required={required}
      className={clsx(
        baseClasses,
        variantClasses.textarea,
        errorClasses,
        className
      )}
      {...props}
    />
  ) : (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      required={required}
      className={clsx(
        baseClasses,
        variantClasses[type],
        errorClasses,
        className
      )}
      {...props}
    />
  );

  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-sm font-medium text-gray-300">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      {inputElement}
      {error && (
        <p className="text-sm text-red-500">{error}</p>
      )}
    </div>
  );
};

export default Input;