// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react';
import { CardProps } from '../../types';
import { clsx } from 'clsx';

const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  icon,
  actions,
  className,
  children,
  ...props
}) => {
  return (
    <div
      className={clsx(
        'bg-gray-800 rounded-lg border border-gray-700 p-6',
        className
      )}
      {...props}
    >
      {(title || subtitle || icon || actions) && (
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            {icon && (
              <div className="flex-shrink-0">
                {icon}
              </div>
            )}
            <div>
              {title && (
                <h3 className="text-lg font-semibold text-white">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="text-sm text-gray-400">
                  {subtitle}
                </p>
              )}
            </div>
          </div>
          {actions && (
            <div className="flex-shrink-0">
              {actions}
            </div>
          )}
        </div>
      )}
      {children}
    </div>
  );
};

export default Card;