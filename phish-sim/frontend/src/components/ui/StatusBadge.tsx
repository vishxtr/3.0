// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react';
import { clsx } from 'clsx';

interface StatusBadgeProps {
  status: 'healthy' | 'degraded' | 'down' | 'initializing' | 'connected' | 'disconnected';
  className?: string;
  showIcon?: boolean;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  className,
  showIcon = true,
}) => {
  const statusConfig = {
    healthy: {
      label: 'Healthy',
      classes: 'bg-green-100 text-green-800 border-green-200',
      icon: '✓',
    },
    degraded: {
      label: 'Degraded',
      classes: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      icon: '⚠',
    },
    down: {
      label: 'Down',
      classes: 'bg-red-100 text-red-800 border-red-200',
      icon: '✗',
    },
    initializing: {
      label: 'Initializing',
      classes: 'bg-blue-100 text-blue-800 border-blue-200',
      icon: '⏳',
    },
    connected: {
      label: 'Connected',
      classes: 'bg-green-100 text-green-800 border-green-200',
      icon: '✓',
    },
    disconnected: {
      label: 'Disconnected',
      classes: 'bg-red-100 text-red-800 border-red-200',
      icon: '✗',
    },
  };

  const config = statusConfig[status];

  return (
    <span
      className={clsx(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border',
        config.classes,
        className
      )}
    >
      {showIcon && (
        <span className="mr-1">{config.icon}</span>
      )}
      {config.label}
    </span>
  );
};

export default StatusBadge;