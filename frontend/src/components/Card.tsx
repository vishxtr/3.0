import React from 'react'

export default function Card({ children, title }: { children: React.ReactNode, title?: string }) {
  return (
    <div className="bg-gray-800 p-4 rounded-2xl shadow-lg border border-gray-700">
      {title && <div className="text-sm text-gray-400 mb-2">{title}</div>}
      <div>{children}</div>
    </div>
  )
}