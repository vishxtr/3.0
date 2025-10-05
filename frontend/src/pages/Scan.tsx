import React, { useState } from 'react'
import Card from '../components/Card'

export default function Scan() {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any|null>(null)
  const [error, setError] = useState<string|null>(null)

  const doScan = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch('/api/verdict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url})
      })
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(txt || 'Server error')
      }
      const data = await res.json()
      setResult(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <Card title="URL scanner">
        <label htmlFor="url-input" className="sr-only">URL</label>
        <div className="flex gap-2">
          <input id="url-input" value={url} onChange={e=>setUrl(e.target.value)} placeholder="https://example.com/login" className="flex-1 bg-gray-900 border border-gray-700 p-3 rounded-md focus:ring-2 focus:ring-brand-500" />
          <button onClick={doScan} disabled={!url || loading} className="px-4 py-2 bg-brand-700 rounded-md">Scan</button>
        </div>
        {loading && <div className="text-sm text-gray-400 mt-2">Scanning…</div>}
        {error && <div className="text-sm text-red-400 mt-2">Error: {error}</div>}
        {result && (
          <div className="mt-4">
            <div className="text-sm text-gray-400">Verdict</div>
            <div className={`mt-2 inline-block px-3 py-1 rounded-full ${result.verdict === 'phish' ? 'bg-red-600' : 'bg-green-600'}`}>{result.verdict.toUpperCase()}</div>
            <div className="text-sm text-gray-400 mt-2">Confidence: {result.confidence}</div>
            <pre className="text-xs mt-2 bg-gray-900 p-3 rounded-md border border-gray-800">{JSON.stringify(result.features, null, 2)}</pre>
          </div>
        )}
      </Card>
    </div>
  )
}