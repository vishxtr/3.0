import React, { useEffect, useState } from 'react'
import Card from '../components/Card'
import { motion } from 'framer-motion'

export default function Dashboard() {
  const [recent, setRecent] = useState<any[]>([])
  useEffect(() => {
    // demo-only: load demo_campaigns.json locally
    fetch('/data/demo_campaigns.json').then(r => r.json()).then(d => {
      setRecent(d)
    }).catch(() => {
      setRecent([])
    })
  }, [])
  return (
    <div className="grid gap-6 grid-cols-1 md:grid-cols-3">
      <motion.div initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
        <Card title="Verdicts today">
          <div className="text-3xl font-semibold">24</div>
          <div className="text-sm text-gray-400">Threats flagged (demo)</div>
        </Card>
      </motion.div>

      <motion.div initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:0.12}}>
        <Card title="Live feed">
          <div className="space-y-2">
            {recent.map((r,i) => (
              <div key={i} className="text-sm text-gray-300">{r.name} • {r.campaign_id}</div>
            ))}
            {recent.length === 0 && <div className="text-sm text-gray-500">No recent campaigns (demo)</div>}
          </div>
        </Card>
      </motion.div>

      <motion.div initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:0.24}}>
        <Card title="Quick actions">
          <div className="flex gap-2">
            <a href="/scan" className="px-3 py-2 bg-brand-700 rounded-md text-white">Scan URL</a>
            <a href="/model" className="px-3 py-2 border rounded-md text-gray-200">Model insights</a>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}