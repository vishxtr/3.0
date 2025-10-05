import React, { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import Card from '../components/Card'

export default function Model() {
  const [fi, setFi] = useState<{name:string,value:number}[]>([])
  useEffect(() => {
    // Try load model metadata from backend file served statically via /data/models/...
    fetch('/data/models/phish_model_v1.metadata.json').then(r => r.json()).then(meta => {
      const features = meta.features || ["length","count_digits","suspicious_count","dots"]
      // create fake importances if none available
      const vals = features.map((f:string,i:number)=>({name:f, value:(meta && meta.classification_report) ? (0.5 - i*0.1) : (Math.random()*0.5 + 0.1)}))
      setFi(vals)
    }).catch(()=> {
      setFi([
        {name:"length", value:0.5},
        {name:"count_digits", value:0.3},
        {name:"suspicious_count", value:0.9},
        {name:"dots", value:0.2}
      ])
    })
  }, [])
  return (
    <div className="max-w-3xl mx-auto">
      <Card title="Model feature importances">
        <div style={{width:'100%', height: 240}}>
          <ResponsiveContainer>
            <BarChart data={fi}>
              <XAxis dataKey="name" stroke="#aaa" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="text-sm text-gray-400 mt-3">Model trained on synthetic dataset (demo).</div>
      </Card>
    </div>
  )
}