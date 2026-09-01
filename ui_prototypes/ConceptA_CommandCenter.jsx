import React, { useState } from 'react';
import { 
  Globe, 
  Activity, 
  Layers, 
  Eye, 
  Zap, 
  Cpu, 
  Sliders, 
  Maximize2, 
  Search, 
  FileText, 
  Sparkles,
  Compass,
  Radio,
  BarChart2,
  ChevronRight,
  ShieldAlert
} from 'lucide-react';

export default function ConceptA_CommandCenter() {
  const [activeBand, setActiveBand] = useState('optical'); // optical, sar, thermal, infrared
  const [selectedAgent, setSelectedAgent] = useState('all');
  const [queryText, setQueryText] = useState("Detect urban expansion and water body shrinkage around Ahmedabad SAC corridor");

  return (
    <div className="min-h-screen bg-[#07090e] text-zinc-100 font-sans p-4 space-y-4 antialiased selection:bg-cyan-500/30">
      
      {/* Top Tactical Command Bar */}
      <header className="flex items-center justify-between bg-zinc-900/90 border border-zinc-800/80 rounded-2xl px-5 py-3 backdrop-blur-xl shadow-2xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Radio className="h-5 w-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-extrabold text-base tracking-wider text-white uppercase font-mono">
                SATQUERY OPS CENTER
              </h1>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-400 border border-cyan-500/40">
                LIVE TELEMETRY
              </span>
            </div>
            <p className="text-[11px] text-zinc-400">ISRO / SAC Space Applications Corridor • PS 26167</p>
          </div>
        </div>

        {/* Live Satellite Status Badges */}
        <div className="hidden md:flex items-center space-x-4 text-xs font-mono">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-zinc-800/60 border border-zinc-700/60">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
            <span className="text-zinc-400">SENSOR:</span>
            <span className="text-emerald-400 font-bold">Sentinel-2B / S1-SAR</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-zinc-800/60 border border-zinc-700/60">
            <span className="text-zinc-400">CRS:</span>
            <span className="text-cyan-400 font-bold">EPSG:32632</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-zinc-800/60 border border-zinc-700/60">
            <span className="text-zinc-400">LAT/LON:</span>
            <span className="text-amber-400 font-bold">23.0225° N, 72.5714° E</span>
          </div>
        </div>
      </header>

      {/* Main Grid: Left Control Dock (1/4), Center 3D Map Portal (2/4), Right Telemetry (1/4) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        
        {/* LEFT DOCK: Spectral Band & Specialist Agent Routing */}
        <div className="lg:col-span-3 space-y-4">
          
          {/* Spectral Band Selector */}
          <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                <Layers className="h-4 w-4 text-cyan-400" />
                Spectral Modality
              </h3>
              <span className="text-[10px] font-mono text-zinc-500">4 MODES</span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {[
                { id: 'optical', label: 'RGB Optical', icon: Eye, color: 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10' },
                { id: 'sar', label: 'SAR Radar', icon: Radio, color: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10' },
                { id: 'thermal', label: 'Thermal IR', icon: Activity, color: 'text-amber-400 border-amber-500/40 bg-amber-500/10' },
                { id: 'infrared', label: 'NDVI Flora', icon: Zap, color: 'text-purple-400 border-purple-500/40 bg-purple-500/10' }
              ].map((band) => {
                const Icon = band.icon;
                const isActive = activeBand === band.id;
                return (
                  <button
                    key={band.id}
                    onClick={() => setActiveBand(band.id)}
                    className={`p-2.5 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between h-20 ${
                      isActive 
                        ? `${band.color} font-bold shadow-lg shadow-cyan-500/5` 
                        : 'border-zinc-800 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-white'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="text-xs">{band.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Specialist Node Topology */}
          <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-4 space-y-3">
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
              <Cpu className="h-4 w-4 text-emerald-400" />
              Agentic Node Mesh
            </h3>

            <div className="space-y-2 text-xs">
              {[
                { id: 'spec_a', name: 'Specialist A (VQA/RGB)', port: ':8001', status: 'ACTIVE', score: '96.5%', load: '14%' },
                { id: 'spec_b', name: 'Specialist B (Change-VQA)', port: ':8002', status: 'READY', score: '94.8%', load: '8%' },
                { id: 'spec_c', name: 'Specialist C (SAR Fusion)', port: ':8003', status: 'STANDBY', score: '98.1%', load: '0%' }
              ].map((node) => (
                <div 
                  key={node.id}
                  className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800/80 flex items-center justify-between hover:border-zinc-700 transition-colors"
                >
                  <div>
                    <div className="font-semibold text-white">{node.name}</div>
                    <div className="text-[10px] font-mono text-zinc-500">{node.port} • Acc: {node.score}</div>
                  </div>
                  <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                    {node.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* CENTER STAGE: Satellite Observation Frame with Floating Prompt */}
        <div className="lg:col-span-6 space-y-4">
          <div className="relative rounded-2xl overflow-hidden border border-zinc-800 bg-zinc-950 shadow-2xl group min-h-[420px] flex flex-col justify-between p-4">
            
            {/* Mock Satellite Imagery Canvas */}
            <div className="absolute inset-0 bg-cover bg-center opacity-80 transition-transform duration-700 group-hover:scale-105"
                 style={{ backgroundImage: `url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1200')` }}>
              <div className="absolute inset-0 bg-gradient-to-t from-[#07090e] via-transparent to-black/60" />
            </div>

            {/* Tactical Grid Overlay Overlay */}
            <div className="relative z-10 flex items-center justify-between font-mono text-[11px] text-zinc-300">
              <div className="flex items-center space-x-2 px-3 py-1 rounded-xl bg-black/60 backdrop-blur-md border border-white/10">
                <Compass className="h-3.5 w-3.5 text-cyan-400 animate-spin" style={{ animationDuration: '10s' }} />
                <span>GRID: 23.0225° N, 72.5714° E</span>
              </div>
              <div className="flex items-center space-x-2 px-3 py-1 rounded-xl bg-black/60 backdrop-blur-md border border-white/10 text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                <span>ZOOM 15X • RESOLUTION 0.5M/PX</span>
              </div>
            </div>

            {/* Simulated Bounding Box Overlay */}
            <div className="relative z-10 my-auto border-2 border-cyan-400/80 rounded-xl p-4 bg-cyan-500/5 max-w-sm mx-auto backdrop-blur-xs animate-pulse">
              <div className="text-[10px] font-mono text-cyan-400 font-bold mb-1">BOUNDING BOX #1 [240, 180, 410, 390]</div>
              <div className="text-xs text-white font-medium">Urban Construction & Paved Road Expansion (Area: 1.42 km²)</div>
            </div>

            {/* Bottom Floating Prompt Box */}
            <div className="relative z-10 space-y-2">
              <div className="flex items-center space-x-2 bg-zinc-900/90 border border-zinc-700/80 rounded-2xl p-2 pl-4 backdrop-blur-xl shadow-2xl">
                <Search className="h-4 w-4 text-cyan-400 shrink-0" />
                <input
                  type="text"
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  placeholder="Ask any remote sensing query..."
                  className="w-full bg-transparent text-xs text-white placeholder-zinc-500 focus:outline-none font-medium"
                />
                <button className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs transition-all cursor-pointer flex items-center space-x-1 shrink-0">
                  <span>Analyze</span>
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT PANEL: Land Cover Breakdown & Synthesis Intelligence */}
        <div className="lg:col-span-3 space-y-4">
          
          {/* Surface Composition Analytics */}
          <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-4 space-y-3">
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
              <BarChart2 className="h-4 w-4 text-amber-400" />
              Surface Composition (%)
            </h3>

            <div className="space-y-2.5 text-xs">
              {[
                { name: 'Built-up / Urban', percent: 48, color: 'bg-amber-400' },
                { name: 'Dense Vegetation', percent: 32, color: 'bg-emerald-400' },
                { name: 'Water Bodies', percent: 12, color: 'bg-cyan-400' },
                { name: 'Barren Land', percent: 8, color: 'bg-zinc-500' }
              ].map((item, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="text-zinc-400">{item.name}</span>
                    <span className="text-white font-bold">{item.percent}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                    <div className={`h-full ${item.color} rounded-full`} style={{ width: `${item.percent}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Executive Synthesis Card */}
          <div className="bg-gradient-to-b from-cyan-950/30 to-zinc-900/90 border border-cyan-500/30 rounded-2xl p-4 space-y-2">
            <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs font-bold">
              <Sparkles className="h-4 w-4" />
              <span>SYNTHESIS OVERVIEW</span>
            </div>
            <p className="text-xs text-zinc-300 leading-relaxed font-sans">
              High-resolution optical VQA confirms rapid urban infill with 14% increase in paved transportation corridors over the 5-year observation baseline.
            </p>
            <div className="pt-2 flex items-center justify-between text-[10px] font-mono text-zinc-500 border-t border-zinc-800">
              <span>CONFIDENCE: 96.8%</span>
              <span>LATENCY: 184ms</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
