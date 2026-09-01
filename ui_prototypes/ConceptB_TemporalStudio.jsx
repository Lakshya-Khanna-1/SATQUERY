import React, { useState } from 'react';
import { 
  Sliders, 
  ArrowLeftRight, 
  Clock, 
  Layers, 
  Sparkles, 
  Download, 
  Maximize2, 
  ChevronRight, 
  CheckCircle2,
  TrendingUp,
  MapPin
} from 'lucide-react';

export default function ConceptB_TemporalStudio() {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [activeTab, setActiveTab] = useState('bitemporal');
  const [overlayOpacity, setOverlayOpacity] = useState(70);

  return (
    <div className="min-h-screen bg-[#090b10] text-zinc-100 font-sans p-5 space-y-5 antialiased selection:bg-emerald-500/30">
      
      {/* Top Header & Mission Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-zinc-900/90 border border-zinc-800 rounded-2xl p-4 backdrop-blur-xl">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              BI-TEMPORAL STUDIO
            </span>
            <h1 className="font-bold text-lg text-white font-sans">
              Geospatial Change Detection & Spectral Inspector
            </h1>
          </div>
          <p className="text-xs text-zinc-400 mt-1">Comparing Baseline (2021) vs Present Scene (2026) • High-Res Earth Observation</p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          <button className="px-3.5 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-xs font-semibold text-zinc-200 transition-all cursor-pointer flex items-center space-x-2 border border-zinc-700">
            <Download className="h-3.5 w-3.5" />
            <span>Export Intelligence PDF</span>
          </button>
          <button className="px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black font-bold text-xs transition-all cursor-pointer shadow-lg shadow-emerald-500/10">
            Run Specialist B
          </button>
        </div>
      </div>

      {/* Hero Interactive Split-Curtain Comparison Stage */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        
        {/* LEFT 8-COLS: Interactive Image Comparison Canvas */}
        <div className="lg:col-span-8 space-y-3">
          
          <div className="relative h-[480px] w-full rounded-2xl overflow-hidden border border-zinc-800 bg-zinc-950 shadow-2xl group select-none">
            
            {/* Time 2 (Present - Right Base) */}
            <img 
              src="https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1200" 
              alt="Time 2 Present Observation" 
              className="absolute inset-0 h-full w-full object-cover"
            />
            <div className="absolute top-4 right-4 z-20 px-3 py-1 rounded-xl bg-black/70 backdrop-blur-md border border-white/20 text-xs font-mono font-bold text-emerald-400">
              TIME 2: PRESENT DAY (2026)
            </div>

            {/* Time 1 (Historical - Clipped Left) */}
            <div 
              className="absolute inset-y-0 left-0 overflow-hidden z-10"
              style={{ width: `${sliderPosition}%` }}
            >
              <img 
                src="https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1200" 
                alt="Time 1 Baseline Observation" 
                className="absolute inset-0 h-full w-full object-cover max-w-none"
                style={{ width: '100%', height: '100%' }}
              />
              <div className="absolute top-4 left-4 z-20 px-3 py-1 rounded-xl bg-black/70 backdrop-blur-md border border-white/20 text-xs font-mono font-bold text-cyan-400">
                TIME 1: HISTORICAL BASELINE (2021)
              </div>
            </div>

            {/* Interactive Vertical Curtain Slider Line */}
            <div 
              className="absolute inset-y-0 z-30 w-1 bg-white/90 shadow-2xl cursor-ew-resize flex items-center justify-center"
              style={{ left: `${sliderPosition}%` }}
            >
              <div className="h-10 w-10 rounded-full bg-white text-zinc-900 shadow-2xl flex items-center justify-between px-1.5 border-2 border-emerald-500">
                <ArrowLeftRight className="h-4 w-4 mx-auto" />
              </div>
            </div>

            {/* Range Input Triggering Slider Movement */}
            <input 
              type="range" 
              min="0" 
              max="100" 
              value={sliderPosition} 
              onChange={(e) => setSliderPosition(e.target.value)} 
              className="absolute inset-0 opacity-0 cursor-ew-resize z-40"
            />
          </div>

          {/* Slider Slider Instruction & Opacity Bar */}
          <div className="flex items-center justify-between text-xs font-mono text-zinc-400 px-1">
            <span>Drag slider horizontally to compare temporal transitions</span>
            <div className="flex items-center space-x-3">
              <span>Change Mask Opacity:</span>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={overlayOpacity} 
                onChange={(e) => setOverlayOpacity(e.target.value)}
                className="w-24 accent-emerald-500"
              />
              <span className="text-white font-bold">{overlayOpacity}%</span>
            </div>
          </div>
        </div>

        {/* RIGHT 4-COLS: Detailed Analytical Metrics & Change Log */}
        <div className="lg:col-span-4 space-y-4">
          
          {/* Change Statistics Card */}
          <div className="bg-zinc-900/90 border border-zinc-800 rounded-2xl p-4 space-y-3">
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              Bi-Temporal Dynamics Log
            </h3>

            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800">
                <div className="text-zinc-500 text-[10px]">TOTAL CHANGE AREA</div>
                <div className="text-base font-bold text-emerald-400 mt-1">1.48 km²</div>
                <div className="text-[10px] text-zinc-400 mt-0.5">+12.4% vs baseline</div>
              </div>

              <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800">
                <div className="text-zinc-500 text-[10px]">CONFIDENCE SCORE</div>
                <div className="text-base font-bold text-cyan-400 mt-1">94.8%</div>
                <div className="text-[10px] text-zinc-400 mt-0.5">Specialist B LoRA</div>
              </div>
            </div>
          </div>

          {/* Detailed Change Narrative */}
          <div className="bg-zinc-900/90 border border-zinc-800 rounded-2xl p-4 space-y-3">
            <div className="flex items-center space-x-2 text-xs font-bold text-zinc-200 uppercase tracking-wider">
              <Sparkles className="h-4 w-4 text-amber-400" />
              <span>AI Change Narrative</span>
            </div>

            <div className="space-y-2 text-xs text-zinc-300 leading-relaxed font-sans">
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span><strong className="text-white">Forest Canopy Removal</strong>: 0.85 km² of dense vegetation cleared along the central sector.</span>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
                <span><strong className="text-white">Road Corridor Construction</strong>: New dual-lane asphalt transportation route spanning 2.4 km.</span>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                <span><strong className="text-white">Residential Expansion</strong>: 14 new housing structures detected in the lower right quadrant.</span>
              </div>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
