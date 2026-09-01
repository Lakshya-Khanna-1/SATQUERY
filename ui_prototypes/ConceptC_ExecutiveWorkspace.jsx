import React, { useState } from 'react';
import { 
  FileText, 
  Sparkles, 
  Layers, 
  MapPin, 
  Send, 
  ChevronDown, 
  ChevronUp, 
  Download, 
  Check, 
  Globe, 
  Compass, 
  Terminal,
  ShieldCheck
} from 'lucide-react';

export default function ConceptC_ExecutiveWorkspace() {
  const [activeTab, setActiveTab] = useState('report');
  const [traceOpen, setTraceOpen] = useState(true);

  return (
    <div className="min-h-screen bg-[#fcfcfd] dark:bg-[#09090b] text-zinc-900 dark:text-zinc-100 font-sans p-6 space-y-6 antialiased selection:bg-cyan-500/20">
      
      {/* Executive Clean Header */}
      <header className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800/80 pb-4">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-black text-sm shadow-md">
            SQ
          </div>
          <div>
            <h1 className="font-bold text-base text-zinc-900 dark:text-white font-sans tracking-tight">
              SatQuery Executive Intelligence Suite
            </h1>
            <p className="text-xs text-zinc-500">ISRO / Space Applications Centre • Auditable EO Intelligence Report</p>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs">
          <button className="px-4 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800/80 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-200 font-semibold transition-all cursor-pointer">
            New Workspace Session
          </button>
          <button className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold transition-all shadow-md cursor-pointer flex items-center space-x-1.5">
            <Download className="h-3.5 w-3.5" />
            <span>Generate Executive PDF</span>
          </button>
        </div>
      </header>

      {/* Main Dual-Column Content */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT 7-COLS: Hero Satellite Stage & Primary Answer */}
        <div className="lg:col-span-7 space-y-5">
          
          {/* Main Visual Stage */}
          <div className="relative rounded-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-950 shadow-xl group">
            <img 
              src="https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1200" 
              alt="Satellite Scene" 
              className="w-full h-80 object-cover"
            />
            
            {/* Top Coordinate Badge */}
            <div className="absolute top-3 left-3 px-3 py-1 rounded-xl bg-black/70 backdrop-blur-md border border-white/20 text-xs font-mono font-semibold text-cyan-400">
              ISRO SAC Corridor • 23.0225° N, 72.5714° E
            </div>
            <div className="absolute top-3 right-3 px-3 py-1 rounded-xl bg-black/70 backdrop-blur-md border border-white/20 text-xs font-mono font-semibold text-emerald-400">
              CONFIDENCE 96.8%
            </div>
          </div>

          {/* Synthesis Answer Card */}
          <div className="p-6 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm space-y-4">
            <div className="flex items-center space-x-2 text-xs font-bold text-cyan-600 dark:text-cyan-400 uppercase tracking-wider font-mono">
              <Sparkles className="h-4 w-4" />
              <span>Executive Analytical Findings</span>
            </div>

            <div className="space-y-3 text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed font-sans">
              <p>
                <strong>Observation Summary</strong>: The optical satellite observation displays a vibrant commercial transportation corridor bordered by industrial facilities, paved access roads, and distinct white circular storage structures.
              </p>
              <ul className="list-disc pl-5 space-y-1.5 text-xs text-zinc-600 dark:text-zinc-400">
                <li>Built-up infrastructure accounts for 62% of the scene footprint.</li>
                <li>Vegetation and open soil represent the remaining 38%.</li>
                <li>No significant environmental degradation detected in adjacent agricultural land.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* RIGHT 5-COLS: Auditable Execution Trace & Technical Parameters */}
        <div className="lg:col-span-5 space-y-5">
          
          {/* Collapsible Auditable Trace Card */}
          <div className="rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm overflow-hidden">
            <div 
              onClick={() => setTraceOpen(!traceOpen)}
              className="p-4 flex items-center justify-between cursor-pointer border-b border-zinc-100 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800/40 transition-colors"
            >
              <div className="flex items-center space-x-2 font-mono text-xs font-bold text-zinc-900 dark:text-white">
                <Terminal className="h-4 w-4 text-emerald-500" />
                <span>AUDITABLE EXECUTION TRACE</span>
              </div>
              {traceOpen ? <ChevronUp className="h-4 w-4 text-zinc-400" /> : <ChevronDown className="h-4 w-4 text-zinc-400" />}
            </div>

            {traceOpen && (
              <div className="p-4 space-y-3 font-mono text-xs text-zinc-600 dark:text-zinc-400 bg-zinc-50/50 dark:bg-zinc-950/40">
                <div className="flex justify-between border-b border-zinc-200/60 dark:border-zinc-800 pb-2">
                  <span>Task Intent:</span>
                  <span className="text-cyan-600 dark:text-cyan-400 font-bold">Single-Image VQA</span>
                </div>
                <div className="flex justify-between border-b border-zinc-200/60 dark:border-zinc-800 pb-2">
                  <span>Specialists Invoked:</span>
                  <span className="text-zinc-900 dark:text-white">Specialist A (Qwen2.5-VL)</span>
                </div>
                <div className="flex justify-between border-b border-zinc-200/60 dark:border-zinc-800 pb-2">
                  <span>Spatial Alignment:</span>
                  <span className="text-emerald-600 dark:text-emerald-400 font-bold">100% Compliant (CRS OK)</span>
                </div>
                <div className="flex justify-between">
                  <span>Processing Latency:</span>
                  <span className="text-amber-600 dark:text-amber-400 font-bold">184.2 ms</span>
                </div>
              </div>
            )}
          </div>

          {/* Compliance & Hardware Telemetry Badge */}
          <div className="p-4 rounded-2xl bg-zinc-100/70 dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800 space-y-2 text-xs font-mono">
            <div className="flex items-center space-x-2 text-emerald-600 dark:text-emerald-400 font-bold">
              <ShieldCheck className="h-4 w-4" />
              <span>ISRO SIH 2026 Audit Compliant</span>
            </div>
            <p className="text-[11px] text-zinc-500 leading-normal">
              Validated on RTX A2000 12GB CUDA hardware using 4-bit QLoRA quantized vision-language adapters.
            </p>
          </div>

        </div>

      </div>
    </div>
  );
}
