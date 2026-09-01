import React, { useState } from 'react';
import ConceptA_CommandCenter from './ConceptA_CommandCenter';
import ConceptB_TemporalStudio from './ConceptB_TemporalStudio';
import ConceptC_ExecutiveWorkspace from './ConceptC_ExecutiveWorkspace';
import { Sparkles, Layers, ShieldCheck, ArrowRight, Eye, Layout } from 'lucide-react';

export default function Showcase() {
  const [activeConcept, setActiveConcept] = useState('concept_a');

  return (
    <div className="min-h-screen bg-[#07090e] text-zinc-100 font-sans flex flex-col antialiased">
      
      {/* Top Concept Switcher Banner */}
      <div className="bg-zinc-900 border-b border-zinc-800 p-3 px-6 sticky top-0 z-50 flex items-center justify-between shadow-2xl backdrop-blur-xl">
        <div className="flex items-center space-x-3">
          <div className="p-1.5 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/40">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-xs font-bold font-mono tracking-wider text-white uppercase">
              IMPECCABLE UI DESIGN EXPLORATORIUM
            </h2>
            <p className="text-[10px] text-zinc-400">SIH 2026 Earth Observation Studio • Uncommitted Prototype Branch</p>
          </div>
        </div>

        {/* Concept Selector Buttons */}
        <div className="flex items-center space-x-2 text-xs">
          {[
            { id: 'concept_a', label: 'Concept A: Command Center', desc: 'Tactical Multi-Spectral Dashboard' },
            { id: 'concept_b', label: 'Concept B: Bi-Temporal Lens', desc: 'Curtain Slider Change Inspector' },
            { id: 'concept_c', label: 'Concept C: Executive Workspace', desc: 'Minimalist Content Suite' }
          ].map((c) => (
            <button
              key={c.id}
              onClick={() => setActiveConcept(c.id)}
              className={`px-3.5 py-1.5 rounded-xl border font-medium transition-all cursor-pointer ${
                activeConcept === c.id
                  ? 'bg-cyan-500 text-black font-bold border-cyan-400 shadow-lg shadow-cyan-500/20'
                  : 'bg-zinc-800/80 border-zinc-700/80 text-zinc-300 hover:bg-zinc-700 hover:text-white'
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>

      {/* Concept Render Surface */}
      <div className="flex-1">
        {activeConcept === 'concept_a' && <ConceptA_CommandCenter />}
        {activeConcept === 'concept_b' && <ConceptB_TemporalStudio />}
        {activeConcept === 'concept_c' && <ConceptC_ExecutiveWorkspace />}
      </div>
    </div>
  );
}
