import React from 'react';
import { ShieldCheck, History, Film, Image as ImageIcon, Mic, FileText, Cpu } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand */}
        <div 
          onClick={() => setActiveTab('video')}
          className="flex items-center space-x-3 cursor-pointer group shrink-0"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-glow-cyan">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-base sm:text-lg text-white tracking-tight group-hover:text-cyan-400 transition-colors">
                AuthentiGuard AI
              </span>
              <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-400 border border-cyan-800/60 font-mono">
                5 Pillars
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono hidden sm:block">Unified Deepfake & Forgery Detection</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 sm:space-x-1.5 overflow-x-auto py-1">
          <button
            onClick={() => setActiveTab('video')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
              activeTab === 'video' || activeTab === 'processing' || activeTab === 'result'
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Film className="w-3.5 h-3.5 text-cyan-400" />
            <span>Pillar 2: Video</span>
          </button>

          <button
            onClick={() => setActiveTab('image')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
              activeTab === 'image'
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5 text-blue-400" />
            <span>Pillars 1 & 5: Image</span>
          </button>

          <button
            onClick={() => setActiveTab('audio')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
              activeTab === 'audio'
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Mic className="w-3.5 h-3.5 text-purple-400" />
            <span>Pillar 3: Audio</span>
          </button>

          <button
            onClick={() => setActiveTab('document')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
              activeTab === 'document'
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <FileText className="w-3.5 h-3.5 text-emerald-400" />
            <span>Pillar 4: Document</span>
          </button>

          <button
            onClick={() => setActiveTab('history')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
              activeTab === 'history'
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <History className="w-3.5 h-3.5 text-amber-400" />
            <span>History</span>
          </button>

          <button
            onClick={() => setActiveTab('architecture')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
              activeTab === 'architecture'
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Cpu className="w-3.5 h-3.5 text-slate-400" />
            <span>Architecture</span>
          </button>
        </nav>

      </div>
    </header>
  );
}
