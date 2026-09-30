import React from 'react';
import { ShieldCheck, History, Sparkles } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        
        {/* Brand Identity */}
        <div 
          onClick={() => setActiveTab('unified')}
          className="flex items-center space-x-3 cursor-pointer group shrink-0"
        >
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-base tracking-wider text-slate-900 uppercase">USMFE</span>
            <span className="h-4 w-px bg-slate-200"></span>
            <span className="text-[11px] font-mono uppercase tracking-wider bg-slate-100 text-slate-600 px-2 py-0.5 rounded border border-slate-200 font-medium">
              Universal Forensics Engine
            </span>
          </div>
        </div>

        {/* Navigation Tabs (Unified Workspace, Audit History) */}
        <nav className="flex items-center space-x-1 sm:space-x-1.5 overflow-x-auto py-1">
          <button
            onClick={() => setActiveTab('unified')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
              activeTab === 'unified' || activeTab === 'processing' || activeTab === 'result'
                ? 'bg-slate-900 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Universal Workspace</span>
          </button>

          <button
            onClick={() => setActiveTab('history')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
              activeTab === 'history'
                ? 'bg-slate-900 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            <span>Audit Ledger</span>
          </button>
        </nav>

        {/* Pipeline Status Indicator */}
        <div className="hidden md:flex items-center space-x-2">
          <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 bg-emerald-50 border border-emerald-200 rounded-full">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-600"></span>
            </span>
            <span className="text-[11px] font-mono uppercase tracking-wider text-emerald-700 font-semibold">
              All 5 Pillars Active
            </span>
          </div>
        </div>

      </div>
    </header>
  );
}
