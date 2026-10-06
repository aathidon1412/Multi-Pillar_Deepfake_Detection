import React from 'react';
import { ShieldCheck, Cpu, Scissors, CheckCircle, AlertTriangle } from 'lucide-react';

export default function ProbabilityChart({ classification = {} }) {
  const prediction = classification.prediction || 'REAL';
  const confidence = classification.confidence || 0.85;
  const scores = classification.scores || { real: 0.1, ai_generated: 0.8, forged: 0.1 };

  const getVerdictTheme = (pred) => {
    switch (pred) {
      case 'REAL':
        return {
          bg: 'from-emerald-950/40 to-slate-900/80',
          border: 'border-emerald-500/50',
          badgeBg: 'bg-emerald-900/60',
          badgeText: 'text-emerald-300',
          badgeBorder: 'border-emerald-500/40',
          icon: <ShieldCheck className="w-8 h-8 text-emerald-400" />,
          accent: 'text-emerald-400',
          desc: 'No significant generative manipulation or digital tampering detected.'
        };
      case 'AI_GENERATED':
        return {
          bg: 'from-rose-950/40 to-slate-900/80',
          border: 'border-rose-500/50',
          badgeBg: 'bg-rose-900/60',
          badgeText: 'text-rose-300',
          badgeBorder: 'border-rose-500/40',
          icon: <Cpu className="w-8 h-8 text-rose-400" />,
          accent: 'text-rose-400',
          desc: 'Synthetic generative video, diffusion temporal morphing, or deepfake model patterns identified.'
        };
      case 'FORGED':
        return {
          bg: 'from-amber-950/40 to-slate-900/80',
          border: 'border-amber-500/50',
          badgeBg: 'bg-amber-900/60',
          badgeText: 'text-amber-300',
          badgeBorder: 'border-amber-500/40',
          icon: <Scissors className="w-8 h-8 text-amber-400" />,
          accent: 'text-amber-400',
          desc: 'Non-generative splicing, temporal discontinuities, or digital frame alteration detected.'
        };
      default:
        return {
          bg: 'from-slate-950 to-slate-900',
          border: 'border-slate-800',
          badgeBg: 'bg-slate-800',
          badgeText: 'text-slate-300',
          badgeBorder: 'border-slate-700',
          icon: <AlertTriangle className="w-8 h-8 text-cyan-400" />,
          accent: 'text-cyan-400',
          desc: 'Forensic evaluation complete.'
        };
    }
  };

  const theme = getVerdictTheme(prediction);

  return (
    <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col justify-between">
      
      {/* Verdict Header */}
      <div className={`p-5 rounded-xl bg-gradient-to-r ${theme.bg} border ${theme.border} mb-6 flex items-center justify-between`}>
        <div className="flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-black/40 border border-slate-700/60">
            {theme.icon}
          </div>
          <div>
            <span className="text-xs uppercase font-mono tracking-wider text-slate-400 block mb-0.5">
              FINAL VERDICT
            </span>
            <h2 className={`text-2xl sm:text-3xl font-black tracking-tight ${theme.accent}`}>
              {prediction.replace('_', ' ')}
            </h2>
            <p className="text-xs text-slate-300 mt-1 max-w-sm">
              {theme.desc}
            </p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xs font-mono text-slate-400 uppercase block mb-1">Confidence</span>
          <span className="text-3xl sm:text-4xl font-black text-white font-mono">
            {Math.round(confidence * 100)}%
          </span>
        </div>
      </div>

      {/* Probability Bars */}
      <div className="space-y-4">
        <h4 className="text-xs uppercase font-mono text-slate-400 tracking-wider">
          Class Probability Distribution
        </h4>

        {/* AI GENERATED */}
        <div>
          <div className="flex justify-between items-center text-xs font-mono mb-1.5">
            <span className="flex items-center space-x-1.5 text-slate-200">
              <span className="w-2.5 h-2.5 rounded-sm bg-rose-500"></span>
              <span className="font-semibold">AI GENERATED</span>
            </span>
            <span className="text-rose-400 font-bold">{Math.round((scores.ai_generated || 0) * 100)}%</span>
          </div>
          <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
            <div
              className="h-full bg-gradient-to-r from-rose-600 to-rose-400 rounded-full transition-all duration-700 shadow-glow-rose"
              style={{ width: `${Math.round((scores.ai_generated || 0) * 100)}%` }}
            />
          </div>
        </div>

        {/* REAL */}
        <div>
          <div className="flex justify-between items-center text-xs font-mono mb-1.5">
            <span className="flex items-center space-x-1.5 text-slate-200">
              <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500"></span>
              <span className="font-semibold">REAL / AUTHENTIC</span>
            </span>
            <span className="text-emerald-400 font-bold">{Math.round((scores.real || 0) * 100)}%</span>
          </div>
          <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
            <div
              className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-all duration-700 shadow-glow-emerald"
              style={{ width: `${Math.round((scores.real || 0) * 100)}%` }}
            />
          </div>
        </div>

        {/* FORGED */}
        <div>
          <div className="flex justify-between items-center text-xs font-mono mb-1.5">
            <span className="flex items-center space-x-1.5 text-slate-200">
              <span className="w-2.5 h-2.5 rounded-sm bg-amber-500"></span>
              <span className="font-semibold">DIGITALLY FORGED</span>
            </span>
            <span className="text-amber-400 font-bold">{Math.round((scores.forged || 0) * 100)}%</span>
          </div>
          <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
            <div
              className="h-full bg-gradient-to-r from-amber-600 to-amber-400 rounded-full transition-all duration-700"
              style={{ width: `${Math.round((scores.forged || 0) * 100)}%` }}
            />
          </div>
        </div>

      </div>

    </div>
  );
}
