import React from 'react';
import { Cpu, HardDrive, ShieldCheck, Activity, Mic, Layers, ArrowDown } from 'lucide-react';

export default function ArchitecturePage() {
  return (
    <div className="max-w-5xl mx-auto py-8 space-y-10">
      
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-800/50 text-xs font-mono text-cyan-400">
          <Cpu className="w-3.5 h-3.5" />
          <span>SYSTEM ARCHITECTURE & TECHNICAL SPECIFICATIONS</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white">
          Multi-Pillar Video Authenticity Engine
        </h1>
        <p className="text-slate-400 text-sm max-w-2xl mx-auto font-mono">
          USMFE Pillar 2: Combining Visual Spatial ViT, Biological rPPG, Temporal Stability, Audio Acoustics, and Lip-Sync Coherence.
        </p>
      </div>

      {/* Workflow Diagram Card */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <h3 className="text-sm font-bold font-mono text-cyan-400 uppercase tracking-wider">
          End-to-End Analysis Workflow
        </h3>
        
        <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 font-mono text-xs text-slate-300 overflow-x-auto leading-relaxed">
          <pre className="text-cyan-300">
{`                        USER
                          │
                          ▼
                 ┌─────────────────┐
                 │ Upload Video    │
                 └────────┬────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ File Validation     │
               │ • Format            │
               │ • Size              │
               │ • Duration & FPS    │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Save Original Video │  (storage/uploads/VID_xxx.mp4)
               └──────────┬──────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Video Preprocessing      │
             │ • Metadata extraction    │
             │ • Frame extraction       │
             │ • Audio extraction       │
             └────────────┬─────────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Visual   │ │ Temporal │ │  Audio   │
       │ Analysis │ │ Analysis │ │ Analysis │
       └────┬─────┘ └────┬─────┘ └────┬─────┘
            │            │            │
            └────────────┼────────────┘
                         ▼
               ┌─────────────────────┐
               │ Feature Fusion      │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Classification Model│
               └──────────┬──────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
          REAL      AI_GENERATED      FORGED
            │             │             │
            └─────────────┼─────────────┘
                          ▼
               ┌─────────────────────┐
               │ Explainability      │
               │ • Suspicious frames │
               │ • Forensic reasons  │
               │ • Timestamp markers │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ JSON File Storage   │  (storage/results/VID_xxx.json)
               └──────────┬──────────┘
                          │
                          ▼
                   Web Dashboard`}
          </pre>
        </div>
      </div>

      {/* JSON Storage Architecture */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex items-center space-x-2">
          <HardDrive className="w-5 h-5 text-emerald-400" />
          <h3 className="text-base font-bold text-white font-mono uppercase tracking-wider">
            Zero-Database JSON Storage Strategy
          </h3>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed font-mono">
          Instead of introducing MySQL, PostgreSQL, or Mongo, the system utilizes an atomic, localized JSON document storage system in <code className="text-cyan-400">storage/results/{'{video_id}'}.json</code>. Each record contains the video parameters, complete 5-pillar telemetry, model confidence, and annotated suspicious frame references.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-cyan-400 block font-bold mb-1">storage/uploads/</span>
            <span className="text-slate-400">Original unaltered video files preserved permanently.</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-rose-400 block font-bold mb-1">storage/suspicious_frames/</span>
            <span className="text-slate-400">Flagged anomaly frames with visual bounding boxes and badges.</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-amber-400 block font-bold mb-1">storage/frames/</span>
            <span className="text-slate-400">Temporary raw sampled frames, cleaned up immediately after analysis.</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-emerald-400 block font-bold mb-1">storage/results/</span>
            <span className="text-slate-400">Permanent atomic JSON reports queried directly for history.</span>
          </div>
        </div>
      </div>

    </div>
  );
}
