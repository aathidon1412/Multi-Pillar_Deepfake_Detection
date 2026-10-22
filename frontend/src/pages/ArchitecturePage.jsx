import React from 'react';
import { Cpu, HardDrive, ShieldCheck, Activity, Mic, Layers, ArrowDown } from 'lucide-react';

export default function ArchitecturePage() {
  return (
    <div className="max-w-5xl mx-auto py-8 px-4 sm:px-6 space-y-8">
      
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs font-mono text-slate-700">
          <Cpu className="w-3.5 h-3.5" />
          <span className="uppercase tracking-wider font-semibold">TECHNICAL SPECIFICATIONS</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900 tracking-tight">
          Multi-Pillar Video Authenticity Engine Architecture
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 max-w-2xl mx-auto">
          USMFE Pillar 2: Combining Visual Spatial ViT, Biological rPPG, Temporal Stability, Audio Acoustics, and Lip-Sync Coherence.
        </p>
      </div>

      {/* Workflow Diagram Card */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <h3 className="text-xs font-semibold font-mono text-slate-800 uppercase tracking-wider">
          End-to-End Pipeline Execution Topology
        </h3>
        
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 font-mono text-xs text-slate-700 overflow-x-auto leading-relaxed">
          <pre className="text-slate-800">
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
       │ Visual   │ │ Temporal │ │ Audio    │
       │ (ViT)    │ │ Optical  │ │ Acoustic │
       └────┬─────┘ └────┬─────┘ └────┬─────┘
            │            │            │
            └────────────┼────────────┘
                         ▼
             ┌───────────────────────┐
             │ Lip-Sync SyncNet      │
             │ Correlation Engine    │
             └───────────┬───────────┘
                         ▼
             ┌───────────────────────┐
             │ Multi-Pillar Ensemble │
             │ & Feature Fusion      │
             └───────────┬───────────┘
                         ▼
             ┌───────────────────────┐
             │ JSON Dossier Output   │
             │ (storage/results/     │
             │  VID_xxx.json)        │
             └───────────────────────┘`}
          </pre>
        </div>
      </div>

      {/* Pillar Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-slate-900" />
            <h4 className="text-sm font-semibold text-slate-900">Pillar 1 & 5: Visual ViT & Spatial ELA</h4>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            Hugging Face ViT base patch16 models classify deepfake artifacts and generative noise patterns. Error Level Analysis (ELA) identifies JPEG resaving inconsistencies and composite bounding boundaries.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-slate-900" />
            <h4 className="text-sm font-semibold text-slate-900">Pillar 2: Temporal Consistency & Optical Flow</h4>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            Farnebäck dense optical flow tracks pixel displacement across adjacent frames to identify unnatural temporal jitter, flicker, and blending seams across face borders.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-slate-900" />
            <h4 className="text-sm font-semibold text-slate-900">Pillar 3: Audio Acoustics & Demixing</h4>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            Wav2Vec2 / Audio Spectrogram Transformer feature vectors evaluate synthetic vocoder harmonics and robotic unnatural cadence with optional HPSS vocal isolation.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-slate-900" />
            <h4 className="text-sm font-semibold text-slate-900">Pillar 4: Document PDF Structure & Revision Trees</h4>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            XREF stream table analysis, font descriptor integrity verification, and metadata modification audit trails detect digital invoice, contract, and PDF forgery.
          </p>
        </div>

      </div>

    </div>
  );
}
