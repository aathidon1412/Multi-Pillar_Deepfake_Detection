import React, { useEffect, useState } from 'react';
import { CheckCircle2, AlertCircle, Cpu, Film, Eye, Activity, Mic, MessageSquare, Layers } from 'lucide-react';
import { checkStatus } from '../services/api';

export default function ProcessingPage({ videoId, filename, onProcessingComplete, onCancel }) {
  const [statusData, setStatusData] = useState({
    status: 'processing',
    progress: 5,
    current_stage: 'Initializing Forensic Pipeline',
    error: null,
  });

  const pipelineStages = [
    { id: 'prep', label: 'Video Preprocessing', threshold: 10 },
    { id: 'meta', label: 'Metadata & Format Validation', threshold: 20 },
    { id: 'frames', label: 'Keyframe Extraction & Normalization', threshold: 30 },
    { id: 'face', label: 'Facial Landmark & Biometric Detection', threshold: 45 },
    { id: 'visual', label: 'ViT Spatial & Diffusion Seam Analysis', threshold: 60 },
    { id: 'temporal', label: 'Temporal Consistency & Optical Flow', threshold: 72 },
    { id: 'audio', label: 'Acoustic Signal & Vocoder Forensics', threshold: 80 },
    { id: 'lipsync', label: 'Audio-Visual Lip-Sync Cross-Correlation', threshold: 86 },
    { id: 'class', label: 'Multi-Pillar Feature Fusion & Ensemble', threshold: 92 },
    { id: 'report', label: 'Compiling Forensic Dossier & Evidence', threshold: 98 },
  ];

  useEffect(() => {
    if (!videoId) return;

    let isSubscribed = true;
    const pollInterval = setInterval(async () => {
      try {
        const res = await checkStatus(videoId);
        if (!isSubscribed) return;

        setStatusData(res);

        if (res.status === 'completed') {
          clearInterval(pollInterval);
          setTimeout(() => {
            if (onProcessingComplete) onProcessingComplete(videoId);
          }, 800);
        } else if (res.status === 'failed') {
          clearInterval(pollInterval);
        }
      } catch (err) {
        console.error('Status poll error:', err);
      }
    }, 900);

    return () => {
      isSubscribed = false;
      clearInterval(pollInterval);
    };
  }, [videoId, onProcessingComplete]);

  const progress = statusData.progress || 5;
  const currentStage = statusData.current_stage || 'Analyzing';
  const isFailed = statusData.status === 'failed';

  return (
    <div className="w-full max-w-3xl mx-auto py-12 px-4 sm:px-6 flex flex-col items-center">
      
      {/* Header */}
      <div className="text-center max-w-lg mb-8">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-mono mb-3">
          <span className="w-1.5 h-1.5 rounded-full bg-slate-900 animate-pulse"></span>
          <span className="uppercase tracking-wider font-semibold">Active Pipeline Execution</span>
        </div>
        <h2 className="text-2xl font-semibold text-slate-900 tracking-tight">
          Executing Multi-Pillar Forensic Analysis
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 mt-1 font-mono truncate">
          Target File: {filename || `VID_${videoId}.mp4`}
        </p>
      </div>

      {/* Main Processing Card */}
      <div className="w-full bg-white border border-slate-200 rounded-xl p-6 shadow-sm flex flex-col gap-6">
        
        {/* Progress Bar & Readout */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-700 font-medium">{currentStage}</span>
            <span className="text-slate-900 font-bold">{progress}%</span>
          </div>
          <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
            <div
              className={`h-full transition-all duration-300 ${
                isFailed ? 'bg-rose-600' : 'bg-slate-900'
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Pipeline Stage Checklist */}
        <div className="space-y-1.5 border-t border-slate-100 pt-4">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 font-semibold block mb-2">
            Pipeline Diagnostic Stages
          </span>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {pipelineStages.map((stage) => {
              const isDone = progress >= stage.threshold;
              const isCurrent = !isDone && progress >= (stage.threshold - 15);

              return (
                <div
                  key={stage.id}
                  className={`flex items-center gap-2.5 p-2 rounded-lg text-xs transition-colors ${
                    isDone 
                      ? 'bg-slate-50 text-slate-800' 
                      : isCurrent 
                        ? 'bg-slate-100 text-slate-900 font-medium'
                        : 'text-slate-400'
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full shrink-0 ${
                    isDone ? 'bg-emerald-600' : isCurrent ? 'bg-slate-900 animate-ping' : 'bg-slate-300'
                  }`} />
                  <span className="truncate">{stage.label}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Error Banner if Failed */}
        {isFailed && (
          <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs font-mono flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <strong className="block font-semibold">Pipeline Execution Terminated</strong>
              <span>{statusData.error || 'An internal error occurred during video feature extraction.'}</span>
            </div>
          </div>
        )}

        {/* Footer controls */}
        <div className="flex items-center justify-between border-t border-slate-100 pt-4 text-xs font-mono text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600"></span>
            <span>Zero Data Leakage Sandboxed Worker</span>
          </span>
          <button
            type="button"
            onClick={onCancel}
            className="text-slate-500 hover:text-slate-800 underline transition-colors"
          >
            Cancel Run
          </button>
        </div>

      </div>

    </div>
  );
}
