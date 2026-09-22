import React, { useEffect, useState } from 'react';
import { CheckCircle2, Circle, Loader2, AlertCircle, Cpu, Film, Eye, Activity, Mic, MessageSquare, Layers } from 'lucide-react';
import { checkStatus } from '../services/api';

export default function ProcessingPage({ videoId, filename, onProcessingComplete, onCancel }) {
  const [statusData, setStatusData] = useState({
    status: 'processing',
    progress: 5,
    current_stage: 'Initializing',
    error: null,
  });

  const pipelineStages = [
    { id: 'prep', label: 'Video Preprocessing', icon: Film, threshold: 10 },
    { id: 'meta', label: 'Metadata Extraction', icon: Layers, threshold: 20 },
    { id: 'frames', label: 'Frame Extraction', icon: Film, threshold: 30 },
    { id: 'face', label: 'Face Detection', icon: Eye, threshold: 45 },
    { id: 'visual', label: 'Visual Analysis', icon: Cpu, threshold: 60 },
    { id: 'temporal', label: 'Temporal Analysis', icon: Activity, threshold: 72 },
    { id: 'audio', label: 'Audio Analysis', icon: Mic, threshold: 80 },
    { id: 'lipsync', label: 'Lip-Sync Analysis', icon: MessageSquare, threshold: 86 },
    { id: 'class', label: 'Feature Fusion & Classification', icon: Layers, threshold: 92 },
    { id: 'report', label: 'Report & Suspicious Keyframes', icon: CheckCircle2, threshold: 98 },
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

  const progress = statusData.progress || 0;
  const currentStage = statusData.current_stage || 'Analyzing';
  const isFailed = statusData.status === 'failed';

  return (
    <div className="max-w-3xl mx-auto py-10 space-y-8">
      
      {/* Processing Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-800/50 text-xs font-mono text-cyan-400">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
          <span>VIDEO FORENSIC PIPELINE</span>
        </div>
        <h2 className="text-3xl font-black text-white tracking-tight">
          {isFailed ? 'Analysis Interrupted' : 'Analyzing Video Authenticity...'}
        </h2>
        <p className="text-sm text-slate-400 font-mono">
          File: <span className="text-slate-200">{filename || videoId}</span> • ID: <span className="text-cyan-400">{videoId}</span>
        </p>
      </div>

      {/* Progress Card */}
      <div className="glass-panel p-8 rounded-2xl border border-slate-800 shadow-2xl space-y-6">
        
        {/* Progress Bar & Numerical Gauge */}
        <div className="space-y-3">
          <div className="flex justify-between items-end">
            <div>
              <span className="text-xs uppercase font-mono text-slate-400 block mb-1">Current Active Stage</span>
              <span className="text-lg font-bold text-white font-mono flex items-center space-x-2">
                {!isFailed && <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />}
                <span>{currentStage}</span>
              </span>
            </div>
            <div className="text-right">
              <span className="text-4xl font-black font-mono text-cyan-400">
                {progress}%
              </span>
            </div>
          </div>

          {/* Glowing Progress Track */}
          <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800 p-0.5">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isFailed 
                  ? 'bg-rose-500 shadow-glow-rose' 
                  : 'bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500 shadow-glow-cyan'
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Failure Message */}
        {isFailed && (
          <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-300 text-xs font-mono space-y-2">
            <div className="flex items-center space-x-2 font-bold">
              <AlertCircle className="w-4 h-4 text-rose-400" />
              <span>Pipeline execution error:</span>
            </div>
            <p>{statusData.error || 'An unexpected error occurred during frame extraction or model inference.'}</p>
            <button
              onClick={onCancel}
              className="mt-2 px-4 py-2 rounded bg-rose-900/60 hover:bg-rose-800 text-rose-200 text-xs font-mono transition-colors"
            >
              Return to Upload
            </button>
          </div>
        )}

        {/* 10-Stage Pipeline Stepper */}
        <div className="pt-4 border-t border-slate-800/80 space-y-2">
          <h4 className="text-xs uppercase font-mono text-slate-400 mb-3 tracking-wider">
            Verification Pipeline Stages
          </h4>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {pipelineStages.map((stage, idx) => {
              const isCompleted = progress >= stage.threshold || statusData.status === 'completed';
              const isCurrent = !isCompleted && (idx === 0 || progress >= pipelineStages[idx - 1].threshold);
              const StageIcon = stage.icon;

              return (
                <div
                  key={stage.id}
                  className={`p-2.5 rounded-lg border flex items-center justify-between transition-all ${
                    isCompleted
                      ? 'bg-emerald-950/20 border-emerald-900/40 text-slate-300'
                      : isCurrent
                      ? 'bg-cyan-950/40 border-cyan-500/50 text-cyan-300 shadow-glow-cyan'
                      : 'bg-slate-900/30 border-slate-800/60 text-slate-600'
                  }`}
                >
                  <div className="flex items-center space-x-2.5">
                    <StageIcon className={`w-4 h-4 ${
                      isCompleted ? 'text-emerald-400' : isCurrent ? 'text-cyan-400' : 'text-slate-600'
                    }`} />
                    <span className="text-xs font-mono font-medium">{stage.label}</span>
                  </div>

                  {isCompleted ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  ) : isCurrent ? (
                    <Loader2 className="w-4 h-4 text-cyan-400 animate-spin shrink-0" />
                  ) : (
                    <Circle className="w-3.5 h-3.5 text-slate-700 shrink-0" />
                  )}
                </div>
              );
            })}
          </div>
        </div>

      </div>

    </div>
  );
}
