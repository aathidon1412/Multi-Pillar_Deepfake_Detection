import React from 'react';
import { Eye, Activity, Mic, MessageSquare, FileCode, CheckCircle2, AlertTriangle, HelpCircle } from 'lucide-react';

export default function PillarBreakdown({ analysis = {}, metadata = {} }) {
  const visual = analysis.visual || {};
  const temporal = analysis.temporal || {};
  const audio = analysis.audio || {};
  const lipSync = analysis.lip_sync || {};
  const faceDetect = analysis.face_detection || {};

  const getStatusBadge = (status, isAvailable = true) => {
    if (!isAvailable || status === 'not_applicable') {
      return (
        <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-slate-800 text-slate-400 border border-slate-700">
          N/A
        </span>
      );
    }
    switch (status) {
      case 'suspicious':
        return (
          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-rose-950/80 text-rose-400 border border-rose-800 flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3" />
            <span>Suspicious</span>
          </span>
        );
      case 'slight_anomaly':
        return (
          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-amber-950/80 text-amber-400 border border-amber-800 flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3" />
            <span>Slight Anomaly</span>
          </span>
        );
      case 'normal':
        return (
          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-emerald-950/80 text-emerald-400 border border-emerald-800 flex items-center space-x-1">
            <CheckCircle2 className="w-3 h-3" />
            <span>Normal</span>
          </span>
        );
      case 'inconclusive':
      default:
        return (
          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-slate-800 text-slate-300 border border-slate-700 flex items-center space-x-1">
            <HelpCircle className="w-3 h-3" />
            <span>Inconclusive</span>
          </span>
        );
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-bold text-white font-mono uppercase tracking-wider flex items-center space-x-2">
          <span>Multi-Pillar Evidence Decomposition</span>
        </h3>
        <span className="text-xs text-slate-400 font-mono">5 Independent Forensic Channels</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        
        {/* Pillar 1: Visual Analysis */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-800 text-cyan-400">
                  <Eye className="w-4 h-4" />
                </div>
                <span className="text-sm font-semibold text-slate-100">Visual Artifacts</span>
              </div>
              {getStatusBadge(visual.status)}
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Facial boundaries, high-frequency texture, and GAN blend artifacts.
            </p>
            {visual.anomalies && visual.anomalies.length > 0 ? (
              <div className="space-y-1 mb-3">
                {visual.anomalies.map((anom, i) => (
                  <div key={i} className="text-[11px] font-mono text-rose-300 bg-rose-950/40 px-2 py-0.5 rounded border border-rose-900/50">
                    • {anom}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-[11px] font-mono text-emerald-400/90 mb-3">
                ✓ No overt seam or texture anomalies
              </div>
            )}
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Anomaly Score</span>
            <span className="font-bold text-slate-200">{Math.round((visual.score || 0) * 100)}%</span>
          </div>
        </div>

        {/* Pillar 2: Temporal Analysis */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-lg bg-blue-950 border border-blue-800 text-blue-400">
                  <Activity className="w-4 h-4" />
                </div>
                <span className="text-sm font-semibold text-slate-100">Temporal Stability</span>
              </div>
              {getStatusBadge(temporal.status)}
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Inter-frame flickering, optical motion flow, and identity consistency.
            </p>
            <div className="space-y-1.5 text-xs font-mono mb-3">
              <div className="flex justify-between">
                <span className="text-slate-400">Flickering:</span>
                <span className={temporal.flickering_detected ? 'text-rose-400 font-bold' : 'text-emerald-400'}>
                  {temporal.flickering_detected ? 'DETECTED' : 'CLEAR'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Motion Flow:</span>
                <span className={temporal.motion_inconsistency ? 'text-amber-400 font-bold' : 'text-emerald-400'}>
                  {temporal.motion_inconsistency ? 'INCONSISTENT' : 'CONTINUOUS'}
                </span>
              </div>
            </div>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Anomaly Score</span>
            <span className="font-bold text-slate-200">{Math.round((temporal.score || 0) * 100)}%</span>
          </div>
        </div>

        {/* Pillar 3: Audio Analysis */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-lg bg-purple-950 border border-purple-800 text-purple-400">
                  <Mic className="w-4 h-4" />
                </div>
                <span className="text-sm font-semibold text-slate-100">Audio & Voice</span>
              </div>
              {getStatusBadge(audio.status, audio.available)}
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Synthetic voice indicators, spectral rolloff, and acoustic integrity.
            </p>
            <div className="text-xs font-mono text-slate-400 space-y-1 mb-3">
              <div>Audio Stream: <span className="text-slate-200 font-semibold">{audio.available ? '16kHz WAV Available' : 'No Track Detected'}</span></div>
              {audio.available && (
                <div>HF Energy Ratio: <span className="text-cyan-400 font-semibold">{audio.hf_ratio || 'N/A'}</span></div>
              )}
            </div>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Anomaly Score</span>
            <span className="font-bold text-slate-200">{audio.available ? `${Math.round((audio.score || 0) * 100)}%` : 'N/A'}</span>
          </div>
        </div>

        {/* Pillar 4: Lip-Sync Analysis */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-lg bg-indigo-950 border border-indigo-800 text-indigo-400">
                  <MessageSquare className="w-4 h-4" />
                </div>
                <span className="text-sm font-semibold text-slate-100">Lip-Sync Coherence</span>
              </div>
              {getStatusBadge(lipSync.status, lipSync.available)}
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Correlation between mouth aspect ratio (MAR) and speech phoneme energy.
            </p>
            <div className="text-xs font-mono text-slate-400 space-y-1 mb-3">
              <div>Sync Status: <span className="text-slate-200 font-semibold">{lipSync.available ? (lipSync.status || 'Active') : 'Insufficient Face/Audio'}</span></div>
              {lipSync.correlation !== undefined && (
                <div>Speech-Mouth Corr: <span className="text-cyan-400 font-semibold">{lipSync.correlation}</span></div>
              )}
            </div>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Anomaly Score</span>
            <span className="font-bold text-slate-200">{lipSync.available ? `${Math.round((lipSync.score || 0) * 100)}%` : 'N/A'}</span>
          </div>
        </div>

        {/* Pillar 5: Metadata & Container */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col justify-between md:col-span-2 lg:col-span-2">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-lg bg-slate-900 border border-slate-700 text-slate-300">
                  <FileCode className="w-4 h-4" />
                </div>
                <span className="text-sm font-semibold text-slate-100">Forensic Metadata & Encoding</span>
              </div>
              {getStatusBadge(metadata.metadata_status)}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs font-mono mb-3">
              <div>
                <span className="text-slate-500 block">Codec:</span>
                <span className="text-slate-200 font-semibold truncate block">{metadata.codec || 'Unknown'}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Encoder:</span>
                <span className="text-slate-200 font-semibold truncate block">{metadata.encoder || 'Unknown'}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Faces Detected:</span>
                <span className="text-cyan-400 font-semibold">{faceDetect.faces_detected ? `${faceDetect.face_count} found` : 'None'}</span>
              </div>
            </div>
            {metadata.forensic_flags && metadata.forensic_flags.length > 0 && (
              <div className="text-[11px] font-mono text-amber-300 bg-amber-950/40 p-2 rounded border border-amber-900/50">
                ⚠️ Flags: {metadata.forensic_flags.join(', ')}
              </div>
            )}
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Supporting Evidence Weight</span>
            <span className="font-bold text-slate-200">10%</span>
          </div>
        </div>

      </div>
    </div>
  );
}
