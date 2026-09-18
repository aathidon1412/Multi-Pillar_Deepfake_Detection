import React, { useState } from 'react';
import { ZoomIn, X, AlertTriangle, Image as ImageIcon } from 'lucide-react';
import { getMediaUrl } from '../services/api';

export default function SuspiciousGallery({ suspiciousFrames = [], onSelectFrame }) {
  const [selectedImage, setSelectedImage] = useState(null);

  const formatTimestamp = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  if (!suspiciousFrames || suspiciousFrames.length === 0) {
    return (
      <div className="glass-panel p-6 rounded-xl border border-slate-800 text-center">
        <ImageIcon className="w-8 h-8 text-slate-500 mx-auto mb-2" />
        <p className="text-slate-400 text-sm">No anomalous frames detected.</p>
      </div>
    );
  }

  return (
    <div className="glass-panel p-5 rounded-xl border border-slate-800">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-rose-400" />
          <h3 className="text-sm font-semibold text-white tracking-wide uppercase font-mono">
            Extracted Suspicious Frames Gallery
          </h3>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          {suspiciousFrames.length} Keyframes Flagged
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {suspiciousFrames.map((frame, index) => {
          const imgUrl = getMediaUrl(frame.image);
          const scorePercent = Math.round((frame.score || 0) * 100);
          const isHighAnomaly = scorePercent >= 70;

          return (
            <div
              key={index}
              className="group relative rounded-lg overflow-hidden bg-slate-950 border border-slate-800/90 hover:border-cyan-500/50 transition-all hover:shadow-glow-cyan flex flex-col cursor-pointer"
              onClick={() => {
                if (onSelectFrame) onSelectFrame(frame);
                setSelectedImage(frame);
              }}
            >
              {/* Image Container */}
              <div className="relative aspect-video bg-black overflow-hidden flex items-center justify-center">
                <img
                  src={imgUrl}
                  alt={`Suspicious frame ${frame.frame_number}`}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
                <div className="hidden absolute inset-0 items-center justify-center text-xs text-slate-500 font-mono">
                  Frame #{frame.frame_number}
                </div>

                {/* Zoom overlay on hover */}
                <div className="absolute inset-0 bg-cyan-950/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <div className="p-2 rounded-full bg-slate-900/90 text-cyan-300 border border-cyan-500/40">
                    <ZoomIn className="w-4 h-4" />
                  </div>
                </div>

                {/* Timestamp Pill */}
                <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-slate-900/80 backdrop-blur-md border border-slate-700 text-[10px] font-mono text-cyan-300">
                  {formatTimestamp(frame.timestamp_seconds)}
                </div>

                {/* Score Pill */}
                <div className={`absolute top-2 right-2 px-2 py-0.5 rounded backdrop-blur-md text-[10px] font-mono font-bold border ${
                  isHighAnomaly 
                    ? 'bg-rose-950/80 text-rose-300 border-rose-700' 
                    : 'bg-amber-950/80 text-amber-300 border-amber-700'
                }`}>
                  {scorePercent}%
                </div>
              </div>

              {/* Frame Metadata Footer */}
              <div className="p-3 bg-slate-900/80 border-t border-slate-800/80 flex-1 flex flex-col justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-200 line-clamp-1 mb-1">
                    {frame.reason || "Artifact detected"}
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span>Frame #{frame.frame_number}</span>
                    <span>{frame.timestamp_seconds}s</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* High-Resolution Zoom Modal */}
      {selectedImage && (
        <div 
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div 
            className="relative max-w-4xl w-full bg-slate-900 border border-slate-700 rounded-2xl overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-white text-base">
                  Frame #{selectedImage.frame_number} Forensic Inspection
                </h4>
                <p className="text-xs font-mono text-slate-400">
                  Timestamp: {formatTimestamp(selectedImage.timestamp_seconds)} ({selectedImage.timestamp_seconds}s)
                </p>
              </div>
              <button
                onClick={() => setSelectedImage(null)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Image */}
            <div className="relative bg-black max-h-[70vh] flex items-center justify-center overflow-auto p-2">
              <img
                src={getMediaUrl(selectedImage.image)}
                alt="Zoomed frame"
                className="max-h-[65vh] w-auto object-contain rounded"
              />
            </div>

            {/* Modal Footer */}
            <div className="p-4 bg-slate-950 border-t border-slate-800 flex items-center justify-between flex-wrap gap-2">
              <div>
                <span className="text-xs text-slate-400 uppercase font-mono block">Detected Forensic Anomaly:</span>
                <span className="text-sm font-semibold text-rose-400">{selectedImage.reason}</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono text-slate-400">Suspicion Score:</span>
                <span className="px-2.5 py-1 rounded bg-rose-950/80 border border-rose-800 text-rose-300 font-mono text-sm font-bold">
                  {Math.round((selectedImage.score || 0) * 100)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
