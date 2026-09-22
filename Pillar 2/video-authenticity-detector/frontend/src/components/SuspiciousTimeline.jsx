import React from 'react';
import { AlertCircle, Clock } from 'lucide-react';

export default function SuspiciousTimeline({
  duration = 30,
  currentTime = 0,
  suspiciousFrames = [],
  onSelectTimestamp
}) {
  const safeDuration = Math.max(duration || 1, 1);

  const formatTimestamp = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  return (
    <div className="glass-panel p-5 rounded-xl border border-slate-800">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-white tracking-wide uppercase font-mono">
            Suspicious Timeline & Anomaly Markers
          </h3>
        </div>
        <span className="text-xs font-mono text-slate-400">
          {suspiciousFrames.length} Flagged Timestamps
        </span>
      </div>

      {/* Timeline Bar Track */}
      <div className="relative h-10 w-full bg-slate-950/80 rounded-lg border border-slate-800 flex items-center px-4 overflow-visible select-none">
        
        {/* Base Track Line */}
        <div className="absolute left-4 right-4 h-1.5 bg-slate-800 rounded-full overflow-hidden">
          {/* Progress fill */}
          <div
            className="h-full bg-cyan-500/50 transition-all duration-100"
            style={{ width: `${Math.min(100, (currentTime / safeDuration) * 100)}%` }}
          />
        </div>

        {/* Current Playhead Scrubber */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-cyan-400 border-2 border-white shadow-glow-cyan z-20 pointer-events-none transition-all duration-100"
          style={{
            left: `calc(1rem + ${(Math.min(safeDuration, currentTime) / safeDuration)} * (100% - 2rem) - 7px)`
          }}
        />

        {/* Suspicious Marker Pins */}
        {suspiciousFrames.map((frame, index) => {
          const ratio = Math.min(1, Math.max(0, frame.timestamp_seconds / safeDuration));
          const leftPercent = ratio * 100;
          const isHighAnomaly = (frame.score || 0) >= 0.70;

          return (
            <button
              key={index}
              onClick={() => onSelectTimestamp(frame.timestamp_seconds, frame)}
              className="absolute top-1/2 -translate-y-1/2 group z-30 focus:outline-none"
              style={{
                left: `calc(1rem + ${ratio} * (100% - 2rem) - 8px)`
              }}
              title={`Jump to ${formatTimestamp(frame.timestamp_seconds)}: ${frame.reason}`}
            >
              {/* Outer pulsing ring */}
              <span className={`absolute -inset-1 rounded-full animate-ping opacity-75 ${
                isHighAnomaly ? 'bg-rose-500' : 'bg-amber-500'
              }`} />

              {/* Pin Core */}
              <span className={`relative block w-4 h-4 rounded-full border-2 border-slate-900 shadow-md transform group-hover:scale-125 transition-transform ${
                isHighAnomaly ? 'bg-rose-500 shadow-glow-rose' : 'bg-amber-400'
              }`} />

              {/* Tooltip on Hover */}
              <div className="absolute bottom-full mb-2.5 left-1/2 -translate-x-1/2 hidden group-hover:flex flex-col items-center pointer-events-none z-40 whitespace-nowrap">
                <div className="px-2.5 py-1 rounded bg-slate-900/95 border border-slate-700 shadow-xl text-[11px] font-mono text-slate-200">
                  <div className="text-cyan-400 font-bold">{formatTimestamp(frame.timestamp_seconds)}</div>
                  <div className="text-slate-300">{frame.reason}</div>
                  <div className="text-rose-400 font-semibold">Anomaly: {Math.round((frame.score || 0) * 100)}%</div>
                </div>
                <div className="w-2 h-2 bg-slate-900 border-r border-b border-slate-700 rotate-45 -mt-1" />
              </div>
            </button>
          );
        })}
      </div>

      {/* Timestamp bounds & legend */}
      <div className="flex items-center justify-between text-xs font-mono text-slate-500 mt-2 px-1">
        <span>00:00.00</span>
        <div className="flex items-center space-x-4">
          <span className="flex items-center space-x-1 text-amber-400/90 text-[11px]">
            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
            <span>Suspicious Artifact</span>
          </span>
          <span className="flex items-center space-x-1 text-rose-400/90 text-[11px]">
            <span className="w-2 h-2 rounded-full bg-rose-500"></span>
            <span>Severe Anomaly</span>
          </span>
        </div>
        <span>{formatTimestamp(safeDuration)}</span>
      </div>

      {/* Quick Jump Buttons */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap gap-2">
        <span className="text-xs text-slate-400 self-center mr-1 font-mono">Jump to:</span>
        {suspiciousFrames.map((frame, index) => (
          <button
            key={index}
            onClick={() => onSelectTimestamp(frame.timestamp_seconds, frame)}
            className="px-2.5 py-1 rounded-md bg-slate-900 hover:bg-slate-800 border border-slate-700/80 hover:border-cyan-500/50 text-xs font-mono text-cyan-300 transition-all flex items-center space-x-1.5"
          >
            <AlertCircle className="w-3 h-3 text-amber-400" />
            <span>{formatTimestamp(frame.timestamp_seconds)}</span>
            <span className="text-slate-500 text-[10px]">({Math.round((frame.score || 0) * 100)}%)</span>
          </button>
        ))}
      </div>

    </div>
  );
}
