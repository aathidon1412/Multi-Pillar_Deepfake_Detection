import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX } from 'lucide-react';

export default function VideoPlayer({ videoUrl, seekTime, onTimeUpdate }) {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);

  useEffect(() => {
    if (seekTime !== null && seekTime !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekTime;
      setCurrentTime(seekTime);
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  }, [seekTime]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    videoRef.current.muted = !videoRef.current.muted;
    setIsMuted(videoRef.current.muted);
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const cur = videoRef.current.currentTime;
    setCurrentTime(cur);
    if (onTimeUpdate) {
      onTimeUpdate(cur);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration || 0);
    }
  };

  const formatSeconds = (sec) => {
    if (isNaN(sec)) return '00:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  return (
    <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl flex flex-col">
      <div className="relative aspect-video flex items-center justify-center bg-black">
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full h-full object-contain"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
          playsInline
        />

        {/* Video Overlay Watermark / HUD */}
        <div className="absolute top-3 left-3 px-2 py-1 rounded bg-black/60 backdrop-blur-md border border-cyan-500/30 text-[11px] font-mono text-cyan-400 flex items-center space-x-1.5 pointer-events-none">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>
          <span>EVIDENCE PREVIEW</span>
        </div>

        <div className="absolute top-3 right-3 px-2.5 py-1 rounded bg-black/60 backdrop-blur-md border border-slate-700 text-xs font-mono text-slate-300 pointer-events-none">
          {formatSeconds(currentTime)} / {formatSeconds(duration)}
        </div>
      </div>

      {/* Embedded Controls */}
      <div className="p-3 bg-slate-900/90 border-t border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <button
            onClick={togglePlay}
            className="p-2 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 border border-cyan-500/40 transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </button>

          <button
            onClick={() => {
              if (videoRef.current) {
                videoRef.current.currentTime = 0;
                setCurrentTime(0);
              }
            }}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Restart"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={toggleMute}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4" />}
          </button>
        </div>

        <div className="text-xs font-mono text-slate-400">
          Scrubber sync active
        </div>
      </div>
    </div>
  );
}
