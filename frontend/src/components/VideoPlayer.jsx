import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX, SkipBack, SkipForward } from 'lucide-react';

export default function VideoPlayer({ 
  videoUrl, 
  seekTime, 
  onTimeUpdate, 
  duration: initialDuration = 0,
  suspiciousFrames = []
}) {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(initialDuration || 0);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

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
      setDuration(videoRef.current.duration || initialDuration || 0);
    }
  };

  const handleSpeedChange = (speed) => {
    setPlaybackSpeed(speed);
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
  };

  const formatSeconds = (sec) => {
    if (isNaN(sec)) return '00:00.00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  const safeDuration = Math.max(duration || 1, 1);
  const currentPct = Math.min(100, Math.max(0, (currentTime / safeDuration) * 100));

  const handleScrubberClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, clickX / rect.width));
    const newTime = pct * safeDuration;
    if (videoRef.current) {
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
      if (onTimeUpdate) onTimeUpdate(newTime);
    }
  };

  // Find active anomaly frame near current time
  const currentFrame = suspiciousFrames.find(f => Math.abs(f.timestamp_seconds - currentTime) < 1.0);

  return (
    <div className="flex flex-col gap-3">
      {/* Video Viewport Surface */}
      <div className="relative w-full aspect-[16/10] bg-slate-100 rounded-xl overflow-hidden border border-slate-200/90 flex items-center justify-center shadow-sm select-none group">
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full h-full object-contain bg-black"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
          playsInline
        />

        {/* Subtle Scanning HUD Overlays inside video */}
        <div className="absolute inset-0 pointer-events-none p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-white font-mono text-[11px] drop-shadow-sm">
            <div className="flex items-center gap-1.5 bg-slate-900/80 backdrop-blur-sm px-2.5 py-1 rounded">
              <span className={`w-2 h-2 rounded-full ${currentFrame ? 'bg-rose-500 animate-pulse' : 'bg-emerald-500'}`}></span>
              <span className="tracking-wider">
                {currentFrame ? `FRAME #${String(currentFrame.frame_number).padStart(4, '0')}` : 'DIAGNOSTIC VIEW'}
              </span>
            </div>
            {currentFrame && (
              <div className="flex items-center gap-1.5 bg-slate-900/80 backdrop-blur-sm px-2.5 py-1 rounded">
                <span className="text-white/70">DELTA:</span>
                <span className="text-rose-400 font-semibold">{Math.round((currentFrame.score || 0.8) * 100)}% ANOMALY</span>
              </div>
            )}
          </div>

          {/* Forensic Dynamic Reticle Box if frame anomaly detected */}
          {currentFrame && (
            <div className="absolute top-[22%] left-[34%] w-[32%] h-[48%] border border-rose-500/90 bg-rose-500/10 rounded-sm pointer-events-none transition-all duration-200">
              <span className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-rose-500"></span>
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-rose-500"></span>
              <span className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-rose-500"></span>
              <span className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-rose-500"></span>
              <div className="absolute -top-5 left-0 px-1.5 py-0.5 bg-rose-600 text-white font-mono text-[10px] tracking-tight rounded-t-sm flex items-center gap-1">
                <span>{currentFrame.reason || 'ANOMALY DETECTED'}</span>
              </div>
            </div>
          )}

          {/* Bottom Left On-Screen Time Badge inside frame */}
          <div className="flex items-center gap-1.5 self-start bg-slate-900/80 backdrop-blur-sm px-2.5 py-1 rounded text-white font-mono text-[11px]">
            <span>{formatSeconds(currentTime)} / {formatSeconds(safeDuration)}</span>
          </div>
        </div>
      </div>

      {/* Scrubber Track Component */}
      <div className="flex flex-col gap-1.5 px-0.5">
        <div 
          onClick={handleScrubberClick}
          className="relative w-full h-6 flex items-center cursor-pointer group"
          title="Click to seek"
        >
          {/* Base track */}
          <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden relative">
            <div 
              className="h-full bg-slate-900 rounded-full transition-all duration-100" 
              style={{ width: `${currentPct}%` }}
            />
          </div>

          {/* Anomaly markers along track */}
          {suspiciousFrames.map((frame, i) => {
            const markerPct = Math.min(100, Math.max(0, (frame.timestamp_seconds / safeDuration) * 100));
            const isNear = Math.abs(markerPct - currentPct) < 2;
            return (
              <button
                key={i}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  if (videoRef.current) {
                    videoRef.current.currentTime = frame.timestamp_seconds;
                    setCurrentTime(frame.timestamp_seconds);
                    if (onTimeUpdate) onTimeUpdate(frame.timestamp_seconds);
                  }
                }}
                className={`absolute top-1/2 -translate-y-1/2 w-3 h-3 -ml-1.5 rounded-full flex items-center justify-center focus:outline-none transition-transform ${
                  isNear 
                    ? 'scale-125 bg-rose-600 border-2 border-white ring-2 ring-rose-400/50' 
                    : 'bg-rose-50 border border-rose-500 hover:scale-125'
                }`}
                style={{ left: `${markerPct}%` }}
                title={`${formatSeconds(frame.timestamp_seconds)}: ${frame.reason}`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-rose-600"></span>
              </button>
            );
          })}
        </div>

        {/* Playback Controls Row */}
        <div className="flex items-center justify-between pt-0.5">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={togglePlay}
              className="w-8 h-8 rounded-lg bg-white border border-slate-300 flex items-center justify-center text-slate-800 hover:bg-slate-50 transition-colors shadow-sm focus:outline-none"
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>

            <button
              type="button"
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.currentTime = Math.max(0, currentTime - 2);
                }
              }}
              className="w-7 h-7 rounded border border-slate-200 bg-white flex items-center justify-center text-slate-600 hover:text-slate-900 hover:bg-slate-50 transition-colors"
              title="Step back 2s"
            >
              <SkipBack className="w-3.5 h-3.5" />
            </button>

            <button
              type="button"
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.currentTime = Math.min(safeDuration, currentTime + 2);
                }
              }}
              className="w-7 h-7 rounded border border-slate-200 bg-white flex items-center justify-center text-slate-600 hover:text-slate-900 hover:bg-slate-50 transition-colors"
              title="Step forward 2s"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </button>

            <span className="font-mono text-xs text-slate-800 ml-1 font-medium">
              {formatSeconds(currentTime)} / {formatSeconds(safeDuration)}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-slate-600">
              <button 
                type="button"
                onClick={toggleMute} 
                className="hover:text-slate-900 transition-colors flex items-center" 
                title={isMuted ? 'Unmute' : 'Mute'}
              >
                {isMuted ? <VolumeX className="w-4 h-4 text-rose-600" /> : <Volume2 className="w-4 h-4" />}
              </button>
            </div>

            {/* Speed Toggle Pill Group */}
            <div className="flex items-center p-0.5 bg-slate-100 rounded border border-slate-200">
              {[0.25, 0.5, 1.0].map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => handleSpeedChange(s)}
                  className={`px-2 py-0.5 rounded font-mono text-[11px] transition-colors ${
                    playbackSpeed === s
                      ? 'bg-white font-semibold text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Analytical Legend Banner */}
      <div className="px-3.5 py-2 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-slate-600 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="font-medium text-slate-800">Spectral forensic lens:</span>
          <span>EHR Multi-scale Artifact Detector v3.4</span>
        </div>
        <span className="font-mono text-[11px] text-slate-500">
          Unaltered Frame Buffer
        </span>
      </div>
    </div>
  );
}
