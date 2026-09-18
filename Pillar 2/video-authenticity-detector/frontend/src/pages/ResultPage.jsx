import React, { useState, useEffect } from 'react';
import { Download, RefreshCw, AlertTriangle, FileText, CheckCircle2, Video, Eye, Clock, ShieldAlert } from 'lucide-react';
import { getResult, getMediaUrl } from '../services/api';
import ProbabilityChart from '../components/ProbabilityChart';
import PillarBreakdown from '../components/PillarBreakdown';
import VideoPlayer from '../components/VideoPlayer';
import SuspiciousTimeline from '../components/SuspiciousTimeline';
import SuspiciousGallery from '../components/SuspiciousGallery';

export default function ResultPage({ videoId, onAnalyzeAnother }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [seekTime, setSeekTime] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!videoId) return;

    const fetchReport = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getResult(videoId);
        setReport(data);
      } catch (err) {
        console.error('Failed to load report:', err);
        setError('Failed to fetch forensic report. The analysis may still be compiling or file is missing.');
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [videoId]);

  const handleDownloadJson = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${report.video_id}_authenticity_report.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleSeek = (timeSec) => {
    setSeekTime(timeSec);
  };

  if (loading) {
    return (
      <div className="py-24 text-center space-y-4">
        <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin mx-auto" />
        <h3 className="text-xl font-bold text-white">Loading Forensic Report...</h3>
        <p className="text-sm font-mono text-slate-400">Retrieving JSON telemetry from storage layer</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-xl mx-auto py-16 text-center space-y-4">
        <div className="p-4 rounded-full bg-rose-950/60 text-rose-400 border border-rose-800 w-16 h-16 mx-auto flex items-center justify-center">
          <AlertTriangle className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-white">Report Not Found</h3>
        <p className="text-sm font-mono text-slate-400">{error}</p>
        <button
          onClick={onAnalyzeAnother}
          className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium transition-colors"
        >
          Analyze A Video
        </button>
      </div>
    );
  }

  const { file, video_details, metadata, analysis, classification, suspicious_frames, processing } = report;
  const videoUrl = getMediaUrl(`uploads/${file?.stored_filename}`);

  return (
    <div className="max-w-7xl mx-auto space-y-8 py-6">
      
      {/* Top Action & Metadata Header */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
            <span>PILLAR 2 REPORT</span>
            <span>•</span>
            <span className="text-slate-400">ID: {report.video_id}</span>
            <span>•</span>
            <span className="text-slate-400">{processing?.processed_at?.replace('T', ' ').slice(0, 19)}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white truncate max-w-xl">
            {file?.original_filename || 'Video Forensic Report'}
          </h1>
          <div className="flex items-center flex-wrap gap-4 text-xs font-mono text-slate-400 mt-2">
            <span>Duration: <strong className="text-slate-200">{video_details?.duration_seconds}s</strong></span>
            <span>Resolution: <strong className="text-slate-200">{video_details?.resolution}</strong></span>
            <span>FPS: <strong className="text-slate-200">{video_details?.fps}</strong></span>
            <span>Frames: <strong className="text-slate-200">{video_details?.frame_count}</strong></span>
            <span>Size: <strong className="text-slate-200">{file?.size_mb} MB</strong></span>
          </div>
        </div>

        <div className="flex items-center space-x-3 shrink-0">
          <button
            onClick={handleDownloadJson}
            className="px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 hover:border-cyan-500/40 text-xs font-mono transition-all flex items-center space-x-2"
            title="Download full raw JSON verdict"
          >
            <Download className="w-4 h-4 text-cyan-400" />
            <span>Download JSON</span>
          </button>

          <button
            onClick={onAnalyzeAnother}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-xs font-bold transition-all shadow-glow-cyan flex items-center space-x-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Analyze Another</span>
          </button>
        </div>
      </div>

      {/* Primary Split: Verdict Hero + Synced Video Player */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Verdict Hero & Probability Breakdown */}
        <div className="lg:col-span-5 flex flex-col justify-between">
          <ProbabilityChart classification={classification} />
        </div>

        {/* Right Column: Evidence Video Player */}
        <div className="lg:col-span-7 flex flex-col justify-between">
          <VideoPlayer
            videoUrl={videoUrl}
            seekTime={seekTime}
            onTimeUpdate={(t) => setCurrentTime(t)}
          />
        </div>

      </div>

      {/* Interactive Suspicious Timeline */}
      <SuspiciousTimeline
        duration={video_details?.duration_seconds || 30}
        currentTime={currentTime}
        suspiciousFrames={suspicious_frames || []}
        onSelectTimestamp={(t) => handleSeek(t)}
      />

      {/* 5-Pillar Evidence Grid */}
      <PillarBreakdown
        analysis={analysis || {}}
        metadata={metadata || {}}
      />

      {/* Suspicious Keyframe Gallery */}
      <SuspiciousGallery
        suspiciousFrames={suspicious_frames || []}
        onSelectFrame={(f) => handleSeek(f.timestamp_seconds)}
      />

    </div>
  );
}
