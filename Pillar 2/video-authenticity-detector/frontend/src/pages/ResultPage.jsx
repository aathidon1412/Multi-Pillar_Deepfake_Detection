import React, { useState, useEffect } from 'react';
import { ArrowLeft, Download, RefreshCw, AlertTriangle, FileText } from 'lucide-react';
import { getResult, getMediaUrl } from '../services/api';
import VideoPlayer from '../components/VideoPlayer';
import UnifiedFindingsPanel from '../components/UnifiedFindingsPanel';

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

  const handleDownloadPdf = () => {
    window.print();
  };

  const handleSeek = (timeSec) => {
    setSeekTime(timeSec);
  };

  if (loading) {
    return (
      <div className="py-24 text-center space-y-4">
        <div className="w-10 h-10 border-2 border-slate-300 border-t-slate-900 rounded-full animate-spin mx-auto" />
        <h3 className="text-lg font-semibold text-slate-900">Loading Forensic Dossier...</h3>
        <p className="text-xs font-mono text-slate-500">Retrieving evidentiary telemetry from storage layer</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-md mx-auto py-16 text-center space-y-4">
        <div className="p-3 rounded-full bg-rose-50 text-rose-600 border border-rose-200 w-12 h-12 mx-auto flex items-center justify-center">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-semibold text-slate-900">Report Not Found</h3>
        <p className="text-xs text-slate-500">{error}</p>
        <button
          onClick={onAnalyzeAnother}
          className="px-5 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium transition-colors"
        >
          Analyze Another Video
        </button>
      </div>
    );
  }

  const { file, video_details, metadata, analysis, classification, suspicious_frames, processing } = report;
  const videoUrl = getMediaUrl(`uploads/${file?.stored_filename}`);
  const isManipulated = classification?.prediction === 'AI_GENERATED' || classification?.prediction === 'FORGED';
  const confidencePct = Math.round((classification?.confidence || 0.85) * 100);

  return (
    <div className="w-full max-w-[1520px] mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6">
      
      {/* Top Action & Metadata Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div className="flex flex-col gap-1.5">
          {/* Breadcrumb / Back trigger */}
          <div className="flex items-center gap-2">
            <button
              onClick={onAnalyzeAnother}
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 transition-colors font-medium"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Intake</span>
            </button>
            <span className="text-slate-300">/</span>
            <span className="text-xs font-mono text-slate-600 uppercase tracking-wider font-semibold">
              USMFE Forensic Video Report
            </span>
          </div>

          <div className="flex items-center gap-3 flex-wrap mt-1">
            <h1 className="text-xl sm:text-2xl font-semibold text-slate-900 tracking-tight">
              {file?.original_filename || `Evidence_${report.video_id}.mp4`}
            </h1>

            {/* Verdict Chip */}
            <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-mono text-xs font-semibold tracking-wide ${
              isManipulated
                ? 'bg-rose-50 border border-rose-200 text-rose-700'
                : 'bg-emerald-50 border border-emerald-200 text-emerald-700'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${isManipulated ? 'bg-rose-600 animate-pulse' : 'bg-emerald-600'}`} />
              {confidencePct}% CONFIDENCE
            </span>

            {/* SHA-256 Badge */}
            <span className="px-2 py-0.5 rounded bg-slate-100 font-mono text-[11px] text-slate-600 border border-slate-200">
              SHA-256: {report.video_id?.slice(0, 8)}...{report.video_id?.slice(-4)}
            </span>
          </div>

          <div className="text-xs text-slate-500 flex items-center flex-wrap gap-2">
            <span>Analyzed: {processing?.processed_at?.replace('T', ' ').slice(0, 16) || 'Current Session'}</span>
            <span className="text-slate-300">•</span>
            <span>Duration: <strong className="text-slate-700 font-mono">{video_details?.duration_seconds}s</strong></span>
            <span className="text-slate-300">•</span>
            <span>Resolution: <strong className="text-slate-700 font-mono">{video_details?.resolution || '1080p'}</strong></span>
            <span className="text-slate-300">•</span>
            <span>FPS: <strong className="text-slate-700 font-mono">{video_details?.fps || 30}</strong></span>
            <span className="text-slate-300">•</span>
            <span>Size: <strong className="text-slate-700 font-mono">{file?.size_mb || '—'} MB</strong></span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 self-start md:self-auto shrink-0">
          <button
            type="button"
            onClick={handleDownloadJson}
            className="h-8 px-3 rounded-lg border border-slate-200 bg-white font-mono text-xs text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-1.5 focus:outline-none shadow-sm"
            title="Export raw JSON dossier"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Export JSON</span>
          </button>

          <button
            type="button"
            onClick={handleDownloadPdf}
            className="h-8 px-3.5 rounded-lg bg-slate-900 text-white font-medium text-xs hover:bg-slate-800 transition-colors flex items-center gap-1.5 focus:outline-none shadow-sm"
            title="Print or Save PDF report"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Download PDF</span>
          </button>

          <button
            type="button"
            onClick={onAnalyzeAnother}
            className="h-8 px-3 rounded-lg border border-slate-200 bg-white text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-1.5 shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
            <span>New Video</span>
          </button>
        </div>
      </div>

      {/* Main Diagnostic Split Canvas: Left Player (60%), Right Findings (40%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* LEFT COLUMN: Canvas, Video Player, Scrubber, Playback Controls (7 cols = ~58-60%) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <VideoPlayer
            videoUrl={videoUrl}
            seekTime={seekTime}
            duration={video_details?.duration_seconds || 30}
            suspiciousFrames={suspicious_frames || []}
            onTimeUpdate={(t) => setCurrentTime(t)}
          />
        </div>

        {/* RIGHT COLUMN: The Findings (Unified 3-Section Card - 40%) */}
        <div className="lg:col-span-5 flex flex-col">
          <UnifiedFindingsPanel
            report={report}
            seekTime={seekTime}
            onSeek={handleSeek}
            onAnalyzeAnother={onAnalyzeAnother}
            onDownloadJson={handleDownloadJson}
          />
        </div>

      </div>

    </div>
  );
}
