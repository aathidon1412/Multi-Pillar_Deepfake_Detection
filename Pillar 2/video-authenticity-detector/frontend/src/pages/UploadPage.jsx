import React, { useState, useRef } from 'react';
import { UploadCloud, FileVideo, PlayCircle, AlertCircle, ArrowRight, Check } from 'lucide-react';
import { uploadVideo, startAnalysis } from '../services/api';

export default function UploadPage({ onStartProcessing }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const supportedFormats = ['MP4', 'MOV', 'AVI', 'MKV', 'WEBM'];

  const handleFileChange = (file) => {
    if (!file) return;

    // Check extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (!supportedFormats.map(f => f.toLowerCase()).includes(ext)) {
      setErrorMessage(`Unsupported format .${ext}. Please select an MP4, MOV, AVI, MKV, or WEBM video.`);
      return;
    }

    setErrorMessage(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setErrorMessage(null);
    setUploadProgress(0);

    try {
      // 1. Upload video
      const uploadRes = await uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      const videoId = uploadRes.video_id;

      // 2. Trigger analysis
      await startAnalysis(videoId);

      // 3. Switch to processing screen
      if (onStartProcessing) {
        onStartProcessing(videoId, selectedFile.name, previewUrl);
      }
    } catch (err) {
      console.error(err);
      let detail = err.response?.data?.detail;
      if (!detail) {
        if (err.message === 'Network Error' || err.code === 'ERR_NETWORK') {
          detail = 'Network Error: Cannot connect to the FastAPI backend server (http://127.0.0.1:8000). Please ensure the backend server is running in a terminal.';
        } else {
          detail = err.message || 'Failed to upload and analyze video.';
        }
      }
      setErrorMessage(detail);
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 py-6">
      
      {/* Hero Section */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800/60 text-xs font-mono text-cyan-300">
          <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
          <span>USMFE PILLAR 2 FORENSIC ENGINE</span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-black text-white tracking-tight">
          Video Authenticity Detector
        </h1>
        <p className="text-base text-slate-400 max-w-2xl mx-auto">
          Analyze video integrity across visual deepfake artifacts, temporal flickering, audio acoustics, and lip-sync synchronization with complete explainable evidence.
        </p>
      </div>

      {/* Upload Box */}
      <div className="glass-panel p-8 rounded-2xl border border-slate-800 shadow-2xl relative overflow-hidden">
        
        {!selectedFile ? (
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
              isDragOver 
                ? 'border-cyan-400 bg-cyan-950/30 shadow-glow-cyan' 
                : 'border-slate-700 hover:border-cyan-500/50 hover:bg-slate-900/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.mov,.avi,.mkv,.webm"
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
            />

            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
              <UploadCloud className="w-8 h-8" />
            </div>

            <h3 className="text-lg font-bold text-white mb-1">
              Choose or Drag & Drop Video File
            </h3>
            <p className="text-sm text-slate-400 mb-6">
              Full forensic inspection with local zero-database JSON persistence
            </p>

            {/* Supported Formats Badges */}
            <div className="flex items-center justify-center flex-wrap gap-2 text-xs font-mono text-slate-400">
              <span className="text-slate-500">Supported:</span>
              {supportedFormats.map((fmt) => (
                <span key={fmt} className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  {fmt}
                </span>
              ))}
              <span className="text-slate-500">• Max 500MB</span>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            
            {/* File Info Bar */}
            <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center space-x-3">
                <div className="p-2.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                  <FileVideo className="w-6 h-6" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-white truncate max-w-md">{selectedFile.name}</h4>
                  <p className="text-xs font-mono text-slate-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || 'Video'}
                  </p>
                </div>
              </div>

              <button
                onClick={() => {
                  setSelectedFile(null);
                  setPreviewUrl(null);
                }}
                disabled={isUploading}
                className="text-xs font-mono text-slate-400 hover:text-rose-400 underline transition-colors disabled:opacity-50"
              >
                Choose Different Video
              </button>
            </div>

            {/* Video Preview */}
            <div className="relative aspect-video max-w-2xl mx-auto rounded-xl overflow-hidden bg-black border border-slate-800 shadow-xl">
              <video
                src={previewUrl}
                controls
                className="w-full h-full object-contain"
              />
              <div className="absolute top-3 left-3 px-2 py-0.5 rounded bg-black/70 backdrop-blur-md border border-cyan-500/40 text-[10px] font-mono text-cyan-400 flex items-center space-x-1.5">
                <PlayCircle className="w-3.5 h-3.5" />
                <span>PREVIEW READY</span>
              </div>
            </div>

            {/* Upload Progress Bar */}
            {isUploading && (
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-mono text-slate-300">
                  <span>Uploading video to forensic server...</span>
                  <span className="text-cyan-400 font-bold">{uploadProgress}%</span>
                </div>
                <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="h-full bg-cyan-500 transition-all duration-200"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Error banner */}
            {errorMessage && (
              <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-800/80 text-rose-300 text-xs font-mono flex items-start space-x-2">
                <AlertCircle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Analyze Button */}
            <div className="flex justify-center pt-2">
              <button
                onClick={handleStartAnalysis}
                disabled={isUploading}
                className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-base shadow-glow-cyan flex items-center justify-center space-x-3 transition-all transform hover:-translate-y-0.5 disabled:opacity-50 disabled:pointer-events-none"
              >
                {isUploading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Initiating Pipeline...</span>
                  </>
                ) : (
                  <>
                    <span>Start Deep Forensic Analysis</span>
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>

          </div>
        )}

      </div>

      {/* Feature Highlights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono text-slate-400">
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex items-start space-x-3">
          <div className="w-2 h-2 rounded-full bg-cyan-400 mt-1.5 shrink-0"></div>
          <div>
            <strong className="text-slate-200 block text-sm font-sans mb-0.5">ViT + Face Forensics</strong>
            Extracts face crops and evaluates seam blending, skin textures, and Laplacian sharpness.
          </div>
        </div>
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex items-start space-x-3">
          <div className="w-2 h-2 rounded-full bg-blue-400 mt-1.5 shrink-0"></div>
          <div>
            <strong className="text-slate-200 block text-sm font-sans mb-0.5">Temporal & Flickering</strong>
            Measures consecutive optical flow, sudden lighting transitions, and motion discontinuities.
          </div>
        </div>
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex items-start space-x-3">
          <div className="w-2 h-2 rounded-full bg-purple-400 mt-1.5 shrink-0"></div>
          <div>
            <strong className="text-slate-200 block text-sm font-sans mb-0.5">Audio & Lip-Sync</strong>
            Cross-correlates speech audio acoustic energy against mouth aspect ratio movements.
          </div>
        </div>
      </div>

    </div>
  );
}
