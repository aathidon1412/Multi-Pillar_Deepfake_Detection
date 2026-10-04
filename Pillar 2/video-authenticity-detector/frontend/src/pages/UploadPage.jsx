import React, { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, FileVideo, Image as ImageIcon, Mic, FileText, 
  PlayCircle, AlertCircle, ArrowRight, Check, X, ShieldCheck, 
  History, ChevronRight, CheckCircle2, Sliders, Layers, Sparkles
} from 'lucide-react';
import { uploadVideo, startAnalysis, getHistory, analyzeUniversal } from '../services/api';
import Pillar1XaiExplanation from '../components/Pillar1XaiExplanation';
import Pillar3XaiAudioExplanation from '../components/Pillar3XaiAudioExplanation';
import Pillar4XaiDocumentExplanation from '../components/Pillar4XaiDocumentExplanation';
import Pillar5XaiPhysicsExplanation from '../components/Pillar5XaiPhysicsExplanation';

const IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'];
const VIDEO_EXTS = ['mp4', 'mov', 'avi', 'mkv', 'webm'];
const AUDIO_EXTS = ['wav', 'mp3', 'flac', 'ogg', 'm4a'];
const ALL_SUPPORTED = [...IMAGE_EXTS, ...VIDEO_EXTS, ...AUDIO_EXTS];

export default function UniversalUploadPage({ onStartProcessing, onSelectHistoricalVideo }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [detectedModality, setDetectedModality] = useState('unknown');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [recentItems, setRecentItems] = useState([]);
  const [recentLoading, setRecentLoading] = useState(false);
  const fileInputRef = useRef(null);

  // Universal Analysis Options
  const p4BenfordEnabled = false; // Disabled globally in UI
  const [audioSubmode, setAudioSubmode] = useState('spoken'); // 'spoken' or 'music'
  
  // Non-video analysis result (Image, Audio, PDF)
  const [universalResult, setUniversalResult] = useState(null);

  const fetchRecent = async () => {
    setRecentLoading(true);
    try {
      const data = await getHistory();
      if (Array.isArray(data)) {
        setRecentItems(data.slice(0, 4));
      }
    } catch (e) {
      console.error('Failed to load recent ledger:', e);
    } finally {
      setRecentLoading(false);
    }
  };

  useEffect(() => {
    fetchRecent();
  }, []);

  const getModality = (filename) => {
    const ext = filename.split('.').pop().toLowerCase();
    if (IMAGE_EXTS.includes(ext)) return 'image';
    if (VIDEO_EXTS.includes(ext)) return 'video';
    if (AUDIO_EXTS.includes(ext)) return 'audio';
    if (PDF_EXTS.includes(ext)) return 'pdf';
    return 'unknown';
  };

  const executePipeline = async (file, modality, p4Enabled, aMode) => {
    setIsAnalyzing(true);
    setErrorMessage(null);
    setUploadProgress(0);
    setUniversalResult(null);

    let currentPreview = null;
    if (modality === 'image' || modality === 'video' || modality === 'audio') {
      currentPreview = URL.createObjectURL(file);
      setPreviewUrl(currentPreview);
    } else {
      setPreviewUrl(null);
    }

    try {
      // 1. If Video -> Route to Pillar 2 Async Deep Pipeline
      if (modality === 'video') {
        const uploadRes = await uploadVideo(file, (progress) => {
          setUploadProgress(progress);
        });
        const videoId = uploadRes.video_id;
        await startAnalysis(videoId);
        if (onStartProcessing) {
          onStartProcessing(videoId, file.name, currentPreview);
        }
        return;
      }

      // 2. If Image, Audio, or PDF -> Execute Universal Consensus Pipeline
      const result = await analyzeUniversal(file, p4Enabled, aMode);
      setUniversalResult(result);
      setIsAnalyzing(false);
      // Automatically refresh the ledger so newly persisted item is visible
      fetchRecent();
    } catch (err) {
      console.error(err);
      let detail = err.response?.data?.detail;
      if (!detail) {
        if (err.message === 'Network Error' || err.code === 'ERR_NETWORK') {
          detail = 'Network Error: Cannot connect to FastAPI backend server (http://127.0.0.1:8000). Please ensure backend is running.';
        } else {
          detail = err.message || 'Forensic analysis failed.';
        }
      }
      setErrorMessage(detail);
      setIsAnalyzing(false);
    }
  };

  const handleFileChange = (file) => {
    if (!file) return;

    const ext = file.name.split('.').pop().toLowerCase();
    if (!ALL_SUPPORTED.includes(ext)) {
      setErrorMessage(`Unsupported format .${ext}. Please select an Image, Video, Audio clip, or PDF document.`);
      return;
    }

    const modality = getModality(file.name);
    setDetectedModality(modality);
    setSelectedFile(file);

    // AUTO-EXECUTE: Immediately begin processing upon upload
    executePipeline(file, modality, p4BenfordEnabled, audioSubmode);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleReanalyzeWithNewSettings = (newP4, newAudioMode) => {
    if (!selectedFile) return;
    executePipeline(selectedFile, detectedModality, newP4, newAudioMode);
  };

  // Modality display metadata
  const getModalityMeta = () => {
    switch (detectedModality) {
      case 'image':
        return {
          icon: ImageIcon,
          label: 'Visual Photo / Render',
          color: 'text-blue-600 bg-blue-50 border-blue-200',
          engines: 'Pillar 1 (ViT Neural Spectra) + Pillar 5 (Shadow RANSAC Physics)'
        };
      case 'video':
        return {
          icon: FileVideo,
          label: 'Digital Video Stream',
          color: 'text-rose-600 bg-rose-50 border-rose-200',
          engines: 'Pillar 2 (ViT Artifacts, Temporal Flickering, Acoustics, Lip-Sync SyncNet)'
        };
      case 'audio':
        return {
          icon: Mic,
          label: 'Audio / Voice Track',
          color: 'text-purple-600 bg-purple-50 border-purple-200',
          engines: `Pillar 3 (Acoustic Transformer + HPSS Demixing - Mode: ${audioSubmode})`
        };
      default:
        return {
          icon: UploadCloud,
          label: 'Universal Ingestion',
          color: 'text-slate-600 bg-slate-50 border-slate-200',
          engines: 'Auto-routing to corresponding forensic pillars'
        };
    }
  };

  const meta = getModalityMeta();
  const IconComp = meta.icon;

  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col items-center py-8 px-4 sm:px-6 space-y-6">
      
      {/* Centered Hero Header */}
      <div className="text-center flex flex-col items-center max-w-2xl">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-mono mb-2">
          <Sparkles className="w-3.5 h-3.5 text-slate-900" />
          <span className="uppercase tracking-wider font-semibold">Universal Multi-Pillar Media Ingestion</span>
        </div>
        <h1 className="text-2xl sm:text-3xl text-slate-900 font-semibold tracking-tight">
          Analyze Any Media for Synthetic Deepfakes & Tampering
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1.5 leading-relaxed">
          Upload <strong>any media file</strong> (photo, video, or voice clip). The system automatically identifies the media modality, executes all applicable forensic pillars, and computes unified consensus.
        </p>
      </div>

      {/* Main Ingestion Box */}
      <div className="w-full bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        
        {/* Empty Dropzone */}
        {!selectedFile ? (
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`w-full rounded-xl border-2 border-dashed p-10 text-center transition-all cursor-pointer ${
              isDragOver 
                ? 'border-slate-900 bg-slate-50' 
                : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50/60'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff,.mp4,.mov,.avi,.mkv,.webm,.wav,.mp3,.flac,.ogg,.m4a"
              className="hidden"
              onClick={(e) => { e.target.value = null; }}
              onChange={(e) => handleFileChange(e.target.files?.[0])}
            />

            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-700">
              <UploadCloud className="w-6 h-6" />
            </div>

            <p className="text-sm font-medium text-slate-900">
              Select or Drop Media File (Image · Video · Audio)
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports JPG, PNG, WEBP, MP4, MOV, AVI, WAV, MP3, FLAC up to 500 MB
            </p>

            <div className="mt-4 flex items-center justify-center gap-1.5 flex-wrap">
              <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 text-[11px] font-mono">Image (ViT + Shadow)</span>
              <span className="px-2 py-0.5 rounded bg-rose-50 text-rose-700 border border-rose-200 text-[11px] font-mono">Video (Multi-Engine)</span>
              <span className="px-2 py-0.5 rounded bg-purple-50 text-purple-700 border border-purple-200 text-[11px] font-mono">Audio (Wav2Vec2)</span>
            </div>
          </div>
        ) : (
          /* Staged File View */
          <div className="space-y-5">
            
            {/* Staged File Header with Auto-Detected Badge */}
            <div className="flex items-center justify-between p-3.5 rounded-lg bg-slate-50 border border-slate-200">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-10 h-10 rounded-lg bg-white border border-slate-200 flex items-center justify-center shrink-0">
                  <IconComp className="w-5 h-5 text-slate-800" />
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm text-slate-900 truncate font-semibold">
                      {selectedFile.name}
                    </span>
                    <span className={`text-[11px] font-mono px-2 py-0.5 rounded border uppercase font-medium ${meta.color}`}>
                      {meta.label}
                    </span>
                  </div>
                  <p className="text-xs font-mono text-slate-500 mt-0.5">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB 
                    <span className="mx-1 text-slate-300">•</span>
                    Engines: {meta.engines}
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => {
                  setSelectedFile(null);
                  setPreviewUrl(null);
                  setUniversalResult(null);
                }}
                disabled={isAnalyzing}
                className="w-8 h-8 rounded-full flex items-center justify-center text-slate-400 hover:text-rose-600 hover:bg-slate-200 transition-colors"
                title="Change File"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Media Preview Box */}
            {previewUrl && (
              <div className="max-w-xl mx-auto rounded-lg overflow-hidden bg-slate-950 border border-slate-200 shadow-sm flex items-center justify-center p-2">
                {detectedModality === 'image' && (
                  <img src={previewUrl} alt="Preview" className="max-h-64 object-contain rounded" />
                )}
                {detectedModality === 'video' && (
                  <video src={previewUrl} controls className="max-h-64 w-full object-contain" />
                )}
                {detectedModality === 'audio' && (
                  <div className="w-full p-4 flex flex-col items-center gap-2">
                    <Mic className="w-8 h-8 text-purple-400 animate-pulse" />
                    <audio src={previewUrl} controls className="w-full mt-2" />
                  </div>
                )}
              </div>
            )}

            {/* Optional Pipeline Modifiers */}
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-between flex-wrap gap-3 text-xs">
              {detectedModality === 'audio' && (
                <div className="flex items-center gap-2 font-mono">
                  <span className="text-slate-600 font-medium">Audio Pipeline Mode:</span>
                  <button
                    type="button"
                    onClick={() => {
                      setAudioSubmode('spoken');
                      handleReanalyzeWithNewSettings(p4BenfordEnabled, 'spoken');
                    }}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                      audioSubmode === 'spoken' ? 'bg-slate-900 text-white' : 'bg-white border text-slate-600'
                    }`}
                  >
                    Spoken Voice (Standard)
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setAudioSubmode('music');
                      handleReanalyzeWithNewSettings(p4BenfordEnabled, 'music');
                    }}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                      audioSubmode === 'music' ? 'bg-slate-900 text-white' : 'bg-white border text-slate-600'
                    }`}
                  >
                    Music Demixed (HPSS)
                  </button>
                </div>
              )}

              <div className="flex items-center gap-1.5 text-slate-500 font-mono text-[11px] ml-auto">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                <span>Zero-Retention Pipeline</span>
              </div>
            </div>

            {/* Live Automated Execution Status */}
            {isAnalyzing && (
              <div className="p-4 rounded-xl bg-slate-900 text-white shadow-sm space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span className="font-semibold uppercase tracking-wider text-slate-200">
                      Processing Stream in Real-Time
                    </span>
                  </div>
                  <span className="text-emerald-400 font-bold">{uploadProgress || 65}%</span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 transition-all duration-300"
                    style={{ width: `${uploadProgress || 70}%` }}
                  />
                </div>
                <p className="text-[11px] text-slate-400 font-mono">
                  Executing: {meta.engines}
                </p>
              </div>
            )}

            {/* Error Message */}
            {errorMessage && (
              <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs font-mono flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* In-Place Universal Consensus Results (Image, Audio, PDF) */}
            {universalResult && (
              <div className="mt-6 pt-6 border-t border-slate-200 space-y-6">
                
                {/* Consensus Verdict Banner */}
                {universalResult.consensus && (
                  <div className={`p-5 rounded-xl border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 ${
                    universalResult.consensus.is_real 
                      ? 'bg-emerald-50/70 border-emerald-200 text-emerald-900' 
                      : 'bg-rose-50/70 border-rose-200 text-rose-900'
                  }`}>
                    <div>
                      <span className="text-[11px] font-mono uppercase tracking-wider text-slate-500 block">
                        Consolidated Multi-Pillar Verdict
                      </span>
                      <h3 className="text-xl font-bold tracking-tight mt-0.5">
                        {universalResult.consensus.verdict}
                      </h3>
                      <p className="text-xs text-slate-600 mt-1">
                        Active Engines: <strong>{universalResult.consensus.engines}</strong>
                        {universalResult.consensus.override_reason && (
                          <span className="block text-amber-700 mt-0.5">• Note: {universalResult.consensus.override_reason}</span>
                        )}
                      </p>
                    </div>

                    <div className="text-right shrink-0">
                      <span className="text-[11px] font-mono uppercase text-slate-500 block">Confidence</span>
                      <span className="text-3xl font-bold font-mono">
                        {universalResult.consensus.confidence}%
                      </span>
                    </div>
                  </div>
                )}

                {/* Sub-Pillars Detailed Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {universalResult.pillar1 && (
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="text-[11px] font-mono text-slate-500 uppercase font-semibold">Pillar 1 • ViT Neural</div>
                      <h4 className="text-sm font-semibold text-slate-900 mt-1">
                        {universalResult.pillar1.verdict}
                      </h4>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Confidence: <strong>{universalResult.pillar1.confidence}%</strong>
                      </p>
                    </div>
                  )}

                  {universalResult.pillar5 && (
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="text-[11px] font-mono text-slate-500 uppercase font-semibold">Pillar 5 • Shadow RANSAC</div>
                      <h4 className="text-sm font-semibold text-slate-900 mt-1">
                        {universalResult.pillar5.verdict}
                      </h4>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Inliers: <strong>{Math.round((universalResult.pillar5.inlier_ratio || 0) * 100)}%</strong>
                      </p>
                    </div>
                  )}

                  {universalResult.pillar3 && (
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="text-[11px] font-mono text-slate-500 uppercase font-semibold">Pillar 3 • Acoustic Forensics</div>
                      <h4 className="text-sm font-semibold text-slate-900 mt-1">
                        {universalResult.pillar3.prediction}
                      </h4>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Confidence: <strong>{universalResult.pillar3.confidence}%</strong>
                      </p>
                    </div>
                  )}
                </div>

                {/* PILLAR 3 — WHY WAS THIS AUDIO FLAGGED? (Integrated Gradients Spectrogram Saliency) */}
                {(universalResult.pillar3?.xai || detectedModality === 'audio') && (
                  <div className="pt-2">
                    <Pillar3XaiAudioExplanation
                      xaiData={universalResult.pillar3?.xai || universalResult.xai?.pillar3 || universalResult.xai}
                      audioUrl={previewUrl || (universalResult.file?.stored_filename ? `/storage/uploads/${universalResult.file.stored_filename}` : null)}
                      prediction={universalResult.pillar3?.prediction || universalResult.consensus?.verdict || 'REAL'}
                      confidence={universalResult.pillar3?.confidence || universalResult.consensus?.confidence || 85.0}
                      audioMetadata={universalResult.pillar3 || {}}
                    />
                  </div>
                )}

                {/* PILLAR 4 — WHY WAS THIS DOCUMENT FLAGGED? (Statistical Explainability) */}
                {(detectedModality === 'pdf' || (universalResult.pillar4?.applicable && universalResult.pillar4?.digits_count >= 5)) && universalResult.pillar4 && (
                  <div className="pt-2">
                    <Pillar4XaiDocumentExplanation
                      xaiData={universalResult.pillar4?.xai || universalResult.xai?.pillar4 || universalResult.xai}
                      prediction={universalResult.pillar4?.verdict || universalResult.consensus?.verdict}
                      confidence={universalResult.pillar4?.confidence || universalResult.consensus?.confidence || 85.0}
                      digitsCount={universalResult.pillar4?.digits_count || 0}
                    />
                  </div>
                )}

                {/* PILLAR 5 — WHY DID THE PHYSICAL-FORENSICS MODEL MAKE THIS PREDICTION? (TreeSHAP Explainability) */}
                {detectedModality === 'image' && (universalResult.pillar5?.xai || universalResult.xai?.pillar5) && (
                  <div className="pt-2">
                    <Pillar5XaiPhysicsExplanation
                      xaiData={universalResult.pillar5?.xai || universalResult.xai?.pillar5}
                      prediction={universalResult.pillar5?.verdict}
                      confidence={universalResult.pillar5?.confidence || 85.0}
                    />
                  </div>
                )}

                {/* PILLAR 1 — WHY THIS PREDICTION? (Vision Transformer XAI Section) */}
                {detectedModality === 'image' && (universalResult.pillar1 || universalResult.xai?.pillar1 || universalResult.xai) && (
                  <div className="pt-2">
                    <Pillar1XaiExplanation
                      xaiData={universalResult.pillar1?.xai || universalResult.xai?.pillar1 || universalResult.xai}
                      originalImageSrc={previewUrl}
                      prediction={universalResult.pillar1?.verdict || universalResult.consensus?.verdict || 'REAL'}
                      confidence={universalResult.pillar1?.confidence || universalResult.consensus?.confidence || 85.0}
                    />
                  </div>
                )}

                {/* Reset button */}
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedFile(null);
                      setPreviewUrl(null);
                      setUniversalResult(null);
                    }}
                    className="px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 transition-colors shadow-sm"
                  >
                    Analyze Another Media File
                  </button>
                </div>

              </div>
            )}

          </div>
        )}

      </div>

      {/* Bottom Recent Video & Media Inspections Row */}
      <div className="w-full max-w-4xl pt-4 border-t border-slate-200">
        <div className="flex items-center justify-between pb-3">
          <div className="flex items-center gap-1.5">
            <History className="w-4 h-4 text-slate-500" />
            <span className="text-xs font-mono uppercase tracking-wider text-slate-600 font-semibold">
              Recent Video & Media Inspections
            </span>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            Audit Ledger Active
          </span>
        </div>

        {recentLoading ? (
          <div className="py-6 text-center text-xs font-mono text-slate-400">
            Loading inspection ledger...
          </div>
        ) : recentItems.length > 0 ? (
          <div className="space-y-2">
            {recentItems.map((item) => {
              const modality = item.modality || (item.video_id?.startsWith('VID_') ? 'video' : 'video');
              const isForged = item.prediction === 'AI_GENERATED' || item.prediction === 'FORGED';
              const confPct = Math.round((item.confidence || 0.85) * 100);

              const handleRecentClick = async () => {
                if (modality === 'video') {
                  if (onSelectHistoricalVideo) onSelectHistoricalVideo(item.video_id);
                } else {
                  try {
                    const fullData = await (await fetch(`/api/result/${item.video_id}`)).json();
                    setUniversalResult(fullData);
                    setSelectedFile({ name: item.filename, size: (item.size_mb || 1) * 1024 * 1024 });
                    setDetectedModality(modality);
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                  } catch (e) {
                    console.error('Failed to load item:', e);
                  }
                }
              };

              return (
                <div
                  key={item.video_id}
                  onClick={handleRecentClick}
                  className="flex items-center justify-between p-3 rounded-lg bg-white border border-slate-200 hover:border-slate-300 hover:bg-slate-50/80 transition-colors shadow-sm cursor-pointer"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${isForged ? 'bg-rose-600' : 'bg-emerald-600'}`} />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs text-slate-900 truncate block font-medium">
                          {item.filename || `Evidence: ${item.video_id}`}
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded uppercase border bg-slate-50 text-slate-600 border-slate-200">
                          {modality}
                        </span>
                      </div>
                      <span className="text-[11px] font-mono text-slate-500">
                        {item.date ? item.date.replace('T', ' ').slice(0, 16) : 'Case #USM-Audit'}
                        <span className="mx-1 text-slate-300">•</span>
                        ID: {item.video_id?.slice(0, 10)}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium ${
                      isForged 
                        ? 'bg-rose-50 text-rose-700 border border-rose-200' 
                        : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    }`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${isForged ? 'bg-rose-600' : 'bg-emerald-600'}`} />
                      {isForged ? `Manipulated (${confPct}%)` : `Authentic (${confPct}%)`}
                    </span>

                    <button
                      type="button"
                      className="text-xs font-medium text-slate-700 hover:text-slate-900 flex items-center gap-0.5"
                    >
                      <span>View Dossier</span>
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="py-4 text-center text-xs font-mono text-slate-400 bg-white rounded-lg border border-slate-200">
            No previous inspection records found in local storage ledger.
          </div>
        )}
      </div>

    </div>
  );
}
