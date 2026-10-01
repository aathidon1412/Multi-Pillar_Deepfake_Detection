import React, { useState } from 'react';
import { Mic, UploadCloud, Play, AlertCircle, ShieldCheck, Cpu, ArrowRight } from 'lucide-react';
import { analyzeAudio, getMediaUrl } from '../services/api';
import Pillar3XaiAudioExplanation from '../components/Pillar3XaiAudioExplanation';

export default function Pillar3Page() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioMode, setAudioMode] = useState('spoken');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [localAudioUrl, setLocalAudioUrl] = useState(null);

  const handleFileChange = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setLocalAudioUrl(url);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const data = await analyzeAudio(selectedFile, audioMode);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Audio analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 py-8 px-4 sm:px-6">
      
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs font-mono text-slate-700">
          <Mic className="w-3.5 h-3.5 text-slate-800" />
          <span className="uppercase tracking-wider font-semibold">PILLAR 3 • ACOUSTIC & SPEECH DEEPFAKE FORENSICS</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900 tracking-tight">
          Audio & Speech Authenticity Detection
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 max-w-xl mx-auto">
          Detects AI voice cloning, speech synthesis (TTS), and audio manipulation with Wav2Vec2 Integrated Gradients time-frequency saliency.
        </p>
      </div>

      {/* Main Container */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-5">
        
        {/* Mode selector */}
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="text-slate-500">Analysis Mode:</span>
          <button
            type="button"
            onClick={() => setAudioMode('spoken')}
            className={`px-3 py-1.5 rounded-lg border transition-all ${
              audioMode === 'spoken'
                ? 'bg-slate-900 text-white border-slate-900 shadow-sm'
                : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
          >
            Spoken Voice / Speech
          </button>
          <button
            type="button"
            onClick={() => setAudioMode('music')}
            className={`px-3 py-1.5 rounded-lg border transition-all ${
              audioMode === 'music'
                ? 'bg-slate-900 text-white border-slate-900 shadow-sm'
                : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
          >
            Music / Vocal Separation
          </button>
        </div>

        {/* Upload Zone */}
        {!selectedFile ? (
          <label className="border-2 border-dashed border-slate-300 hover:border-slate-400 hover:bg-slate-50 rounded-xl p-8 text-center flex flex-col items-center justify-center cursor-pointer transition-all">
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
            />
            <div className="w-12 h-12 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-700 mb-3">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-slate-900">
              Select or Drop Target Audio File
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports WAV, MP3, FLAC, M4A, OGG up to 50MB
            </p>
          </label>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="flex items-center gap-2.5 truncate">
                <Mic className="w-5 h-5 text-slate-700 shrink-0" />
                <span className="text-xs font-semibold text-slate-900 truncate">{selectedFile.name}</span>
                <span className="text-[11px] font-mono text-slate-500">({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)</span>
              </div>
              <button
                type="button"
                onClick={() => { setSelectedFile(null); setResult(null); setLocalAudioUrl(null); }}
                className="text-xs text-slate-500 hover:text-rose-600 underline font-mono"
              >
                Change
              </button>
            </div>

            <button
              type="button"
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full py-2.5 px-4 rounded-lg bg-slate-900 hover:bg-slate-800 disabled:opacity-75 text-white text-xs font-medium flex items-center justify-center gap-2 shadow-sm transition-all"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Computing Integrated Gradients & Saliency Spectrogram...</span>
                </>
              ) : (
                <>
                  <span>Run Acoustic Inspection & Saliency Attribution</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs font-mono">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-6 pt-6 border-t border-slate-100 space-y-6">
            <div className="flex items-center justify-between p-3.5 rounded-lg bg-slate-50 border border-slate-200">
              <div>
                <span className="text-[11px] font-mono uppercase text-slate-500 block">Dossier Finding</span>
                <h3 className="text-base font-semibold text-slate-900 mt-0.5">
                  {result.prediction === 'FAKE' || result.is_fake ? 'AI Synthesized Voice Detected' : 'Authentic Human Voice'}
                </h3>
              </div>
              <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-mono text-xs font-semibold ${
                result.prediction === 'FAKE' || result.is_fake ? 'bg-rose-50 text-rose-700 border border-rose-200' : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${result.prediction === 'FAKE' || result.is_fake ? 'bg-rose-600' : 'bg-emerald-600'}`} />
                {Math.round((result.confidence || 0.88) > 1.0 ? result.confidence : result.confidence * 100)}% Confidence
              </span>
            </div>

            {/* PILLAR 3 — WHY THIS AUDIO PREDICTION? (Integrated Gradients Saliency Component) */}
            <Pillar3XaiAudioExplanation
              xaiData={result.xai || result.pillar3?.xai}
              audioUrl={localAudioUrl || (result.file?.stored_filename ? getMediaUrl(`uploads/${result.file.stored_filename}`) : null)}
              prediction={result.prediction || result.pillar3?.prediction || 'REAL'}
              confidence={Number(result.confidence || result.pillar3?.confidence || 95.0)}
              audioMetadata={result.pillar3 || {}}
            />
          </div>
        )}

      </div>
    </div>
  );
}

