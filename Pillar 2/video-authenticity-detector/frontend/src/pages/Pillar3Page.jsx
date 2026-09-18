import React, { useState } from 'react';
import { Mic, UploadCloud, Play, AlertCircle, ShieldCheck, Cpu, ArrowRight } from 'lucide-react';
import { analyzeAudio } from '../services/api';

export default function Pillar3Page() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioMode, setAudioMode] = useState('spoken');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError(null);
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
    <div className="max-w-4xl mx-auto space-y-8 py-6">
      
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-purple-950/70 border border-purple-800/60 text-xs font-mono text-purple-300">
          <Mic className="w-3.5 h-3.5" />
          <span>PILLAR 3 • ACOUSTIC & SPEECH DEEPFAKE FORENSICS</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
          Audio & Speech Authenticity Detection
        </h1>
        <p className="text-sm text-slate-400 max-w-xl mx-auto font-mono">
          Detects AI voice cloning, speech synthesis (TTS), and audio manipulation with optional Harmonic-Percussive (HPSS) vocal demixing.
        </p>
      </div>

      {/* Upload Box */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
        
        {/* Mode selector */}
        <div className="flex items-center space-x-3 text-xs font-mono">
          <span className="text-slate-400">Analysis Mode:</span>
          <button
            onClick={() => setAudioMode('spoken')}
            className={`px-3 py-1.5 rounded-lg border transition-all ${
              audioMode === 'spoken'
                ? 'bg-purple-950/60 text-purple-300 border-purple-600'
                : 'bg-slate-900 text-slate-400 border-slate-800'
            }`}
          >
            🗣️ Spoken Voice (Conversational)
          </button>
          <button
            onClick={() => setAudioMode('music')}
            className={`px-3 py-1.5 rounded-lg border transition-all ${
              audioMode === 'music'
                ? 'bg-purple-950/60 text-purple-300 border-purple-600'
                : 'bg-slate-900 text-slate-400 border-slate-800'
            }`}
          >
            🎵 Song / Music Track (HPSS Demixing)
          </button>
        </div>

        {/* Upload Zone */}
        <label className="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all border-slate-700 hover:border-purple-500/50 hover:bg-slate-900/50 flex flex-col items-center justify-center">
          <input
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg"
            className="hidden"
            onChange={(e) => handleFileChange(e.target.files?.[0])}
          />
          <div className="w-12 h-12 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/30 flex items-center justify-center mb-3">
            <UploadCloud className="w-6 h-6" />
          </div>
          <span className="text-sm font-bold text-white mb-1">
            {selectedFile ? selectedFile.name : 'Upload Audio File'}
          </span>
          <span className="text-xs text-slate-400 font-mono">
            WAV, MP3, FLAC, M4A, OGG
          </span>
        </label>

        {/* Audio Player if file selected */}
        {selectedFile && (
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
            <audio controls src={URL.createObjectURL(selectedFile)} className="w-full sm:w-auto flex-1" />
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold text-xs font-mono transition-all flex items-center space-x-2 shrink-0 disabled:opacity-50"
            >
              {loading ? (
                <span>Extracting Acoustics...</span>
              ) : (
                <>
                  <span>Analyze Audio</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="p-3.5 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-300 text-xs font-mono">
            {error}
          </div>
        )}
      </div>

      {/* Result View */}
      {result && (
        <div className="space-y-6">
          
          {/* Verdict Banner */}
          <div className={`p-6 rounded-2xl border ${
            result.prediction === 'REAL' 
              ? 'bg-emerald-950/30 border-emerald-500/50' 
              : 'bg-rose-950/30 border-rose-500/50'
          } flex flex-col sm:flex-row sm:items-center justify-between gap-4`}>
            <div className="flex items-center space-x-4">
              <div className={`p-3 rounded-xl ${
                result.prediction === 'REAL' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
              }`}>
                {result.prediction === 'REAL' ? <ShieldCheck className="w-8 h-8" /> : <Cpu className="w-8 h-8" />}
              </div>
              <div>
                <span className="text-xs font-mono uppercase text-slate-400">Pillar 3 Audio Verdict</span>
                <h2 className={`text-2xl sm:text-3xl font-black ${
                  result.prediction === 'REAL' ? 'text-emerald-400' : 'text-rose-400'
                }`}>
                  {result.prediction === 'REAL' ? 'AUTHENTIC HUMAN VOICE' : 'AI SYNTHETIC / CLONED SPEECH'}
                </h2>
                <p className="text-xs text-slate-300 mt-0.5">
                  Target: <b>{result.filename}</b> • Duration: {result.duration}s • Mode: {result.mode}
                </p>
              </div>
            </div>

            <div className="text-right">
              <span className="text-xs font-mono text-slate-400 uppercase block mb-1">Confidence</span>
              <span className="text-3xl sm:text-4xl font-black text-white font-mono">
                {result.confidence}%
              </span>
            </div>
          </div>

          {/* Probability cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="glass-panel p-4 rounded-xl border border-slate-800">
              <span className="text-xs font-mono text-slate-400 block mb-1">Human Voice Probability</span>
              <span className="text-2xl font-bold font-mono text-emerald-400">{result.real_confidence}%</span>
            </div>
            <div className="glass-panel p-4 rounded-xl border border-slate-800">
              <span className="text-xs font-mono text-slate-400 block mb-1">Synthetic / AI Cloned Probability</span>
              <span className="text-2xl font-bold font-mono text-rose-400">{result.fake_confidence}%</span>
            </div>
          </div>

        </div>
      )}

    </div>
  );
}
