import React, { useState } from 'react';
import { UploadCloud, Image as ImageIcon, ShieldCheck, AlertTriangle, Eye, Layers, ArrowRight } from 'lucide-react';
import { analyzeImage } from '../services/api';

export default function Pillars1And5Page() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const data = await analyzeImage(selectedFile);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Image analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 py-6">
      
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800/60 text-xs font-mono text-cyan-300">
          <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
          <span>PILLARS 1 & 5 • IMAGE & SHADOW PHYSICS FORENSICS</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
          Universal Multi-Pillar Image Forensics
        </h1>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto font-mono">
          Combines Vision Transformer (ViT) spatial neural feature extraction with Physical Shadow Geometry and RANSAC illumination convergence.
        </p>
      </div>

      {/* Upload Box */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          
          {/* Dropzone */}
          <div>
            <label className="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all border-slate-700 hover:border-cyan-500/50 hover:bg-slate-900/50 flex flex-col items-center justify-center">
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFileChange(e.target.files?.[0])}
              />
              <div className="w-12 h-12 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 flex items-center justify-center mb-3">
                <UploadCloud className="w-6 h-6" />
              </div>
              <span className="text-sm font-bold text-white mb-1">
                {selectedFile ? selectedFile.name : 'Upload Target Image'}
              </span>
              <span className="text-xs text-slate-400 font-mono">
                JPG, PNG, JPEG, WEBP, BMP
              </span>
            </label>
          </div>

          {/* Preview */}
          <div className="aspect-video bg-black/70 rounded-xl border border-slate-800 overflow-hidden flex items-center justify-center relative">
            {previewUrl ? (
              <img src={previewUrl} alt="Target" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center text-slate-600 font-mono text-xs">
                <ImageIcon className="w-8 h-8 mx-auto mb-2 opacity-40" />
                <span>Image preview will appear here</span>
              </div>
            )}
          </div>

        </div>

        {/* Analyze Button */}
        {selectedFile && (
          <div className="flex justify-center pt-2">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-sm shadow-glow-cyan flex items-center space-x-2 transition-all disabled:opacity-50"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Computing Neural & Physics Vectors...</span>
                </>
              ) : (
                <>
                  <span>Execute Multi-Pillar Analysis</span>
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

      {/* Results Section */}
      {result && (
        <div className="space-y-6">
          
          {/* Master Verdict Card */}
          <div className={`p-6 rounded-2xl border ${
            result.is_real 
              ? 'bg-emerald-950/30 border-emerald-500/50' 
              : 'bg-rose-950/30 border-rose-500/50'
          } flex flex-col sm:flex-row sm:items-center justify-between gap-4`}>
            <div className="flex items-center space-x-4">
              <div className={`p-3 rounded-xl ${
                result.is_real ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
              }`}>
                {result.is_real ? <ShieldCheck className="w-8 h-8" /> : <AlertTriangle className="w-8 h-8" />}
              </div>
              <div>
                <span className="text-xs font-mono uppercase text-slate-400">Consolidated Consensus Verdict</span>
                <h2 className={`text-2xl sm:text-3xl font-black ${
                  result.is_real ? 'text-emerald-400' : 'text-rose-400'
                }`}>
                  {result.consensus_verdict}
                </h2>
                <p className="text-xs text-slate-300 mt-0.5">
                  Target: <b>{result.filename}</b> • Engines: Pillar 1 (ViT) + Pillar 5 (Shadow Physics)
                </p>
              </div>
            </div>

            <div className="text-right">
              <span className="text-xs font-mono text-slate-400 uppercase block mb-1">Consensus Confidence</span>
              <span className="text-3xl sm:text-4xl font-black text-white font-mono">
                {result.consensus_confidence}%
              </span>
            </div>
          </div>

          {/* 2-Pillar Comparison Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Pillar 1 */}
            <div className="glass-panel p-5 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-800 text-cyan-400">
                    <Eye className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-xs uppercase font-mono text-cyan-400 font-bold block">Pillar 1</span>
                    <span className="text-sm font-bold text-white">Vision Transformer</span>
                  </div>
                </div>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold ${
                  result.pillar1.verdict === 'AUTHENTIC' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
                }`}>
                  {result.pillar1.verdict}
                </span>
              </div>
              <div className="space-y-1.5 text-xs font-mono text-slate-300 pt-2 border-t border-slate-800">
                <div className="flex justify-between">
                  <span className="text-slate-500">Confidence:</span>
                  <span className="font-bold">{result.pillar1.confidence}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Fake Anomaly Score:</span>
                  <span className="text-cyan-400">{result.pillar1.fake_score}</span>
                </div>
                <div className="text-slate-500 text-[11px] truncate">
                  Model: {result.pillar1.model}
                </div>
              </div>
            </div>

            {/* Pillar 5 */}
            <div className="glass-panel p-5 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="p-2 rounded-lg bg-blue-950 border border-blue-800 text-blue-400">
                    <Layers className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-xs uppercase font-mono text-blue-400 font-bold block">Pillar 5</span>
                    <span className="text-sm font-bold text-white">Shadow Physics & Geometry</span>
                  </div>
                </div>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold ${
                  result.pillar5.inlier_ratio >= 0.5 ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
                }`}>
                  {result.pillar5.inlier_ratio >= 0.5 ? 'AUTHENTIC' : 'SHADOW ANOMALY'}
                </span>
              </div>
              <div className="space-y-1.5 text-xs font-mono text-slate-300 pt-2 border-t border-slate-800">
                <div className="flex justify-between">
                  <span className="text-slate-500">Confidence:</span>
                  <span className="font-bold">{result.pillar5.confidence}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">RANSAC Inliers:</span>
                  <span className="text-blue-400">{result.pillar5.inliers} / {result.pillar5.total_lines} ({Math.round(result.pillar5.inlier_ratio*100)}%)</span>
                </div>
                <div className="text-slate-500 text-[11px] truncate">
                  Engine: {result.pillar5.engine}
                </div>
              </div>
            </div>

          </div>

        </div>
      )}

    </div>
  );
}
