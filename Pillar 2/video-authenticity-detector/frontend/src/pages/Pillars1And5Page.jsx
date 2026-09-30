import React, { useState } from 'react';
import { UploadCloud, Image as ImageIcon, ShieldCheck, AlertTriangle, Eye, Layers, ArrowRight, CheckCircle2, Sliders } from 'lucide-react';
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

  const isReal = result?.is_real ?? false;
  const p1 = result?.pillar1;
  const p5 = result?.pillar5;
  const p4 = result?.pillar4;

  return (
    <div className="max-w-5xl mx-auto space-y-6 py-8 px-4 sm:px-6">
      
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs font-mono text-slate-700">
          <ImageIcon className="w-3.5 h-3.5 text-slate-800" />
          <span className="uppercase tracking-wider font-semibold">PILLARS 1, 4 & 5 • UNIVERSAL IMAGE & SHADOW FORENSICS</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900 tracking-tight">
          Visual Media Deepfake & Synthesis Detector
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 max-w-xl mx-auto">
          Multi-engine forensic decomposition combining Vision Transformer (ViT) patch attention, RANSAC shadow physics, and Benford's Law OCR.
        </p>
      </div>

      {/* Main Container */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-5">
        
        {/* Upload Zone */}
        {!selectedFile ? (
          <label className="border-2 border-dashed border-slate-300 hover:border-slate-400 hover:bg-slate-50 rounded-xl p-8 text-center flex flex-col items-center justify-center cursor-pointer transition-all">
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
            />
            <div className="w-12 h-12 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-700 mb-3">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-slate-900">
              Select or Drop Target Photo / Render
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports JPEG, PNG, WEBP, TIFF, BMP up to 50MB
            </p>
          </label>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="flex items-center gap-2.5 truncate">
                <ImageIcon className="w-5 h-5 text-slate-700 shrink-0" />
                <span className="text-xs font-semibold text-slate-900 truncate">{selectedFile.name}</span>
                <span className="text-[11px] font-mono text-slate-500">({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)</span>
              </div>
              <button
                type="button"
                onClick={() => { setSelectedFile(null); setPreviewUrl(null); setResult(null); }}
                className="text-xs text-slate-500 hover:text-rose-600 underline font-mono"
              >
                Change Photo
              </button>
            </div>

            {previewUrl && (
              <div className="max-h-80 rounded-lg overflow-hidden bg-slate-900 border border-slate-200 flex items-center justify-center p-2">
                <img src={previewUrl} alt="Preview" className="max-h-72 object-contain rounded" />
              </div>
            )}

            <button
              type="button"
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full py-3 px-4 rounded-lg bg-slate-900 hover:bg-slate-800 disabled:opacity-75 text-white text-xs font-medium flex items-center justify-center gap-2 shadow-sm transition-all"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Executing ViT Neural Attention & Shadow RANSAC Pipeline...</span>
                </>
              ) : (
                <>
                  <span>Run Multi-Pillar Deepfake Inspection</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs font-mono">
            {error}
          </div>
        )}

        {/* Results View */}
        {result && (
          <div className="mt-6 pt-6 border-t border-slate-100 space-y-6">
            
            {/* Unified Consensus Verdict Card */}
            <div className={`p-5 rounded-xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 ${
              isReal ? 'bg-emerald-50/70 border-emerald-200 text-emerald-900' : 'bg-rose-50/70 border-rose-200 text-rose-900'
            }`}>
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-500 block">
                  Consolidated Multi-Pillar Ensemble Verdict
                </span>
                <h3 className="text-xl font-bold tracking-tight mt-1">
                  {result.verdict}
                </h3>
                <p className="text-xs text-slate-600 mt-1">
                  Engines: <strong>{result.engines}</strong>
                  {result.override_reason && (
                    <span className="block text-amber-700 mt-0.5">• Note: {result.override_reason}</span>
                  )}
                </p>
              </div>

              <div className="text-right shrink-0">
                <span className="text-[11px] font-mono uppercase text-slate-500 block">Confidence</span>
                <span className="text-3xl font-bold font-mono">
                  {result.confidence}%
                </span>
              </div>
            </div>

            {/* 3-Column Pillar Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              
              {/* Pillar 1: ViT */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-slate-500 font-semibold">PILLAR 1 • ViT SPECTRA</span>
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
                      p1?.verdict === 'AUTHENTIC' ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                    }`}>
                      {p1?.verdict || 'EVALUATED'}
                    </span>
                  </div>
                  <h4 className="text-sm font-semibold text-slate-900">Neural Patch Artifacts</h4>
                  <p className="text-xs text-slate-500 mt-1">
                    Evaluates transformer self-attention representations for diffusion and GAN synthesis signatures.
                  </p>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-200/60 font-mono text-xs text-slate-700 flex justify-between">
                  <span>Confidence:</span>
                  <span className="font-semibold">{p1?.confidence}%</span>
                </div>
              </div>

              {/* Pillar 5: Shadow RANSAC */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-slate-500 font-semibold">PILLAR 5 • RANSAC PHYSICS</span>
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
                      p5?.is_real ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                    }`}>
                      {p5?.verdict || 'PHYSICS CHECK'}
                    </span>
                  </div>
                  <h4 className="text-sm font-semibold text-slate-900">Shadow & Vanishing Lines</h4>
                  <p className="text-xs text-slate-500 mt-1">
                    Geometric perspective ray consistency across cast shadows and optical lighting points.
                  </p>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-200/60 font-mono text-xs text-slate-700 flex justify-between">
                  <span>Inlier Consensus:</span>
                  <span className="font-semibold">{Math.round((p5?.inlier_ratio || 0) * 100)}%</span>
                </div>
              </div>

              {/* Pillar 4: Benford OCR */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-slate-500 font-semibold">PILLAR 4 • STATISTICAL OCR</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-200 text-slate-700">
                      {p4?.applicable ? p4?.verdict : 'N/A (NON-DOC)'}
                    </span>
                  </div>
                  <h4 className="text-sm font-semibold text-slate-900">Benford's Law Digits</h4>
                  <p className="text-xs text-slate-500 mt-1">
                    {p4?.applicable 
                      ? `Tested ${p4.digits_count} leading numeric values across document surface.`
                      : 'No numeric invoice tabular records detected in visual crop.'}
                  </p>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-200/60 font-mono text-xs text-slate-700 flex justify-between">
                  <span>Digits Detected:</span>
                  <span className="font-semibold">{p4?.digits_count || 0}</span>
                </div>
              </div>

            </div>

          </div>
        )}

      </div>
    </div>
  );
}
