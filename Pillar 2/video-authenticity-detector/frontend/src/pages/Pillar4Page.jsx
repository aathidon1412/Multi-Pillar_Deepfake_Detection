import React, { useState } from 'react';
import { FileText, UploadCloud, ArrowRight, ShieldCheck, AlertTriangle, BarChart3 } from 'lucide-react';
import { analyzeDocument } from '../services/api';

export default function Pillar4Page() {
  const [selectedFile, setSelectedFile] = useState(null);
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
      const data = await analyzeDocument(selectedFile);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Document analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 py-6">
      
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-950/70 border border-emerald-800/60 text-xs font-mono text-emerald-300">
          <FileText className="w-3.5 h-3.5" />
          <span>PILLAR 4 • DOCUMENT & PDF BENFORD FORENSICS</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
          Document & Invoice Authenticity Inspection
        </h1>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto font-mono">
          Performs OCR digit extraction and applies Benford's Law Chi-Square goodness-of-fit to detect manipulated, altered, or AI-generated financial records.
        </p>
      </div>

      {/* Upload Box */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
        <label className="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all border-slate-700 hover:border-emerald-500/50 hover:bg-slate-900/50 flex flex-col items-center justify-center">
          <input
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.tiff"
            className="hidden"
            onChange={(e) => handleFileChange(e.target.files?.[0])}
          />
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center justify-center mb-3">
            <UploadCloud className="w-6 h-6" />
          </div>
          <span className="text-sm font-bold text-white mb-1">
            {selectedFile ? selectedFile.name : 'Upload Document or PDF'}
          </span>
          <span className="text-xs text-slate-400 font-mono">
            PDF, PNG, JPG, JPEG, TIFF
          </span>
        </label>

        {selectedFile && (
          <div className="flex justify-center pt-2">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-sm font-mono shadow-glow-emerald flex items-center space-x-2 transition-all disabled:opacity-50"
            >
              {loading ? (
                <span>Executing Tesseract OCR & Chi-Square Fit...</span>
              ) : (
                <>
                  <span>Analyze Document Integrity</span>
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

      {/* Results View */}
      {result && (
        <div className="space-y-6">
          
          {result.applicable ? (
            <>
              {/* Verdict Banner */}
              <div className={`p-6 rounded-2xl border ${
                result.is_authentic 
                  ? 'bg-emerald-950/30 border-emerald-500/50' 
                  : 'bg-rose-950/30 border-rose-500/50'
              } flex flex-col sm:flex-row sm:items-center justify-between gap-4`}>
                <div className="flex items-center space-x-4">
                  <div className={`p-3 rounded-xl ${
                    result.is_authentic ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                  }`}>
                    {result.is_authentic ? <ShieldCheck className="w-8 h-8" /> : <AlertTriangle className="w-8 h-8" />}
                  </div>
                  <div>
                    <span className="text-xs font-mono uppercase text-slate-400">Pillar 4 Statistical Verdict</span>
                    <h2 className={`text-2xl sm:text-3xl font-black ${
                      result.is_authentic ? 'text-emerald-400' : 'text-rose-400'
                    }`}>
                      {result.verdict}
                    </h2>
                    <p className="text-xs text-slate-300 mt-0.5">
                      Target: <b>{result.filename}</b> • Digits Analyzed: {result.digits_count} • MAE: {result.mae}
                    </p>
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-xs font-mono text-slate-400 uppercase block mb-1">Statistical Confidence</span>
                  <span className="text-3xl sm:text-4xl font-black text-white font-mono">
                    {result.confidence}%
                  </span>
                </div>
              </div>

              {/* Statistical Metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="glass-panel p-4 rounded-xl border border-slate-800">
                  <span className="text-xs font-mono text-slate-400 block mb-1">Digits Extracted</span>
                  <span className="text-2xl font-bold font-mono text-white">{result.digits_count}</span>
                </div>
                <div className="glass-panel p-4 rounded-xl border border-slate-800">
                  <span className="text-xs font-mono text-slate-400 block mb-1">Mean Absolute Error</span>
                  <span className="text-2xl font-bold font-mono text-cyan-400">{result.mae}</span>
                </div>
                <div className="glass-panel p-4 rounded-xl border border-slate-800">
                  <span className="text-xs font-mono text-slate-400 block mb-1">Chi-Square (χ²)</span>
                  <span className="text-2xl font-bold font-mono text-blue-400">{result.chi_square}</span>
                </div>
                <div className="glass-panel p-4 rounded-xl border border-slate-800">
                  <span className="text-xs font-mono text-slate-400 block mb-1">P-Value</span>
                  <span className="text-2xl font-bold font-mono text-purple-400">{result.p_value}</span>
                </div>
              </div>

              {/* Benford's Law Comparison Table */}
              <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                <h4 className="text-sm font-bold font-mono text-white flex items-center space-x-2">
                  <BarChart3 className="w-4 h-4 text-emerald-400" />
                  <span>First-Digit Frequencies vs Benford's Law Distribution</span>
                </h4>

                <div className="grid grid-cols-9 gap-2 text-center text-xs font-mono">
                  {result.obs_freqs && result.obs_freqs.map((obs, idx) => (
                    <div key={idx} className="p-2 rounded bg-slate-950 border border-slate-800 space-y-1">
                      <span className="text-slate-400 font-bold block text-sm">{idx + 1}</span>
                      <div className="text-[11px] text-emerald-400 font-bold">{obs}%</div>
                      <div className="text-[10px] text-slate-500">{result.expected_freqs[idx]}% exp</div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="p-6 rounded-xl bg-amber-950/40 border border-amber-800 text-amber-300 text-xs font-mono">
              ⚠️ {result.reason}
            </div>
          )}

        </div>
      )}

    </div>
  );
}
