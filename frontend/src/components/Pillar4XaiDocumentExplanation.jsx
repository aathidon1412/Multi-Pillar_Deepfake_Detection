import React, { useState } from 'react';
import { 
  FileText, 
  BarChart3, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle2, 
  Sparkles, 
  ChevronDown, 
  ChevronUp, 
  HelpCircle, 
  ShieldAlert, 
  Hash, 
  Calculator, 
  Sigma, 
  Percent, 
  Layers
} from 'lucide-react';
import { getMediaUrl } from '../services/api';

export default function Pillar4XaiDocumentExplanation({
  xaiData,
  prediction = 'AUTHENTIC DOCUMENT',
  confidence = 85.0,
  digitsCount = 0
}) {
  const [showTechDetails, setShowTechDetails] = useState(false);
  const [activeDigitHover, setActiveDigitHover] = useState(null);
  const [viewMode, setViewMode] = useState('both'); // 'both' | 'chart' | 'table'

  const isAvailable = xaiData && xaiData.xai_available !== false;
  const isSparse = xaiData?.is_sparse || (digitsCount > 0 && digitsCount < 5) || (xaiData?.digits_count !== undefined && xaiData.digits_count < 5);

  const displayPred = String(
    xaiData?.prediction || prediction || 'AUTHENTIC DOCUMENT'
  ).toUpperCase();
  
  const isFake = displayPred.includes('FORGED') || displayPred.includes('SYNTHESIZED') || displayPred.includes('MANIPULATED') || displayPred.includes('FAKE');
  const confPct = xaiData?.confidence !== undefined
    ? Number(xaiData.confidence).toFixed(1)
    : (typeof confidence === 'number' ? confidence.toFixed(1) : '85.0');

  // Sparse or Unavailable State
  if (!isAvailable || isSparse) {
    const sparseCount = xaiData?.digits_count ?? digitsCount ?? 0;
    return (
      <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden transition-all">
        {/* Header */}
        <div className="p-5 bg-slate-900 text-white flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-amber-400" />
              <span className="text-[11px] font-mono uppercase tracking-widest text-amber-400 font-bold">
                Pillar 4 • Statistical Benford Forensic Explainability
              </span>
            </div>
            <h2 className="text-lg font-bold text-white tracking-tight">
              Why was this document flagged? (Statistical Explainability)
            </h2>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-3 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
              Sample Size: N = {sparseCount}
            </span>
            <span className="text-xs font-mono px-3 py-1 rounded bg-amber-950 text-amber-300 border border-amber-800 font-bold">
              SPARSE / LIMITED DATA
            </span>
          </div>
        </div>

        {/* Sparse Notice Body */}
        <div className="p-6 space-y-4">
          <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-1.5 leading-relaxed">
              <p className="font-bold text-sm text-amber-950">
                Statistical Explanation Unavailable or Limited (Sparse Numerical Tokens)
              </p>
              <p className="text-amber-800">
                {xaiData?.human_explanation || 
                  `This document contains fewer than the required numerical values (${sparseCount} detected). Benford's Law and Chi-Square goodness-of-fit testing require multi-digit transactional datasets (recommended N ≥ 15) to establish statistically valid attribution. No false statistical significance has been fabricated.`
                }
              </p>
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs font-mono text-slate-600">
            <strong>Preserved Policy:</strong> Non-document or sparse numerical media are safely handled without distorting forensic consensus.
          </div>
        </div>
      </div>
    );
  }

  const comparisonTable = xaiData.comparison_table || [];
  const topDeviations = xaiData.top_deviations || [];
  const maeVal = xaiData.mae !== undefined && xaiData.mae !== null ? Number(xaiData.mae).toFixed(4) : '—';
  const chiSqVal = xaiData.chi_square !== undefined && xaiData.chi_square !== null ? Number(xaiData.chi_square).toFixed(2) : '—';
  const pValFormatted = xaiData.p_value !== undefined && xaiData.p_value !== null
    ? (Number(xaiData.p_value) < 0.001 ? Number(xaiData.p_value).toExponential(4) : Number(xaiData.p_value).toFixed(4))
    : '—';
  const totalN = xaiData.digits_count || digitsCount || 0;

  const chartImgSrc = xaiData.chart_image_data_uri || (xaiData.chart_image ? getMediaUrl(xaiData.chart_image) : null);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden transition-all space-y-0">
      
      {/* 1. SECTION HEADER */}
      <div className="p-5 sm:p-6 bg-slate-900 text-white flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-cyan-400" />
            <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-400 font-bold">
              Pillar 4 • Statistical Benford Law Explainability
            </span>
          </div>
          <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight">
            Why was this document flagged? (Statistical Explainability)
          </h2>
          <p className="text-xs text-slate-300">
            Quantitative analysis comparing OCR extracted leading-digit distribution against theoretical Benford logarithmic PMF.
          </p>
        </div>

        {/* Verdict Badge */}
        <div className="flex items-center gap-2 self-start sm:self-center">
          <span className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold border ${
            isFake 
              ? 'bg-rose-950/90 border-rose-600/80 text-rose-300' 
              : 'bg-emerald-950/90 border-emerald-600/80 text-emerald-300'
          }`}>
            {isFake ? 'FORGED / ANOMALOUS' : 'AUTHENTIC DOCUMENT'} • {confPct}% CONFIDENCE
          </span>
        </div>
      </div>

      {/* 2. SUMMARY METRIC TILES: Extracted Count, MAE, Chi-Square, p-value */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-slate-200 border-b border-slate-200">
        
        {/* Tile 1: Extracted Numbers Count */}
        <div className="p-4 bg-white flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-[11px] font-mono uppercase font-semibold">1. Extracted Values</span>
            <Hash className="w-4 h-4 text-cyan-600" />
          </div>
          <div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-slate-900">{totalN}</div>
            <span className="text-[11px] text-slate-500">Leading digits parsed</span>
          </div>
        </div>

        {/* Tile 2: MAE */}
        <div className="p-4 bg-white flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-[11px] font-mono uppercase font-semibold">2. Mean Abs. Error</span>
            <Percent className="w-4 h-4 text-indigo-600" />
          </div>
          <div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-slate-900">{maeVal}</div>
            <span className="text-[11px] text-slate-500">Target threshold &lt; 0.025</span>
          </div>
        </div>

        {/* Tile 3: Chi-Square Statistic */}
        <div className="p-4 bg-white flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-[11px] font-mono uppercase font-semibold">3. Chi-Square (χ²)</span>
            <Sigma className="w-4 h-4 text-cyan-600" />
          </div>
          <div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-slate-900">{chiSqVal}</div>
            <span className="text-[11px] text-slate-500">Degrees of freedom df=8</span>
          </div>
        </div>

        {/* Tile 4: p-value */}
        <div className="p-4 bg-white flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-[11px] font-mono uppercase font-semibold">4. p-value</span>
            <Calculator className="w-4 h-4 text-emerald-600" />
          </div>
          <div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-slate-900">{pValFormatted}</div>
            <span className="text-[11px] text-slate-500">{Number(xaiData?.p_value || 0) < 0.01 ? 'Statistically significant' : 'Conforms to null hyp.'}</span>
          </div>
        </div>

      </div>

      {/* Main Content Area */}
      <div className="p-5 sm:p-6 space-y-6">

        {/* 3. BENFORD DISTRIBUTION CHART */}
        <div className="space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-cyan-600" />
                Benford Distribution Comparison Chart
              </h4>
              <p className="text-xs text-slate-500">
                Observed first-significant digit frequency (1–9) plotted against theoretical Benford distribution curve.
              </p>
            </div>

            {/* View Mode Switcher */}
            <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg self-start sm:self-auto">
              <button
                type="button"
                onClick={() => setViewMode('both')}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                  viewMode === 'both' ? 'bg-white text-slate-900 shadow-sm font-semibold' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Chart &amp; Table
              </button>
              <button
                type="button"
                onClick={() => setViewMode('chart')}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                  viewMode === 'chart' ? 'bg-white text-slate-900 shadow-sm font-semibold' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Chart Only
              </button>
              <button
                type="button"
                onClick={() => setViewMode('table')}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                  viewMode === 'table' ? 'bg-white text-slate-900 shadow-sm font-semibold' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Table Only
              </button>
            </div>
          </div>

          {/* Chart Display */}
          {(viewMode === 'both' || viewMode === 'chart') && (
            <div className="rounded-xl overflow-hidden border border-slate-300 bg-slate-950 p-2 shadow-inner">
              {chartImgSrc ? (
                <img
                  src={chartImgSrc}
                  alt="Benford Distribution Comparison Plot"
                  className="w-full h-auto object-contain rounded select-none"
                />
              ) : (
                <div className="py-16 text-center text-xs font-mono text-slate-400">
                  Rendering statistical distribution plot...
                </div>
              )}
            </div>
          )}
        </div>

        {/* 4. LARGEST DEVIATIONS CALLOUT CARDS */}
        {topDeviations && topDeviations.length > 0 && (
          <div className="space-y-2.5 pt-2">
            <h4 className="text-xs font-mono uppercase tracking-wider font-semibold text-slate-700 flex items-center gap-1.5">
              <TrendingUp className="w-4 h-4 text-rose-500" />
              Largest Observed Deviations from Theoretical Benford PMF
            </h4>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {topDeviations.map((dev, idx) => {
                const isOver = dev.diff_pct > 0;
                return (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 hover:border-slate-300 transition-all flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="w-7 h-7 rounded-lg bg-slate-900 text-cyan-300 font-mono font-bold text-sm flex items-center justify-center">
                          {dev.digit}
                        </span>
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold border ${
                          isOver 
                            ? 'bg-rose-50 text-rose-700 border-rose-200' 
                            : 'bg-sky-50 text-sky-700 border-sky-200'
                        }`}>
                          {isOver ? `+${dev.abs_diff_pct}% OVER` : `-${dev.abs_diff_pct}% UNDER`}
                        </span>
                      </div>

                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between text-slate-600">
                          <span>Observed:</span>
                          <strong className="font-mono text-slate-900">{dev.observed_pct}%</strong>
                        </div>
                        <div className="flex justify-between text-slate-600">
                          <span>Expected Benford:</span>
                          <span className="font-mono text-slate-700">{dev.expected_pct}%</span>
                        </div>
                        <div className="flex justify-between text-slate-600 pt-1 border-t border-slate-200">
                          <span>Absolute Difference:</span>
                          <strong className="font-mono text-rose-600">{dev.abs_diff_pct}%</strong>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* 5. COMPARISON TABLE: Digit | Observed | Expected | Difference */}
        {(viewMode === 'both' || viewMode === 'table') && comparisonTable.length > 0 && (
          <div className="space-y-2.5 pt-2">
            <h4 className="text-xs font-mono uppercase tracking-wider font-semibold text-slate-700">
              Quantitative Digit Comparison Table (1–9)
            </h4>

            <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-100/80 text-slate-700 uppercase text-[10px] tracking-wider border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3.5 font-bold">Leading Digit</th>
                    <th className="py-2.5 px-3.5 font-bold">Observed %</th>
                    <th className="py-2.5 px-3.5 font-bold">Expected Benford %</th>
                    <th className="py-2.5 px-3.5 font-bold">Difference (Δ %)</th>
                    <th className="py-2.5 px-3.5 font-bold">Obs / Exp Count</th>
                    <th className="py-2.5 px-3.5 font-bold">Forensic Signal Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-800">
                  {comparisonTable.map((row) => {
                    const isTop = row.is_top_deviant;
                    const isOver = row.diff_pct > 0;
                    return (
                      <tr 
                        key={row.digit}
                        className={`transition-colors hover:bg-slate-50 ${isTop ? 'bg-amber-50/40' : ''}`}
                      >
                        <td className="py-2 px-3.5 font-bold text-slate-900 flex items-center gap-1.5">
                          <span className={`w-5 h-5 rounded flex items-center justify-center text-[11px] font-bold ${
                            isTop ? 'bg-slate-900 text-cyan-300' : 'bg-slate-100 text-slate-700'
                          }`}>
                            {row.digit}
                          </span>
                          {isTop && (
                            <span className="text-[9px] font-bold text-amber-700 bg-amber-100 px-1.5 py-0.2 rounded">
                              Top Anomaly
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-3.5 font-bold text-slate-900">{row.observed_pct}%</td>
                        <td className="py-2 px-3.5 text-slate-600">{row.expected_pct}%</td>
                        <td className={`py-2 px-3.5 font-bold ${
                          Math.abs(row.diff_pct) > 5.0 
                            ? (isOver ? 'text-rose-600' : 'text-sky-600') 
                            : 'text-slate-700'
                        }`}>
                          {row.diff_pct > 0 ? `+${row.diff_pct}%` : `${row.diff_pct}%`}
                        </td>
                        <td className="py-2 px-3.5 text-slate-500">
                          {row.obs_count} / {row.exp_count}
                        </td>
                        <td className="py-2 px-3.5">
                          <span className={`inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded ${
                            Math.abs(row.diff_pct) < 2.5
                              ? 'bg-emerald-50 text-emerald-700'
                              : (isOver ? 'bg-rose-50 text-rose-700' : 'bg-sky-50 text-sky-700')
                          }`}>
                            {Math.abs(row.diff_pct) < 2.5 ? 'Conforms' : (isOver ? 'Over-indexed' : 'Under-indexed')}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 6. DYNAMIC HUMAN-READABLE EXPLANATION */}
        <div className="p-4 rounded-xl bg-cyan-50/70 border border-cyan-200 space-y-1.5">
          <div className="flex items-center gap-2 text-cyan-900 font-bold text-xs uppercase tracking-wider font-mono">
            <Sparkles className="w-4 h-4 text-cyan-700" />
            Plain-Language Statistical Summary
          </div>
          <p className="text-xs sm:text-sm text-slate-800 leading-relaxed font-medium">
            {xaiData.human_explanation}
          </p>
        </div>

        {/* 7. TECHNICAL TELEMETRY ACCORDION */}
        <div>
          <button
            type="button"
            onClick={() => setShowTechDetails(!showTechDetails)}
            className="flex items-center justify-between w-full py-2.5 px-4 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-mono text-slate-700 font-semibold transition-colors"
          >
            <span>🔬 Researcher Technical Goodness-of-Fit Telemetry &amp; Formulae</span>
            {showTechDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showTechDetails && (
            <div className="mt-2 p-4 bg-slate-900 text-slate-200 rounded-lg text-xs font-mono space-y-2 border border-slate-800 leading-relaxed">
              <div><strong>Technical Telemetry:</strong> {xaiData.technical_explanation}</div>
              <div><strong>Probability Mass Function:</strong> P(d) = log₁₀(1 + 1/d) for d ∈ &#123;1, 2, ..., 9&#125;</div>
              <div><strong>Chi-Square Formulation:</strong> χ² = Σ ((Observed_i - Expected_i)² / Expected_i) across 9 bins (df = 8)</div>
              <div><strong>Mean Absolute Error:</strong> MAE = (1/9) Σ |Observed_i - Expected_i|</div>
              <div><strong>Sample Size Scaling:</strong> Threshold strict = 0.020 + (0.010 × 100 / max(N, 30))</div>
              {xaiData.limitations && xaiData.limitations.length > 0 && (
                <div className="pt-2 border-t border-slate-800 text-slate-400 space-y-1">
                  <strong>Methodological Scope &amp; Constraints:</strong>
                  {xaiData.limitations.map((lim, lIdx) => (
                    <div key={lIdx}>• {lim}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 8. MANDATORY NON-CAUSAL FORENSIC DISCLAIMER */}
        <div className="p-3.5 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg text-xs text-amber-900 leading-relaxed shadow-sm">
          ⚠️ <strong>Forensic Evidence Notice:</strong> This statistical pattern is consistent with the forensic signal detected by the system and should be considered alongside the other evidence. Statistical deviation alone does not constitute definitive proof of fraud or forgery.
        </div>

      </div>

    </div>
  );
}
