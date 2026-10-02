import React, { useState } from 'react';
import { Eye, EyeOff, Sliders, Info, ShieldAlert, Sparkles, HelpCircle, Cpu, Layers } from 'lucide-react';
import { getMediaUrl } from '../services/api';

export default function Pillar1XaiExplanation({
  xaiData,
  originalImageSrc,
  prediction = 'REAL',
  confidence = 85.0
}) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.60);
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' | 'split'

  const isAvailable = xaiData && xaiData.xai_available !== false;

  const normalizedPred = String(
    xaiData?.prediction || prediction || 'REAL'
  ).toUpperCase().includes('FAKE') || String(xaiData?.prediction || prediction || '').toUpperCase().includes('AI')
    ? 'FAKE'
    : 'REAL';

  const normalizedConf = xaiData?.confidence !== undefined
    ? Number(xaiData.confidence).toFixed(1)
    : (typeof confidence === 'number' ? confidence.toFixed(1) : '85.0');

  const method = xaiData?.method || 'Attention Rollout';

  if (!isAvailable) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
          <div>
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-500 font-semibold block">
              Explainable AI (XAI)
            </span>
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              PILLAR 1 — WHY THIS PREDICTION?
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-100 text-slate-700 font-semibold">
              Method: {method}
            </span>
            <span className={`text-xs font-mono px-2.5 py-1 rounded font-bold ${
              normalizedPred === 'REAL' ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
            }`}>
              {normalizedPred} ({normalizedConf}%)
            </span>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start gap-3">
          <Info className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold">XAI explanation is unavailable for this sample.</p>
            <p className="text-amber-700">
              The Vision Transformer classification output was generated successfully ({normalizedPred} with {normalizedConf}% confidence), but attention matrices were bypassed or not computed for this specific format.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const visualization = (xaiData.visualizations && xaiData.visualizations.length > 0)
    ? xaiData.visualizations[0]
    : null;

  // Use base64 data URI if available for instant zero-latency rendering, or fallback to relative URL
  const overlaySrc = visualization?.overlay_data_uri || (visualization?.overlay_url ? getMediaUrl(visualization.overlay_url) : (xaiData.overlay_url ? getMediaUrl(xaiData.overlay_url) : null));
  const heatmapSrc = visualization?.heatmap_data_uri || (visualization?.heatmap_url ? getMediaUrl(visualization.heatmap_url) : (xaiData.heatmap_url ? getMediaUrl(xaiData.heatmap_url) : null));
  const rawOrigSrc = originalImageSrc || visualization?.orig_data_uri || (visualization?.orig_url ? getMediaUrl(visualization.orig_url) : (xaiData.orig_url ? getMediaUrl(xaiData.orig_url) : null));

  const peakRegion = visualization?.peak_region || xaiData.highest_attribution_region || 'central focal region';
  const peakCoords = visualization?.peak_coords || [0.5, 0.5];

  const humanExplanation = xaiData.human_explanation || (
    `Attention Rollout identifies the image regions that received higher model attention for the selected ${normalizedPred} prediction (${normalizedConf}% confidence). The highlighted regions around the ${peakRegion} represent areas that contributed more strongly to the model's decision.`
  );

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden transition-all space-y-0">
      
      {/* SECTION HEADER: PILLAR 1 — WHY THIS PREDICTION? */}
      <div className="p-5 sm:p-6 bg-slate-900 text-white flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-lg">🧠</span>
            <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-400 font-bold">
              Explainable AI Layer
            </span>
          </div>
          <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight">
            PILLAR 1 — WHY THIS PREDICTION?
          </h2>
          <p className="text-xs text-slate-300">
            Method: <strong className="text-cyan-300 font-mono">{method}</strong> • Model: <strong className="text-slate-200">Vision Transformer (ViT-Base-16)</strong>
          </p>
        </div>

        {/* Prediction, Confidence & Show/Hide Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-right">
            <span className="text-[10px] font-mono uppercase text-slate-400 block">Target Prediction</span>
            <span className={`text-sm font-mono font-bold ${
              normalizedPred === 'REAL' ? 'text-emerald-400' : 'text-rose-400'
            }`}>
              {normalizedPred} ({normalizedConf}%)
            </span>
          </div>

          <button
            type="button"
            onClick={() => setShowHeatmap(!showHeatmap)}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-cyan-950 hover:bg-cyan-900 text-cyan-200 text-xs font-semibold border border-cyan-800 transition-colors shadow-sm cursor-pointer"
          >
            {showHeatmap ? (
              <>
                <EyeOff className="w-3.5 h-3.5 text-cyan-300" />
                <span>Hide Heatmap</span>
              </>
            ) : (
              <>
                <Eye className="w-3.5 h-3.5 text-cyan-400" />
                <span>Show Heatmap</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Body */}
      <div className="p-5 sm:p-6 space-y-6">
        
        {/* Human-Readable Explanation Box */}
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
          <div className="flex items-center gap-2 text-slate-900 font-semibold text-xs tracking-wide uppercase font-mono">
            <Sparkles className="w-4 h-4 text-cyan-600" />
            <span>Human-Readable Explanation</span>
          </div>
          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
            {humanExplanation}
          </p>
        </div>

        {/* Interactive Controls: Opacity Slider + Mode Switcher */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-3.5 rounded-xl bg-slate-50 border border-slate-200">
          
          {/* Mode Switcher */}
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setViewMode('overlay')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                viewMode === 'overlay'
                  ? 'bg-slate-900 text-white shadow-sm'
                  : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
              }`}
            >
              Interactive Overlay
            </button>
            <button
              type="button"
              onClick={() => setViewMode('split')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                viewMode === 'split'
                  ? 'bg-slate-900 text-white shadow-sm'
                  : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
              }`}
            >
              Side-by-Side View
            </button>
          </div>

          {/* Heatmap Opacity Slider (Only in overlay mode and when heatmap is visible) */}
          {viewMode === 'overlay' && showHeatmap && (
            <div className="flex items-center gap-3 w-full sm:w-80">
              <Sliders className="w-3.5 h-3.5 text-slate-500 shrink-0" />
              <span className="text-xs font-medium text-slate-700 shrink-0">Heatmap Opacity:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.02"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="w-full accent-slate-900 cursor-pointer"
              />
              <span className="text-xs font-mono font-bold text-slate-900 shrink-0 w-10 text-right">
                {Math.round(opacity * 100)}%
              </span>
            </div>
          )}
        </div>

        {/* Visual Heatmap Canvas Container */}
        {viewMode === 'overlay' ? (
          /* Interactive Overlaid Canvas */
          <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-200 min-h-[300px] max-h-[500px] flex items-center justify-center p-3">
            {/* Layer 1: Original Image */}
            {rawOrigSrc ? (
              <img
                src={rawOrigSrc}
                alt="Original Image Target"
                className="max-h-[480px] w-auto max-w-full object-contain rounded select-none"
              />
            ) : (
              <div className="text-slate-500 font-mono text-xs">Original image preview</div>
            )}

            {/* Layer 2: Colored Heatmap Overlay with dynamic opacity */}
            {showHeatmap && heatmapSrc && (
              <img
                src={heatmapSrc}
                alt="Attention Rollout Heatmap Overlay"
                style={{ opacity: opacity }}
                className="absolute inset-0 m-auto max-h-[480px] w-auto max-w-full object-contain rounded pointer-events-none transition-opacity duration-75 mix-blend-screen select-none"
              />
            )}
          </div>
        ) : (
          /* Side-by-Side Comparison Panels */
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Panel 1: Original Image */}
            <div className="rounded-xl border border-slate-200 bg-slate-950 overflow-hidden flex flex-col">
              <div className="px-3.5 py-2.5 bg-slate-900 text-slate-300 text-xs font-mono font-semibold flex items-center justify-between border-b border-slate-800">
                <span>[Original Image]</span>
                <span className="text-[11px] text-slate-400">Input Sample</span>
              </div>
              <div className="p-3 flex items-center justify-center min-h-[280px] max-h-[380px]">
                {rawOrigSrc ? (
                  <img
                    src={rawOrigSrc}
                    alt="Original Image"
                    className="max-h-[360px] w-auto max-w-full object-contain rounded"
                  />
                ) : (
                  <span className="text-xs text-slate-500 font-mono">Original image preview</span>
                )}
              </div>
            </div>

            {/* Panel 2: XAI Heatmap */}
            <div className="rounded-xl border border-slate-200 bg-slate-950 overflow-hidden flex flex-col">
              <div className="px-3.5 py-2.5 bg-slate-900 text-cyan-300 text-xs font-mono font-semibold flex items-center justify-between border-b border-slate-800">
                <span>[AI Explanation Heatmap]</span>
                <span className="text-[11px] text-cyan-400">Attention Rollout</span>
              </div>
              <div className="p-3 flex items-center justify-center min-h-[280px] max-h-[380px]">
                {heatmapSrc ? (
                  <img
                    src={heatmapSrc}
                    alt="AI Explanation Heatmap"
                    className="max-h-[360px] w-auto max-w-full object-contain rounded"
                  />
                ) : (
                  <span className="text-xs text-slate-500 font-mono">Heatmap unavailable</span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Attribution Legend */}
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-slate-700 font-semibold">
            <span>Low influence</span>
            <span className="text-[11px] text-slate-500 font-normal">Transformer Attention Gradient Scale</span>
            <span>High influence</span>
          </div>
          {/* Smooth Continuous Gradient */}
          <div
            className="h-3.5 w-full rounded-full border border-slate-300/80 shadow-inner"
            style={{
              background: 'linear-gradient(to right, #000080 0%, #0000ff 20%, #00ffff 40%, #00ff00 60%, #ffff00 80%, #ff0000 100%)'
            }}
          />
        </div>

        {/* Evidence Factors Summary */}
        {xaiData.evidence && xaiData.evidence.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {xaiData.evidence.map((ev, idx) => (
              <div key={idx} className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-[11px] font-semibold text-slate-800">
                      {ev.feature.replace(/_/g, ' ')}
                    </span>
                    <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded font-bold ${
                      ev.direction === 'supports_fake'
                        ? 'bg-rose-100 text-rose-800'
                        : 'bg-emerald-100 text-emerald-800'
                    }`}>
                      {ev.direction === 'supports_fake' ? 'Supports Fake' : 'Supports Real'}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-600 leading-normal">
                    {ev.description}
                  </p>
                </div>
                <div className="mt-2 pt-2 border-t border-slate-200 text-[10px] font-mono text-slate-500 flex justify-between">
                  <span>Feature Value:</span>
                  <span className="font-semibold text-slate-700">{String(ev.value)}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Expandable Sections: "What does this mean?" & "Technical details" */}
        <div className="space-y-2.5">
          
          {/* 1. What does this mean? */}
          <details className="group rounded-xl border border-slate-200 bg-white p-3.5 transition-colors open:bg-slate-50/50">
            <summary className="flex items-center justify-between cursor-pointer text-xs font-semibold text-slate-800 select-none">
              <div className="flex items-center gap-2">
                <HelpCircle className="w-4 h-4 text-cyan-600 shrink-0" />
                <span>What does this mean?</span>
              </div>
              <span className="text-slate-400 group-open:rotate-180 transition-transform text-xs">▼</span>
            </summary>
            <div className="mt-3 text-xs text-slate-600 leading-relaxed border-t border-slate-200/80 pt-3 space-y-2">
              <p>
                Vision Transformers divide images into a 14×14 grid of 196 distinct patches. The Attention Rollout algorithm tracks how the neural network passes attention across all 12 transformer encoder blocks from the classification token (<code className="font-mono text-[11px] bg-slate-100 px-1 rounded">[CLS]</code>) to individual spatial image patches.
              </p>
              <p>
                Highlighted warmer zones (yellow and red) indicate the specific visual regions that the model prioritized when reaching its <strong>{normalizedPred}</strong> conclusion ({normalizedConf}% confidence).
              </p>
            </div>
          </details>

          {/* 2. Technical details */}
          <details className="group rounded-xl border border-slate-200 bg-white p-3.5 transition-colors open:bg-slate-50/50">
            <summary className="flex items-center justify-between cursor-pointer text-xs font-semibold text-slate-800 select-none">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-indigo-600 shrink-0" />
                <span>Technical details</span>
              </div>
              <span className="text-slate-400 group-open:rotate-180 transition-transform text-xs">▼</span>
            </summary>
            <div className="mt-3 text-xs font-mono text-slate-700 leading-relaxed border-t border-slate-200/80 pt-3 space-y-2">
              <div className="p-2.5 rounded bg-slate-900 text-slate-200 text-[11px] overflow-x-auto">
                {xaiData.technical_explanation || 'Transformer attention matrices propagated recursively through 12 Multi-Head Self-Attention layers with 5% noise pruning.'}
              </div>
              <div className="text-[11px] text-slate-500 space-y-1">
                <div>• Peak Attention Coordinate: X={peakCoords[0]}, Y={peakCoords[1]}</div>
                <div>• Primary Method: Attention Rollout (Abnar &amp; Zuidema, 2020)</div>
                <div>• Patch Grid: 14x14 tokens (196 total patches, 16x16 px patch size)</div>
                <div>• Model Architecture: google/vit-base-patch16-224</div>
              </div>
            </div>
          </details>

        </div>

        {/* Forensic Disclaimer */}
        <div className="p-3.5 rounded-xl bg-amber-50/90 border border-amber-200/90 text-amber-900 text-xs flex items-start gap-2.5 shadow-sm">
          <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
          <p className="leading-normal">
            <strong>Forensic Disclaimer:</strong> Highlighted regions indicate model attribution and should be interpreted as supporting evidence, not proof of manipulation.
          </p>
        </div>

      </div>

    </div>
  );
}
