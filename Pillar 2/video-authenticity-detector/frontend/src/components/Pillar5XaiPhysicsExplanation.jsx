import React, { useState } from 'react';
import { 
  Activity, 
  HelpCircle, 
  TrendingUp, 
  TrendingDown, 
  Layers, 
  Sparkles, 
  ChevronDown, 
  ChevronUp, 
  ShieldAlert, 
  CheckCircle2, 
  AlertTriangle,
  Info,
  Sliders,
  Maximize2,
  BarChart2
} from 'lucide-react';
import { getMediaUrl } from '../services/api';

export default function Pillar5XaiPhysicsExplanation({
  xaiData,
  prediction = 'AUTHENTIC',
  confidence = 85.0
}) {
  const [showTechDetails, setShowTechDetails] = useState(false);
  const [activeTab, setActiveTab] = useState('handcrafted'); // 'handcrafted' | 'all' | 'pos_neg'
  const [showTooltip, setShowTooltip] = useState(false);
  const [showFullPlotModal, setShowFullPlotModal] = useState(false);

  const isAvailable = xaiData && xaiData.xai_available !== false;

  const displayPred = String(
    xaiData?.prediction || prediction || 'AUTHENTIC'
  ).toUpperCase();
  
  const isFake = displayPred.includes('FAKE') || displayPred.includes('SYNTHETIC') || displayPred.includes('MANIPULATED');
  const confPct = xaiData?.confidence !== undefined
    ? Number(xaiData.confidence).toFixed(1)
    : (typeof confidence === 'number' ? confidence.toFixed(1) : '85.0');

  if (!isAvailable) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        <div className="p-5 bg-slate-900 text-white flex items-center justify-between border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-400" />
            <span className="text-[11px] font-mono uppercase tracking-widest text-indigo-400 font-bold">
              Pillar 5 • Physical Geometry & Steganalysis XAI
            </span>
          </div>
          <span className="text-xs font-mono px-3 py-1 rounded bg-slate-800 text-slate-400 border border-slate-700">
            SHAP UNAVAILABLE
          </span>
        </div>
        <div className="p-6 text-sm text-slate-600 bg-slate-50">
          <p>TreeSHAP explanation is currently unavailable for this sample. The ensemble prediction remains intact.</p>
        </div>
      </div>
    );
  }

  const waterfallUrl = xaiData.waterfall_plot_url ? getMediaUrl(xaiData.waterfall_plot_url) : null;
  const waterfallBase64 = xaiData.waterfall_plot_base64 ? `data:image/png;base64,${xaiData.waterfall_plot_base64}` : null;
  const plotSrc = waterfallBase64 || waterfallUrl;

  const handcraftedFeatures = xaiData.handcrafted_forensic_features || [];
  const topAllFeatures = xaiData.feature_importance_ranking || [];
  const topPositive = xaiData.top_positive_contributors || [];
  const topNegative = xaiData.top_negative_contributors || [];

  const baseVal = xaiData.base_value !== undefined ? Number(xaiData.base_value).toFixed(3) : '0.000';
  const outVal = xaiData.model_output_margin !== undefined ? Number(xaiData.model_output_margin).toFixed(3) : '0.000';
  const modelName = xaiData.shap_model_used || 'XGBoost Primary Tree Estimator';

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden transition-all">
      {/* Header */}
      <div className="p-5 bg-slate-900 text-white flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" />
            <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-400 font-bold">
              Pillar 5 • Physics & Steganalysis TreeSHAP
            </span>
          </div>
          <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
            Why did the physical-forensics model make this prediction?
            <div className="relative inline-block">
              <button 
                type="button"
                aria-label="Explain SHAP attribution"
                className="text-slate-400 hover:text-white transition-colors cursor-pointer"
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                onClick={() => setShowTooltip(!showTooltip)}
              >
                <HelpCircle className="w-4 h-4" />
              </button>
              {showTooltip && (
                <div className="absolute left-0 top-6 z-50 w-72 p-3 bg-slate-800 text-white text-xs rounded-xl shadow-xl border border-slate-700 leading-relaxed pointer-events-none">
                  <div className="font-bold text-cyan-300 mb-1">SHAP Direction Guide:</div>
                  <div className="space-y-1 font-mono text-[11px]">
                    <p className="text-rose-300"><strong className="text-rose-400">Positive (+)</strong> → pushes prediction toward Fake / Synthetic.</p>
                    <p className="text-emerald-300"><strong className="text-emerald-400">Negative (−)</strong> → pushes prediction away from Fake (toward Real / Authentic).</p>
                  </div>
                </div>
              )}
            </div>
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-mono px-3 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            Model: {modelName}
          </span>
          <span className={`text-xs font-mono px-3 py-1 rounded font-bold border flex items-center gap-1.5 ${
            isFake 
              ? 'bg-rose-950/80 text-rose-300 border-rose-800/80' 
              : 'bg-emerald-950/80 text-emerald-300 border-emerald-800/80'
          }`}>
            {isFake ? <ShieldAlert className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
            {displayPred} ({confPct}%)
          </span>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Dynamic Human-Readable Explanation Banner */}
        <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-50 via-slate-50 to-blue-50 border border-cyan-200/80 shadow-xs flex items-start gap-3.5">
          <Sparkles className="w-5 h-5 text-cyan-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h3 className="text-xs font-bold font-mono tracking-wider uppercase text-cyan-900">
              Forensic Attribution Summary
            </h3>
            <p className="text-slate-800 text-sm leading-relaxed font-medium">
              {xaiData.human_explanation}
            </p>
          </div>
        </div>

        {/* SHAP Tooltip Notice Callout */}
        <div className="p-3 bg-slate-100 rounded-lg border border-slate-200 text-xs text-slate-600 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-cyan-600 shrink-0" />
            <span>
              <strong>Attribution Convention:</strong> Positive contribution (+) pushes toward Fake/Synthetic; Negative contribution (−) pushes toward Authentic/Real.
            </span>
          </div>
          <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 bg-white px-2 py-0.5 rounded border">
            TreeSHAP Additive
          </span>
        </div>

        {/* Main Grid: Waterfall Plot & Feature Breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Waterfall Plot (7 cols) */}
          <div className="lg:col-span-7 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-slate-700 flex items-center gap-1.5">
                <BarChart2 className="w-4 h-4 text-indigo-600" />
                SHAP Waterfall Plot (Marginal Decision Flow)
              </h3>
              {plotSrc && (
                <button
                  type="button"
                  onClick={() => setShowFullPlotModal(true)}
                  className="text-xs font-mono text-cyan-700 hover:text-cyan-900 flex items-center gap-1 bg-cyan-50 hover:bg-cyan-100 px-2.5 py-1 rounded transition-colors"
                >
                  <Maximize2 className="w-3 h-3" />
                  Expand
                </button>
              )}
            </div>

            {plotSrc ? (
              <div className="rounded-xl border border-slate-800 overflow-hidden bg-slate-950 shadow-inner group relative">
                <img 
                  src={plotSrc} 
                  alt="Pillar 5 SHAP Waterfall Plot" 
                  className="w-full h-auto object-contain cursor-pointer transition-transform duration-200 hover:scale-[1.01]"
                  onClick={() => setShowFullPlotModal(true)}
                />
                <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/70 text-white text-[10px] font-mono px-2 py-1 rounded backdrop-blur-xs">
                  Click to inspect full resolution
                </div>
              </div>
            ) : (
              <div className="h-64 rounded-xl border border-dashed border-slate-300 flex items-center justify-center text-slate-400 text-xs font-mono bg-slate-50">
                Waterfall plot visualization not available for this sample
              </div>
            )}

            {/* Model telemetry stats */}
            <div className="grid grid-cols-3 gap-2 pt-1">
              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-center">
                <div className="text-[10px] font-mono uppercase text-slate-500">Base Value E[f(x)]</div>
                <div className="text-sm font-bold font-mono text-slate-800">{baseVal}</div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-center">
                <div className="text-[10px] font-mono uppercase text-slate-500">Output Margin f(x)</div>
                <div className={`text-sm font-bold font-mono ${Number(outVal) > 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {Number(outVal) > 0 ? `+${outVal}` : outVal}
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-center">
                <div className="text-[10px] font-mono uppercase text-slate-500">Active Features</div>
                <div className="text-sm font-bold font-mono text-slate-800">
                  {xaiData.total_selected_features || 160}
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Feature Explanations (5 cols) */}
          <div className="lg:col-span-5 space-y-3">
            {/* Tab Selector */}
            <div className="flex rounded-lg bg-slate-100 p-1 border border-slate-200">
              <button
                type="button"
                onClick={() => setActiveTab('handcrafted')}
                className={`flex-1 py-1.5 px-2 rounded-md text-xs font-medium transition-all ${
                  activeTab === 'handcrafted'
                    ? 'bg-white text-slate-900 shadow-xs font-bold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Physics / SRM Features
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('pos_neg')}
                className={`flex-1 py-1.5 px-2 rounded-md text-xs font-medium transition-all ${
                  activeTab === 'pos_neg'
                    ? 'bg-white text-slate-900 shadow-xs font-bold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Top Pushers (+/−)
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('all')}
                className={`flex-1 py-1.5 px-2 rounded-md text-xs font-medium transition-all ${
                  activeTab === 'all'
                    ? 'bg-white text-slate-900 shadow-xs font-bold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                All Ranked (SHAP)
              </button>
            </div>

            {/* TAB 1: Handcrafted Forensic Features */}
            {activeTab === 'handcrafted' && (
              <div className="space-y-2.5 max-h-[460px] overflow-y-auto pr-1">
                <div className="text-[11px] text-slate-500 px-1">
                  Human-interpretable physical, geometric & steganalysis indicators (ELA, SRM, Laplacian & vanishing geometry):
                </div>

                {handcraftedFeatures.length > 0 ? (
                  handcraftedFeatures.map((feat, idx) => {
                    const isFakePusher = feat.direction === 'supports_fake';
                    const shapValNum = Number(feat.shap_value);
                    const formattedShap = isNaN(shapValNum) ? '0.000' : (shapValNum > 0 ? `+${shapValNum.toFixed(3)}` : shapValNum.toFixed(3));
                    const valNum = Number(feat.value);
                    const formattedVal = isNaN(valNum) ? String(feat.value) : valNum.toFixed(4);

                    return (
                      <div 
                        key={`handcrafted-${feat.feature || idx}`}
                        className="p-3 rounded-xl border border-slate-200 bg-white hover:border-slate-300 transition-all space-y-2 shadow-2xs"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="font-bold text-xs text-slate-900 font-mono flex items-center gap-1.5">
                              {feat.feature}
                              <span className="text-[10px] font-sans px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 font-normal border">
                                {feat.category || 'Forensic'}
                              </span>
                            </div>
                            <div className="text-[11px] text-slate-500 leading-snug mt-0.5">
                              {feat.human_description}
                            </div>
                          </div>
                          
                          <div className={`shrink-0 px-2 py-1 rounded text-xs font-mono font-bold border flex items-center gap-1 ${
                            isFakePusher 
                              ? 'bg-rose-50 text-rose-700 border-rose-200' 
                              : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          }`}>
                            {isFakePusher ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                            {formattedShap}
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-[11px] font-mono text-slate-600 pt-1 border-t border-slate-100">
                          <span>Raw Value: <strong className="text-slate-800">{formattedVal}</strong></span>
                          <span className={isFakePusher ? 'text-rose-600 font-semibold' : 'text-emerald-600 font-semibold'}>
                            {isFakePusher ? 'Pushes toward FAKE' : 'Pushes toward REAL'}
                          </span>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="p-4 rounded-xl bg-slate-50 border text-center text-xs text-slate-500 font-mono">
                    No handcrafted features in top active indices
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: Top Positive and Negative Contributors */}
            {activeTab === 'pos_neg' && (
              <div className="space-y-4 max-h-[460px] overflow-y-auto pr-1">
                {/* Positive (Supports Fake) */}
                <div className="space-y-2">
                  <div className="text-xs font-bold font-mono text-rose-700 flex items-center gap-1 px-1">
                    <TrendingUp className="w-3.5 h-3.5" />
                    Top Pushing Toward FAKE / SYNTHETIC
                  </div>
                  {topPositive.map((feat, idx) => (
                    <div key={`pos-${idx}`} className="p-2.5 rounded-lg border border-rose-100 bg-rose-50/50 flex items-center justify-between text-xs">
                      <div>
                        <div className="font-bold font-mono text-rose-950">{feat.feature}</div>
                        <div className="text-[11px] text-rose-700/80">Val: {Number(feat.value).toFixed(3)}</div>
                      </div>
                      <div className="font-mono font-bold text-rose-700 bg-white px-2 py-0.5 rounded border border-rose-200">
                        +{Number(feat.shap_value).toFixed(3)}
                      </div>
                    </div>
                  ))}
                  {topPositive.length === 0 && (
                    <div className="text-xs text-slate-400 font-mono p-2">None in this direction</div>
                  )}
                </div>

                {/* Negative (Supports Real) */}
                <div className="space-y-2">
                  <div className="text-xs font-bold font-mono text-emerald-700 flex items-center gap-1 px-1">
                    <TrendingDown className="w-3.5 h-3.5" />
                    Top Pushing Toward REAL / AUTHENTIC
                  </div>
                  {topNegative.map((feat, idx) => (
                    <div key={`neg-${idx}`} className="p-2.5 rounded-lg border border-emerald-100 bg-emerald-50/50 flex items-center justify-between text-xs">
                      <div>
                        <div className="font-bold font-mono text-emerald-950">{feat.feature}</div>
                        <div className="text-[11px] text-emerald-700/80">Val: {Number(feat.value).toFixed(3)}</div>
                      </div>
                      <div className="font-mono font-bold text-emerald-700 bg-white px-2 py-0.5 rounded border border-emerald-200">
                        {Number(feat.shap_value).toFixed(3)}
                      </div>
                    </div>
                  ))}
                  {topNegative.length === 0 && (
                    <div className="text-xs text-slate-400 font-mono p-2">None in this direction</div>
                  )}
                </div>
              </div>
            )}

            {/* TAB 3: All Features Ranking */}
            {activeTab === 'all' && (
              <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1">
                <div className="text-[11px] text-slate-500 px-1">
                  Absolute SHAP impact across the 160 active pipeline features:
                </div>
                {topAllFeatures.slice(0, 10).map((feat, idx) => {
                  const isFakePusher = feat.direction === 'supports_fake';
                  const isHandcrafted = feat.is_handcrafted;
                  return (
                    <div 
                      key={`all-${idx}`}
                      className="p-2.5 rounded-lg border border-slate-200 bg-white flex items-center justify-between text-xs"
                    >
                      <div className="space-y-0.5">
                        <div className="font-bold font-mono text-slate-900 flex items-center gap-1.5">
                          <span className="text-slate-400 text-[10px]">#{idx + 1}</span>
                          {feat.feature}
                          {isHandcrafted && (
                            <span className="text-[9px] font-sans px-1 rounded bg-cyan-100 text-cyan-800 font-semibold">
                              PHYSICS
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-slate-500 font-mono">
                          Val: {Number(feat.value).toFixed(3)}
                        </div>
                      </div>
                      <div className={`font-mono font-bold px-2 py-0.5 rounded border text-xs ${
                        isFakePusher ? 'bg-rose-50 text-rose-700 border-rose-200' : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      }`}>
                        {Number(feat.shap_value) > 0 ? `+${Number(feat.shap_value).toFixed(3)}` : Number(feat.shap_value).toFixed(3)}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Technical Expandable Details */}
        <div className="rounded-xl border border-slate-200 bg-slate-50/70 overflow-hidden">
          <button
            type="button"
            onClick={() => setShowTechDetails(!showTechDetails)}
            className="w-full p-3.5 flex items-center justify-between text-left hover:bg-slate-100/80 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-slate-600" />
              <span className="text-xs font-bold font-mono uppercase tracking-wider text-slate-700">
                Technical SHAP Attribution & Model Telemetry
              </span>
            </div>
            {showTechDetails ? (
              <ChevronUp className="w-4 h-4 text-slate-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-500" />
            )}
          </button>

          {showTechDetails && (
            <div className="p-4 border-t border-slate-200 space-y-3 text-xs bg-white">
              <p className="text-slate-700 leading-relaxed">
                {xaiData.technical_explanation}
              </p>

              <div className="p-3 bg-slate-900 rounded-lg text-slate-300 font-mono text-[11px] space-y-1">
                <div><strong>Pipeline:</strong> 24 Handcrafted Features + 1,280 EfficientNet Embeddings → StandardScaler → SelectFromModel (160) → Soft-Voting Ensemble</div>
                <div><strong>TreeSHAP Target:</strong> {modelName} (Exact marginal tree attribution)</div>
                <div><strong>Base Expected Value E[f(x)]:</strong> {baseVal}</div>
                <div><strong>Current Prediction Margin f(x):</strong> {outVal}</div>
                <div><strong>Additive Fidelity:</strong> f(x) = E[f(x)] + Σ φᵢ</div>
              </div>

              {xaiData.limitations && xaiData.limitations.length > 0 && (
                <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-[11px] space-y-1">
                  <div className="font-bold flex items-center gap-1 text-amber-950">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                    Forensic Scope & Limitations:
                  </div>
                  <ul className="list-disc list-inside space-y-0.5 text-amber-800">
                    {xaiData.limitations.map((lim, i) => (
                      <li key={i}>{lim}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Full Resolution Waterfall Plot Modal */}
      {showFullPlotModal && plotSrc && (
        <div 
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 backdrop-blur-xs"
          onClick={() => setShowFullPlotModal(false)}
        >
          <div 
            className="max-w-4xl w-full bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl space-y-3 p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between text-white border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <span className="font-mono text-xs uppercase font-bold text-cyan-400">
                  SHAP Waterfall Visualization — High Resolution
                </span>
              </div>
              <button
                type="button"
                onClick={() => setShowFullPlotModal(false)}
                className="text-slate-400 hover:text-white text-xs font-mono px-2 py-1 rounded bg-slate-800"
              >
                Close (ESC)
              </button>
            </div>
            <div className="flex justify-center bg-slate-900 rounded-xl p-2">
              <img 
                src={plotSrc} 
                alt="High Resolution SHAP Waterfall" 
                className="max-h-[80vh] w-auto object-contain rounded-lg"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
