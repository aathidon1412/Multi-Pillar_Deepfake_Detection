import React, { useState } from 'react';
import {
  Brain,
  Eye,
  Clock,
  Mic,
  FileText,
  Activity,
  Layers,
  Info,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Shield,
  ShieldCheck,
  ShieldAlert,
  BarChart3,
  Sliders,
  Play,
  Pause,
  Maximize2
} from 'lucide-react';
import { getMediaUrl } from '../services/api';

import Pillar1XaiExplanation from './Pillar1XaiExplanation';
import Pillar2XaiTemporalExplanation from './Pillar2XaiTemporalExplanation';
import Pillar3XaiAudioExplanation from './Pillar3XaiAudioExplanation';
import Pillar4XaiDocumentExplanation from './Pillar4XaiDocumentExplanation';
import Pillar5XaiPhysicsExplanation from './Pillar5XaiPhysicsExplanation';

/**
 * Unified Section: "WHY THIS PREDICTION?"
 * 
 * Provides an integrated, end-to-end Explainable AI (XAI) dossier across all 5 analytical pillars:
 * 1. WHAT the system predicted (verdict & confidence summary)
 * 2. EVIDENCE supporting the prediction (Per-pillar actual evidentiary findings)
 * 3. HOW the evidence was calculated (Technical telemetry & collapsible math)
 * 4. LIMITATIONS & GUIDELINES ("HOW TO INTERPRET THIS EVIDENCE")
 */
export default function WhyThisPredictionSection({
  report,
  seekTime,
  onSeek,
  currentTime = 0,
  originalMediaUrl = null
}) {
  const [activePillarTab, setActivePillarTab] = useState('all'); // 'all' | 'p1' | 'p2' | 'p3' | 'p4' | 'p5'
  const [showInterpretationHelp, setShowInterpretationHelp] = useState(false);

  if (!report) return null;

  const modality = report.modality || (report.video_details?.duration_seconds ? 'video' : 'image');
  const xaiBundle = report.xai || {};

  // Extract per-pillar XAI objects with safe fallback support
  const p1Xai = report.pillar1?.xai || xaiBundle.pillar1 || (xaiBundle.highest_attribution_region ? xaiBundle : null);
  const p2Xai = report.pillar2?.xai || xaiBundle.pillar2 || (xaiBundle.temporal_segments ? xaiBundle : null);
  const p3Xai = report.pillar3?.xai || xaiBundle.pillar3 || (xaiBundle.important_segments || xaiBundle.saliency_image ? xaiBundle : null);
  const p4Xai = report.pillar4?.xai || xaiBundle.pillar4 || (xaiBundle.comparison_table || xaiBundle.top_deviations ? xaiBundle : null);
  const p5Xai = report.pillar5?.xai || xaiBundle.pillar5 || (xaiBundle.waterfall_plot_url || xaiBundle.waterfall_plot_base64 ? xaiBundle : null);

  // Determine which pillars are active for this media item
  const hasP1 = Boolean(p1Xai && p1Xai.xai_available !== false) || Boolean(report.pillar1);
  const hasP2 = (modality === 'video') || Boolean(p2Xai && p2Xai.xai_available !== false) || Boolean(report.analysis?.temporal);
  const hasP3 = (modality === 'audio') || Boolean(p3Xai && p3Xai.xai_available !== false) || Boolean(report.pillar3);
  const hasP4 = false; // Disabled globally in UI
  const hasP5 = Boolean(p5Xai && p5Xai.xai_available !== false) || Boolean(report.pillar5);

  const overallSummary = xaiBundle.overall_summary || (
    report.consensus?.verdict 
      ? `Consensus finding: ${report.consensus.verdict} (${report.consensus.confidence || 85}% confidence). Forensic evidence combined across active analytical channels.`
      : 'Explainable AI telemetry extracted across active forensic neural and statistical models.'
  );

  return (
    <div id="why-this-prediction-container" className="w-full flex flex-col gap-6 mt-4">
      
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          TOP BANNER: WHY THIS PREDICTION?
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 sm:p-8 text-white shadow-xl relative overflow-hidden">
        {/* Subtle background glow effect */}
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2 max-w-3xl">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 font-mono text-xs font-bold uppercase tracking-wider">
                <Brain className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                Explainable AI (XAI) Forensic Engine
              </span>
              <span className="text-slate-500 hidden sm:inline">•</span>
              <span className="text-xs font-mono text-slate-400 uppercase hidden sm:inline">
                Evidence Attribution Dossier
              </span>
            </div>

            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              WHY THIS PREDICTION?
            </h2>

            <p className="text-sm text-slate-300 leading-relaxed">
              {overallSummary}
            </p>
          </div>

          {/* Visual Evidence Category Badges */}
          <div className="flex flex-wrap gap-2 lg:max-w-xs self-start lg:self-center">
            <span className="px-2.5 py-1 rounded-lg bg-cyan-950/70 border border-cyan-500/40 font-mono text-[11px] font-bold text-cyan-300 flex items-center gap-1.5" title="Neural attention rollout & time-frequency saliency">
              <Eye className="w-3 h-3 text-cyan-400" />
              MODEL ATTRIBUTION
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-pink-950/70 border border-pink-500/40 font-mono text-[11px] font-bold text-pink-300 flex items-center gap-1.5" title="Optical flow, flickering & temporal timeline markers">
              <Clock className="w-3 h-3 text-pink-400" />
              TEMPORAL EVIDENCE
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-amber-950/70 border border-amber-500/40 font-mono text-[11px] font-bold text-amber-300 flex items-center gap-1.5" title="Benford's Law OCR first-digit statistical divergence">
              <BarChart3 className="w-3 h-3 text-amber-400" />
              STATISTICAL EVIDENCE
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-indigo-950/70 border border-indigo-500/40 font-mono text-[11px] font-bold text-indigo-300 flex items-center gap-1.5" title="TreeSHAP additive margin feature attribution">
              <Activity className="w-3 h-3 text-indigo-400" />
              FEATURE CONTRIBUTION
            </span>
          </div>
        </div>

        {/* Pillar Filter Navigation Pills */}
        <div className="mt-6 pt-5 border-t border-slate-800 flex items-center gap-2 overflow-x-auto pb-1">
          <span className="text-xs font-mono text-slate-400 uppercase font-semibold mr-1 shrink-0">
            Filter View:
          </span>
          <button
            type="button"
            onClick={() => setActivePillarTab('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-all shrink-0 ${
              activePillarTab === 'all'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            All Active Evidence
          </button>
          
          {hasP1 && (
            <button
              type="button"
              onClick={() => setActivePillarTab('p1')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-all shrink-0 ${
                activePillarTab === 'p1'
                  ? 'bg-cyan-500 text-slate-950 shadow-sm font-bold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Pillar 1: Visual (ViT)
            </button>
          )}

          {hasP2 && (
            <button
              type="button"
              onClick={() => setActivePillarTab('p2')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-all shrink-0 ${
                activePillarTab === 'p2'
                  ? 'bg-pink-500 text-white shadow-sm font-bold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Pillar 2: Temporal
            </button>
          )}

          {hasP3 && (
            <button
              type="button"
              onClick={() => setActivePillarTab('p3')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-all shrink-0 ${
                activePillarTab === 'p3'
                  ? 'bg-purple-500 text-white shadow-sm font-bold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Pillar 3: Audio Saliency
            </button>
          )}

          {hasP4 && (
            <button
              type="button"
              onClick={() => setActivePillarTab('p4')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-all shrink-0 ${
                activePillarTab === 'p4'
                  ? 'bg-amber-500 text-slate-950 shadow-sm font-bold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Pillar 4: Document Benford
            </button>
          )}

          {hasP5 && (
            <button
              type="button"
              onClick={() => setActivePillarTab('p5')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-all shrink-0 ${
                activePillarTab === 'p5'
                  ? 'bg-indigo-500 text-white shadow-sm font-bold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Pillar 5: Physics SHAP
            </button>
          )}
        </div>
      </div>


      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          PILLAR 1 — VISUAL EVIDENCE
          [Original Image + XAI Heatmap]
          "Which regions influenced the prediction?"
          [Human explanation]
          [Technical details ▼]
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {(activePillarTab === 'all' || activePillarTab === 'p1') && hasP1 && (
        <div id="pillar-1-visual-evidence" className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500" />
              <h3 className="text-sm font-bold font-mono tracking-wider uppercase text-slate-800">
                PILLAR 1 — VISUAL EVIDENCE
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded bg-cyan-50 border border-cyan-200 text-cyan-800 text-[11px] font-mono font-bold">
              MODEL ATTRIBUTION
            </span>
          </div>

          <Pillar1XaiExplanation
            xaiData={p1Xai || report.pillar1?.xai || xaiBundle.pillar1}
            originalImageSrc={originalMediaUrl || (report.file?.stored_filename ? getMediaUrl(`uploads/${report.file.stored_filename}`) : null)}
            prediction={report.pillar1?.verdict || report.classification?.prediction || 'REAL'}
            confidence={report.pillar1?.confidence || (report.classification?.confidence ? report.classification.confidence * 100 : 85.0)}
          />
        </div>
      )}


      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          PILLAR 2 — TEMPORAL EVIDENCE
          [Video Timeline]
          Suspicious timestamp markers
          [Selected Keyframe]
          "What happened here?"
          [Signal contribution chart]
          [Human explanation]
          [Technical details ▼]
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {(activePillarTab === 'all' || activePillarTab === 'p2') && hasP2 && (
        <div id="pillar-2-temporal-evidence" className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-pink-500" />
              <h3 className="text-sm font-bold font-mono tracking-wider uppercase text-slate-800">
                PILLAR 2 — TEMPORAL EVIDENCE
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded bg-pink-50 border border-pink-200 text-pink-800 text-[11px] font-mono font-bold">
              TEMPORAL EVIDENCE
            </span>
          </div>

          <Pillar2XaiTemporalExplanation
            xaiData={p2Xai || report.pillar2?.xai || xaiBundle.pillar2}
            suspiciousFrames={report.suspicious_frames || []}
            analysis={report.analysis || {}}
            classification={report.classification || {}}
            duration={report.video_details?.duration_seconds || 30}
            currentTime={currentTime}
            onSeek={onSeek}
            selectedTimestamp={seekTime}
          />
        </div>
      )}


      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          PILLAR 3 — AUDIO EVIDENCE
          [Audio/Spectrogram Saliency]
          "Which audio segments influenced the prediction?"
          [Highlighted segments]
          [Human explanation]
          [Technical details ▼]
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {(activePillarTab === 'all' || activePillarTab === 'p3') && hasP3 && (
        <div id="pillar-3-audio-evidence" className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-500" />
              <h3 className="text-sm font-bold font-mono tracking-wider uppercase text-slate-800">
                PILLAR 3 — AUDIO EVIDENCE
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded bg-purple-50 border border-purple-200 text-purple-800 text-[11px] font-mono font-bold">
              MODEL ATTRIBUTION
            </span>
          </div>

          <Pillar3XaiAudioExplanation
            xaiData={p3Xai || report.pillar3?.xai || xaiBundle.pillar3}
            audioUrl={report.file?.stored_filename ? getMediaUrl(`uploads/${report.file.stored_filename}`) : originalMediaUrl}
            prediction={report.pillar3?.prediction || report.classification?.prediction || 'REAL'}
            confidence={Number(report.pillar3?.confidence || (report.classification?.confidence ? report.classification.confidence * 100 : 95.0))}
            audioMetadata={report.pillar3 || {}}
          />
        </div>
      )}


      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          PILLAR 4 — DOCUMENT EVIDENCE
          [Benford Observed vs Expected Chart]
          [MAE] [Chi-Square] [p-value]
          "Which digits contributed to the deviation?"
          [Human explanation]
          [Technical details ▼]
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {(activePillarTab === 'all' || activePillarTab === 'p4') && hasP4 && (
        <div id="pillar-4-document-evidence" className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              <h3 className="text-sm font-bold font-mono tracking-wider uppercase text-slate-800">
                PILLAR 4 — DOCUMENT EVIDENCE
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded bg-amber-50 border border-amber-200 text-amber-800 text-[11px] font-mono font-bold">
              STATISTICAL EVIDENCE
            </span>
          </div>

          <Pillar4XaiDocumentExplanation
            xaiData={p4Xai || report.pillar4?.xai || xaiBundle.pillar4}
            prediction={report.pillar4?.verdict || report.classification?.prediction || 'AUTHENTIC DOCUMENT'}
            confidence={Number(report.pillar4?.confidence || 85.0)}
            digitsCount={report.pillar4?.digits_count || 0}
          />
        </div>
      )}


      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          PILLAR 5 — PHYSICAL FORENSIC EVIDENCE
          [SHAP Waterfall]
          Top contributing features
          [Human explanation]
          [Technical details ▼]
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {(activePillarTab === 'all' || activePillarTab === 'p5') && hasP5 && (
        <div id="pillar-5-physical-evidence" className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-indigo-500" />
              <h3 className="text-sm font-bold font-mono tracking-wider uppercase text-slate-800">
                PILLAR 5 — PHYSICAL FORENSIC EVIDENCE
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded bg-indigo-50 border border-indigo-200 text-indigo-800 text-[11px] font-mono font-bold">
              FEATURE CONTRIBUTION
            </span>
          </div>

          <Pillar5XaiPhysicsExplanation
            xaiData={p5Xai || report.pillar5?.xai || xaiBundle.pillar5}
            prediction={report.pillar5?.verdict || 'AUTHENTIC'}
            confidence={Number(report.pillar5?.confidence || 85.0)}
          />
        </div>
      )}


      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          FINAL SECTION: HOW TO INTERPRET THIS EVIDENCE
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-6 sm:p-7 shadow-sm space-y-5">
        <div className="flex items-center justify-between border-b border-slate-200 pb-3">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="w-5 h-5 text-slate-800" />
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              HOW TO INTERPRET THIS EVIDENCE
            </h3>
          </div>
          <span className="text-[11px] font-mono font-semibold px-2.5 py-0.5 rounded bg-white border border-slate-200 text-slate-600">
            Forensic Best Practices
          </span>
        </div>

        {/* 4 Core Interpretative Principles */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          
          <div className="p-4 rounded-xl bg-white border border-slate-200 flex items-start gap-3 shadow-xs">
            <div className="w-7 h-7 rounded-lg bg-cyan-50 border border-cyan-200 flex items-center justify-center text-cyan-700 font-mono font-bold text-xs shrink-0">
              1
            </div>
            <div className="space-y-1">
              <h4 className="text-xs font-bold text-slate-900 uppercase font-mono tracking-wide">
                Model Attribution vs. Proof
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                Explainable AI (XAI) highlights <strong>what influenced the model or statistical analysis</strong>. It visualizes neural activations and mathematical deviations, but does not independently constitute absolute legal proof of manipulation.
              </p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-white border border-slate-200 flex items-start gap-3 shadow-xs">
            <div className="w-7 h-7 rounded-lg bg-pink-50 border border-pink-200 flex items-center justify-center text-pink-700 font-mono font-bold text-xs shrink-0">
              2
            </div>
            <div className="space-y-1">
              <h4 className="text-xs font-bold text-slate-900 uppercase font-mono tracking-wide">
                Multi-Signal Corroboration
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                <strong>Multiple forensic signals should be considered together</strong>. High confidence arises when neural patch artifacts (P1), temporal flow jumps (P2), acoustic vocoders (P3), OCR distributions (P4), and shadow geometry (P5) converge on the same conclusion.
              </p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-white border border-slate-200 flex items-start gap-3 shadow-xs">
            <div className="w-7 h-7 rounded-lg bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-700 font-mono font-bold text-xs shrink-0">
              3
            </div>
            <div className="space-y-1">
              <h4 className="text-xs font-bold text-slate-900 uppercase font-mono tracking-wide">
                Supporting Evidence, Not Guarantees
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                A highlighted region, waveform interval, or tabular digit is <strong>supporting evidence</strong>. High contrast edges, natural film grain, reverberation, or low-resolution compression may create localized false attributions without media forgery.
              </p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-white border border-slate-200 flex items-start gap-3 shadow-xs">
            <div className="w-7 h-7 rounded-lg bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-700 font-mono font-bold text-xs shrink-0">
              4
            </div>
            <div className="space-y-1">
              <h4 className="text-xs font-bold text-slate-900 uppercase font-mono tracking-wide">
                Deterministic Physical Overrides
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                When authoritative physical signals exist (e.g. valid camera sensor hardware EXIF or strict shadow vanishing-point inliers), they supersede purely neural soft probabilities to protect against out-of-distribution deep learning hallucinations.
              </p>
            </div>
          </div>

        </div>

        {/* Visual Labels Legend Grid */}
        <div className="pt-2 border-t border-slate-200/60">
          <div className="text-[11px] font-mono text-slate-500 uppercase font-bold mb-2">
            Forensic Evidence Categorization Taxonomy:
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-cyan-100 text-cyan-800">
                MODEL ATTRIBUTION
              </span>
              <p className="text-[11px] text-slate-600 mt-1">
                ViT attention rollouts & Wav2Vec2 spectrogram gradients driving neural logits.
              </p>
            </div>
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-pink-100 text-pink-800">
                TEMPORAL EVIDENCE
              </span>
              <p className="text-[11px] text-slate-600 mt-1">
                Inter-frame motion jumps, boundary seams, and audio-visual synchronization across time.
              </p>
            </div>
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-amber-100 text-amber-800">
                STATISTICAL EVIDENCE
              </span>
              <p className="text-[11px] text-slate-600 mt-1">
                Benford first-digit logarithmic decay, MAE, and Chi-Square goodness-of-fit.
              </p>
            </div>
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-800">
                FEATURE CONTRIBUTION
              </span>
              <p className="text-[11px] text-slate-600 mt-1">
                TreeSHAP additive margin decomposition of handcrafted physical geometry & sensor noise.
              </p>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}
