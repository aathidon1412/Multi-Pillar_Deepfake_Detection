import React, { useState, useRef } from 'react';
import { Mic, Play, Pause, AlertTriangle, Sparkles, Volume2, Info, ChevronDown, ChevronUp, Radio, Activity } from 'lucide-react';
import { getMediaUrl } from '../services/api';

export default function Pillar3XaiAudioExplanation({
  xaiData,
  audioUrl,
  prediction = 'REAL',
  confidence = 95.0,
  audioMetadata = {}
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeSegmentId, setActiveSegmentId] = useState(null);
  const [showTechDetails, setShowTechDetails] = useState(false);
  const audioRef = useRef(null);

  if (!xaiData || xaiData.xai_available === false) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-3">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Mic className="w-5 h-5 text-cyan-600" />
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              Audio Explanation (Pillar 3: Acoustic Forensics)
            </h3>
          </div>
          <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-semibold">
            Integrated Gradients
          </span>
        </div>
        <p className="text-xs text-slate-500">
          Audio saliency explanation is currently being computed or not available for this sample.
        </p>
      </div>
    );
  }

  const importantSegments = xaiData.important_segments || [];
  const saliencyImgUrl = xaiData.saliency_image ? getMediaUrl(xaiData.saliency_image) : (xaiData.saliency_image_data || null);
  const isLocalized = prediction.includes('LOCALIZED') || audioMetadata?.is_localized_tamper;
  const isFake = isLocalized || prediction === 'FAKE' || prediction === 'AI_GENERATED' || prediction === 'AIVoice';
  const displayPrediction = isLocalized 
    ? 'LOCALIZED VOICE SPLICING DETECTED' 
    : (isFake ? 'AI SYNTHESIZED VOICE' : 'AUTHENTIC HUMAN VOICE');

  const handlePlaySegment = (seg) => {
    if (!audioRef.current) return;
    setActiveSegmentId(seg.segment_id);
    audioRef.current.currentTime = seg.start_time;
    audioRef.current.play();
    setIsPlaying(true);

    // Stop playback after segment duration
    const durationMs = (seg.end_time - seg.start_time) * 1000;
    setTimeout(() => {
      if (audioRef.current && !audioRef.current.paused) {
        audioRef.current.pause();
        setIsPlaying(false);
      }
    }, durationMs + 200);
  };

  const toggleFullPlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden transition-all space-y-0">
      
      {/* 1. SECTION HEADER */}
      <div className="p-5 sm:p-6 bg-slate-900 text-white flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Mic className="w-5 h-5 text-cyan-400" />
            <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-400 font-bold">
              Pillar 3 • Wav2Vec2 Integrated Gradients Saliency
            </span>
          </div>
          <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight">
            Why this audio prediction? (Audio Explanation)
          </h2>
          <p className="text-xs text-slate-300">
            {isLocalized 
              ? 'Warning: Localized acoustic anomalies identified. Certain time intervals show deepfake voice synthesis characteristics.'
              : 'Time-frequency saliency spectrogram pinpointing spectral regions and intervals that influenced the model.'}
          </p>
        </div>

        {/* Prediction Tag */}
        <div className="flex items-center gap-2 self-start sm:self-center">
          <span className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold border ${
            isLocalized
              ? 'bg-amber-950/90 border-amber-500/80 text-amber-300 shadow-amber-900/30'
              : (isFake 
                ? 'bg-rose-950/80 border-rose-600/60 text-rose-300' 
                : 'bg-emerald-950/80 border-emerald-600/60 text-emerald-300')
          }`}>
            {displayPrediction} • {confidence.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* 2. AUDIO PLAYER BAR */}
      {audioUrl && (
        <div className="p-4 bg-slate-800 border-b border-slate-700 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-white">
          <div className="flex items-center gap-3">
            <button
              onClick={toggleFullPlay}
              className="w-9 h-9 rounded-full bg-cyan-500 hover:bg-cyan-400 text-slate-900 flex items-center justify-center font-bold transition-transform active:scale-95 shrink-0 shadow-sm"
              title={isPlaying ? 'Pause Audio' : 'Play Audio'}
            >
              {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
            </button>
            <div className="flex flex-col">
              <span className="text-xs font-semibold text-slate-200">Audio Playback Stream</span>
              <span className="text-[11px] font-mono text-slate-400">
                16 kHz Mono • {xaiData.total_duration || audioMetadata.duration || '0.0'}s duration
              </span>
            </div>
          </div>

          <audio
            ref={audioRef}
            src={audioUrl}
            onEnded={() => setIsPlaying(false)}
            onPause={() => setIsPlaying(false)}
            onPlay={() => setIsPlaying(true)}
            className="w-full sm:w-64 h-8"
            controls
          />
        </div>
      )}

      {/* 3. TIME-FREQUENCY SPECTROGRAM SALIENCY MAP */}
      <div className="p-6 space-y-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-600" />
              Time-Frequency Saliency & Spectrogram Heatmap
            </h4>
            <span className="text-xs font-mono text-slate-500">
              Sundararajan Integrated Gradients (m=20 steps)
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Top panel displays raw waveform amplitude and gradient energy. Bottom panel shows Mel-frequency spectrogram (0–8000 Hz) with attribution density overlay.
          </p>

          {saliencyImgUrl ? (
            <div className="rounded-xl overflow-hidden border border-slate-300 bg-slate-950 shadow-inner">
              <img
                src={saliencyImgUrl}
                alt="Wav2Vec2 Integrated Gradients Saliency Spectrogram"
                className="w-full h-auto object-contain"
              />
            </div>
          ) : (
            <div className="p-8 rounded-xl bg-slate-100 text-slate-500 text-center text-xs font-mono">
              Spectrogram saliency visualization rendering...
            </div>
          )}
        </div>

        {/* 4. HIGHLIGHTED CONTRIBUTING SEGMENTS */}
        <div className="space-y-3 pt-2 border-t border-slate-200">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-slate-900">
              Highlighted Contributing Audio Segments ({importantSegments.length})
            </h4>
            <span className="text-xs text-slate-500">
              Click segment to play exact interval
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {importantSegments.map((seg) => {
              const isSegmentActive = activeSegmentId === seg.segment_id;
              const isFakeDir = seg.direction === 'supports_fake';
              return (
                <div
                  key={seg.segment_id}
                  onClick={() => handlePlaySegment(seg)}
                  className={`cursor-pointer rounded-xl p-3.5 border transition-all space-y-2 text-left group ${
                    isSegmentActive
                      ? 'border-cyan-500 bg-cyan-50/50 ring-2 ring-cyan-200 shadow-sm'
                      : 'border-slate-200 bg-slate-50 hover:bg-white hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-slate-900 text-cyan-300 text-[10px] font-bold font-mono flex items-center justify-center">
                        #{seg.segment_id}
                      </span>
                      <span className="font-mono text-xs font-bold text-slate-900">
                        {seg.start_time.toFixed(2)}s – {seg.end_time.toFixed(2)}s
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                        isFakeDir
                          ? 'bg-rose-100 text-rose-800'
                          : 'bg-emerald-100 text-emerald-800'
                      }`}>
                        {seg.attribution_percent}% ATTRIBUTION
                      </span>
                      <div className="w-6 h-6 rounded-full bg-slate-200 group-hover:bg-cyan-500 group-hover:text-white text-slate-700 flex items-center justify-center transition-colors">
                        <Play className="w-3 h-3 fill-current ml-0.5" />
                      </div>
                    </div>
                  </div>

                  {/* Magnitude Bar */}
                  <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        isFakeDir ? 'bg-rose-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.max(8, seg.attribution_percent)}%` }}
                    />
                  </div>

                  <p className="text-[11px] text-slate-600 leading-tight">
                    {seg.explanation}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* 5. HUMAN-READABLE EXPLANATION */}
        <div className="p-4 rounded-xl bg-cyan-50/60 border border-cyan-200 space-y-1.5">
          <div className="flex items-center gap-2 text-cyan-900 font-bold text-xs uppercase tracking-wider font-mono">
            <Sparkles className="w-4 h-4 text-cyan-700" />
            What influenced this prediction?
          </div>
          <p className="text-xs sm:text-sm text-slate-800 leading-relaxed font-medium">
            {xaiData.human_explanation}
          </p>
        </div>

        {/* 6. TECHNICAL DETAILS EXPANDER */}
        <div>
          <button
            onClick={() => setShowTechDetails(!showTechDetails)}
            className="flex items-center justify-between w-full py-2.5 px-4 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-mono text-slate-700 font-semibold transition-colors"
          >
            <span>🔬 Technical Integrated Gradients Telemetry</span>
            {showTechDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showTechDetails && (
            <div className="mt-2 p-4 bg-slate-900 text-slate-200 rounded-lg text-xs font-mono space-y-2 border border-slate-800 leading-relaxed">
              <div><strong>Technical Telemetry:</strong> {xaiData.technical_explanation}</div>
              <div><strong>Model Architecture:</strong> Wav2Vec2ForSequenceClassification (`Hemgg/Deepfake-audio-detection`)</div>
              <div><strong>Input Domain:</strong> Continuous 16kHz raw audio waveform (1D differentiable input path)</div>
              <div><strong>Attribution Baseline:</strong> Silent zero-waveform baseline tensor (x'=0)</div>
              <div><strong>Axioms Satisfied:</strong> Completeness, Implementation Invariance, Linearity (Sundararajan et al., 2017)</div>
            </div>
          )}
        </div>

        {/* 7. MANDATORY NON-CAUSAL FORENSIC DISCLAIMER */}
        <div className="p-3 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg text-xs text-amber-800 leading-relaxed">
          ⚠️ <strong>Forensic Attribution Notice:</strong> Highlighted audio segments and spectrogram saliency reflect neural model attribution and internal feature importance; they should be interpreted as supporting evidence, not independent physical proof of synthesis.
        </div>

      </div>

    </div>
  );
}
