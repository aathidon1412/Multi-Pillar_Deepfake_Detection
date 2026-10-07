import React, { useState, useEffect } from 'react';
import { 
  History as HistoryIcon, Eye, Trash2, Search, Filter, ShieldCheck, 
  Cpu, Scissors, RefreshCw, ChevronRight, Image as ImageIcon, 
  FileVideo, Mic, FileText, X, Sparkles, CheckCircle2, AlertTriangle
} from 'lucide-react';
import { getHistory, deleteHistoryItem, getResult } from '../services/api';

export default function HistoryPage({ onSelectVideo }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterVerdict, setFilterVerdict] = useState('ALL');
  const [filterModality, setFilterModality] = useState('ALL');

  // Selected non-video item detail modal
  const [selectedUniversalDossier, setSelectedUniversalDossier] = useState(null);
  const [modalLoading, setModalLoading] = useState(false);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const data = await getHistory();
      setItems(data);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleDelete = async (videoId, e) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete forensic report ${videoId}?`)) return;

    try {
      await deleteHistoryItem(videoId);
      setItems((prev) => prev.filter((item) => item.video_id !== videoId));
      if (selectedUniversalDossier?.video_id === videoId) {
        setSelectedUniversalDossier(null);
      }
    } catch (err) {
      console.error('Delete failed:', err);
      alert('Failed to delete report.');
    }
  };

  const handleItemClick = async (item) => {
    const mod = item.modality || (item.video_id?.startsWith('VID_') ? 'video' : 'video');
    if (mod === 'video') {
      if (onSelectVideo) onSelectVideo(item.video_id);
    } else {
      // Non-video: fetch complete result and display dossier modal
      setModalLoading(true);
      try {
        const fullReport = await getResult(item.video_id);
        setSelectedUniversalDossier(fullReport);
      } catch (e) {
        console.error('Failed to fetch full record:', e);
      } finally {
        setModalLoading(false);
      }
    }
  };

  const getModalityBadge = (modality) => {
    switch (modality) {
      case 'image':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-blue-50 text-blue-700 border border-blue-200">
            <ImageIcon className="w-3 h-3" />
            <span>Image</span>
          </span>
        );
      case 'audio':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-purple-50 text-purple-700 border border-purple-200">
            <Mic className="w-3 h-3" />
            <span>Audio</span>
          </span>
        );
      case 'pdf':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-amber-50 text-amber-700 border border-amber-200">
            <FileText className="w-3 h-3" />
            <span>PDF</span>
          </span>
        );
      case 'video':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-rose-50 text-rose-700 border border-rose-200">
            <FileVideo className="w-3 h-3" />
            <span>Video</span>
          </span>
        );
    }
  };

  const getVerdictBadge = (prediction, conf) => {
    const confPct = Math.round((conf || 0.85) * 100);
    const isDeepfake = prediction === 'AI_GENERATED' || prediction === 'FAKE' || (typeof prediction === 'string' && prediction.includes('SYNTHESIZED'));
    const isForged = prediction === 'FORGED' || (typeof prediction === 'string' && prediction.includes('FORGED'));
    const isAuthentic = prediction === 'REAL' || (typeof prediction === 'string' && prediction.includes('AUTHENTIC'));

    if (isAuthentic) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-600" />
          <span>Authentic ({confPct}%)</span>
        </span>
      );
    }
    if (isDeepfake) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-rose-50 text-rose-700 border border-rose-200">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-600" />
          <span>Deepfake ({confPct}%)</span>
        </span>
      );
    }
    if (isForged) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-amber-50 text-amber-700 border border-amber-200">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-600" />
          <span>Forged ({confPct}%)</span>
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-mono bg-slate-100 text-slate-700 border border-slate-200">
        {prediction}
      </span>
    );
  };

  const filteredItems = items.filter((item) => {
    const matchesSearch =
      item.filename?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.video_id?.toLowerCase().includes(searchQuery.toLowerCase());

    const itemMod = item.modality || (item.video_id?.startsWith('VID_') ? 'video' : 'video');
    const matchesModality = filterModality === 'ALL' || itemMod === filterModality;

    const isDeepfake = item.prediction === 'AI_GENERATED' || item.prediction === 'FAKE' || (typeof item.prediction === 'string' && item.prediction.includes('SYNTHESIZED'));
    const isAuthentic = item.prediction === 'REAL' || (typeof item.prediction === 'string' && item.prediction.includes('AUTHENTIC'));
    const isForged = item.prediction === 'FORGED' || (typeof item.prediction === 'string' && item.prediction.includes('FORGED'));

    let matchesFilter = true;
    if (filterVerdict === 'AI_GENERATED') matchesFilter = isDeepfake;
    else if (filterVerdict === 'REAL') matchesFilter = isAuthentic;
    else if (filterVerdict === 'FORGED') matchesFilter = isForged;

    return matchesSearch && matchesModality && matchesFilter;
  });

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 sm:px-6 space-y-6">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-mono mb-1">
            <HistoryIcon className="w-3 h-3" />
            <span className="uppercase tracking-wider font-semibold">Universal 5-Pillar Audit Registry</span>
          </div>
          <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
            Forensic Inspection History
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Immutable inspection dossiers for all 5 pillars (Images, Videos, Speech Audio, and PDF Documents).
          </p>
        </div>

        <button
          onClick={fetchHistory}
          className="h-8 px-3 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-mono transition-colors flex items-center gap-1.5 self-start sm:self-auto shadow-sm"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Ledger</span>
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by filename or Case ID..."
            className="w-full pl-9 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-slate-400"
          />
        </div>

        <div className="flex items-center gap-2.5 self-start sm:self-auto flex-wrap">
          {/* Modality Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-slate-500 font-mono">Pillars:</span>
            <select
              value={filterModality}
              onChange={(e) => setFilterModality(e.target.value)}
              className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-slate-700 focus:outline-none"
            >
              <option value="ALL">All Active Pillars ({items.length})</option>
              <option value="video">Videos (Pillar 2)</option>
              <option value="image">Images (Pillars 1 & 5)</option>
              <option value="audio">Audio (Pillar 3)</option>
            </select>
          </div>

          {/* Verdict Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-slate-500 font-mono">Verdict:</span>
            <select
              value={filterVerdict}
              onChange={(e) => setFilterVerdict(e.target.value)}
              className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-slate-700 focus:outline-none"
            >
              <option value="ALL">All Outcomes</option>
              <option value="AI_GENERATED">AI Generated / Deepfake</option>
              <option value="REAL">Authentic / Real</option>
              <option value="FORGED">Forged</option>
            </select>
          </div>
        </div>
      </div>

      {/* Table / List */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
        {loading ? (
          <div className="py-16 text-center text-xs font-mono text-slate-500">
            Loading inspection ledger...
          </div>
        ) : filteredItems.length > 0 ? (
          <div className="divide-y divide-slate-100">
            {filteredItems.map((item) => {
              const modality = item.modality || (item.video_id?.startsWith('VID_') ? 'video' : 'video');
              const isVideo = modality === 'video';

              return (
                <div
                  key={item.video_id}
                  onClick={() => handleItemClick(item)}
                  className="p-4 hover:bg-slate-50 transition-colors flex items-center justify-between gap-4 cursor-pointer"
                >
                  <div className="min-w-0 flex items-center gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm font-semibold text-slate-900 truncate">
                          {item.filename || `Evidence_${item.video_id}`}
                        </span>
                        {getModalityBadge(modality)}
                      </div>
                      
                      <div className="flex items-center gap-2 text-xs font-mono text-slate-500 mt-0.5">
                        <span>Case ID: {item.video_id}</span>
                        <span className="text-slate-300">•</span>
                        <span>{item.date ? item.date.replace('T', ' ').slice(0, 16) : 'Recorded Session'}</span>
                        <span className="text-slate-300">•</span>
                        <span className="truncate max-w-xs">{item.engines || 'Multi-Pillar Engine'}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    {getVerdictBadge(item.prediction, item.confidence)}

                    <button
                      type="button"
                      onClick={(e) => handleDelete(item.video_id, e)}
                      className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-slate-100 transition-colors"
                      title="Delete record"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>

                    <ChevronRight className="w-4 h-4 text-slate-400" />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="py-12 text-center text-xs font-mono text-slate-500">
            No inspection records match current filter.
          </div>
        )}
      </div>

      {/* Universal Non-Video Dossier Inspection Modal */}
      {selectedUniversalDossier && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-xl max-w-2xl w-full p-6 space-y-6 my-8">
            
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-slate-200 pb-4">
              <div className="flex items-center gap-2.5">
                {getModalityBadge(selectedUniversalDossier.modality)}
                <div>
                  <h3 className="text-base font-semibold text-slate-900">
                    {selectedUniversalDossier.filename}
                  </h3>
                  <p className="text-xs font-mono text-slate-500">
                    Record ID: {selectedUniversalDossier.video_id} • Analyzed: {selectedUniversalDossier.processing?.processed_at?.replace('T', ' ').slice(0, 16)}
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setSelectedUniversalDossier(null)}
                className="w-8 h-8 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-700 hover:bg-slate-100"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Verdict Card */}
            {selectedUniversalDossier.consensus && (
              <div className={`p-4 rounded-xl border flex items-center justify-between ${
                selectedUniversalDossier.consensus.is_real 
                  ? 'bg-emerald-50/70 border-emerald-200 text-emerald-900' 
                  : 'bg-rose-50/70 border-rose-200 text-rose-900'
              }`}>
                <div>
                  <span className="text-[11px] font-mono uppercase tracking-wider text-slate-500 block">
                    Consensus Verdict
                  </span>
                  <h4 className="text-lg font-bold mt-0.5">
                    {selectedUniversalDossier.consensus.verdict}
                  </h4>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Engines: {selectedUniversalDossier.consensus.engines}
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-[11px] font-mono uppercase text-slate-500 block">Confidence</span>
                  <span className="text-2xl font-bold font-mono">
                    {selectedUniversalDossier.consensus.confidence}%
                  </span>
                </div>
              </div>
            )}

            {/* Detailed Sub-Pillar Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              {selectedUniversalDossier.pillar1 && (
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="font-mono text-slate-500 uppercase font-semibold text-[11px] block">
                    Pillar 1 • ViT Neural Spectra
                  </span>
                  <div className="font-semibold text-slate-900 mt-1">
                    {selectedUniversalDossier.pillar1.verdict}
                  </div>
                  <div className="text-slate-500 mt-0.5">
                    Confidence: <strong>{selectedUniversalDossier.pillar1.confidence}%</strong>
                  </div>
                </div>
              )}

              {selectedUniversalDossier.pillar5 && (
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="font-mono text-slate-500 uppercase font-semibold text-[11px] block">
                    Pillar 5 • Shadow RANSAC Physics
                  </span>
                  <div className="font-semibold text-slate-900 mt-1">
                    {selectedUniversalDossier.pillar5.verdict}
                  </div>
                  <div className="text-slate-500 mt-0.5">
                    Inliers: <strong>{Math.round((selectedUniversalDossier.pillar5.inlier_ratio || 0) * 100)}%</strong>
                  </div>
                </div>
              )}

              {selectedUniversalDossier.pillar3 && (
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="font-mono text-slate-500 uppercase font-semibold text-[11px] block">
                    Pillar 3 • Speech & Acoustic Transformer
                  </span>
                  <div className="font-semibold text-slate-900 mt-1">
                    {selectedUniversalDossier.pillar3.prediction}
                  </div>
                  <div className="text-slate-500 mt-0.5">
                    Confidence: <strong>{selectedUniversalDossier.pillar3.confidence}%</strong>
                  </div>
                </div>
              )}
            </div>

            {/* Modal Actions */}
            <div className="flex justify-end gap-2 pt-2 border-t border-slate-200">
              <button
                type="button"
                onClick={() => setSelectedUniversalDossier(null)}
                className="px-4 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium transition-colors"
              >
                Close Dossier
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
