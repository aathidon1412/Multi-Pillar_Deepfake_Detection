import React, { useState, useEffect } from 'react';
import { History as HistoryIcon, Eye, Trash2, Search, Filter, ShieldCheck, Cpu, Scissors, RefreshCw } from 'lucide-react';
import { getHistory, deleteHistoryItem } from '../services/api';

export default function HistoryPage({ onSelectVideo }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterVerdict, setFilterVerdict] = useState('ALL');

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
    } catch (err) {
      console.error('Delete failed:', err);
      alert('Failed to delete report.');
    }
  };

  const getVerdictBadge = (prediction) => {
    switch (prediction) {
      case 'REAL':
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 flex items-center space-x-1.5 w-max">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>REAL</span>
          </span>
        );
      case 'AI_GENERATED':
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-rose-950/80 text-rose-400 border border-rose-800/80 flex items-center space-x-1.5 w-max">
            <Cpu className="w-3.5 h-3.5" />
            <span>AI GENERATED</span>
          </span>
        );
      case 'FORGED':
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-amber-950/80 text-amber-400 border border-amber-800/80 flex items-center space-x-1.5 w-max">
            <Scissors className="w-3.5 h-3.5" />
            <span>FORGED</span>
          </span>
        );
      default:
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-mono bg-slate-800 text-slate-300 border border-slate-700 w-max">
            {prediction}
          </span>
        );
    }
  };

  // Filter items
  const filteredItems = items.filter((item) => {
    const matchesSearch =
      item.filename?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.video_id?.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesFilter = filterVerdict === 'ALL' || item.prediction === filterVerdict;

    return matchesSearch && matchesFilter;
  });

  // Summary counts
  const totalCount = items.length;
  const aiCount = items.filter((i) => i.prediction === 'AI_GENERATED').length;
  const realCount = items.filter((i) => i.prediction === 'REAL').length;
  const forgedCount = items.filter((i) => i.prediction === 'FORGED').length;

  return (
    <div className="max-w-7xl mx-auto space-y-6 py-6">
      
      {/* Header & Metrics */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
            <HistoryIcon className="w-4 h-4" />
            <span>JSON STORAGE REGISTRY</span>
          </div>
          <h1 className="text-3xl font-black text-white">Forensic Analysis History</h1>
          <p className="text-sm text-slate-400 font-mono">
            Loaded directly from <code className="text-slate-200">storage/results/*.json</code> without an external database.
          </p>
        </div>

        <button
          onClick={fetchHistory}
          className="self-start md:self-auto px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 text-xs font-mono transition-colors flex items-center space-x-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Records</span>
        </button>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <span className="text-xs font-mono text-slate-400 uppercase block mb-1">Total Analyzed</span>
          <span className="text-2xl font-bold font-mono text-white">{totalCount}</span>
        </div>
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <span className="text-xs font-mono text-rose-400 uppercase block mb-1">AI Generated</span>
          <span className="text-2xl font-bold font-mono text-rose-400">{aiCount}</span>
        </div>
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <span className="text-xs font-mono text-emerald-400 uppercase block mb-1">Real / Genuine</span>
          <span className="text-2xl font-bold font-mono text-emerald-400">{realCount}</span>
        </div>
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <span className="text-xs font-mono text-amber-400 uppercase block mb-1">Forged</span>
          <span className="text-2xl font-bold font-mono text-amber-400">{forgedCount}</span>
        </div>
      </div>

      {/* Search & Filter Toolbar */}
      <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by ID or filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        <div className="flex items-center space-x-2 w-full sm:w-auto overflow-x-auto">
          <Filter className="w-4 h-4 text-slate-500 shrink-0" />
          {['ALL', 'REAL', 'AI_GENERATED', 'FORGED'].map((v) => (
            <button
              key={v}
              onClick={() => setFilterVerdict(v)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all shrink-0 ${
                filterVerdict === v
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'bg-slate-950/60 text-slate-400 border border-slate-800 hover:text-slate-200'
              }`}
            >
              {v.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* History Table */}
      <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-2xl">
        {loading ? (
          <div className="py-16 text-center text-slate-400 font-mono text-xs">
            Scanning JSON reports...
          </div>
        ) : filteredItems.length === 0 ? (
          <div className="py-16 text-center space-y-2">
            <p className="text-slate-300 text-sm font-semibold">No forensic reports found</p>
            <p className="text-slate-500 text-xs font-mono">Upload a video to generate the first JSON report.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-950 border-b border-slate-800 text-slate-400 uppercase tracking-wider">
                <tr>
                  <th className="py-3.5 px-4">Video ID</th>
                  <th className="py-3.5 px-4">Original Filename</th>
                  <th className="py-3.5 px-4">Prediction</th>
                  <th className="py-3.5 px-4">Confidence</th>
                  <th className="py-3.5 px-4">Duration</th>
                  <th className="py-3.5 px-4">Timestamp</th>
                  <th className="py-3.5 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredItems.map((item) => (
                  <tr
                    key={item.video_id}
                    onClick={() => onSelectVideo(item.video_id)}
                    className="hover:bg-slate-800/40 transition-colors cursor-pointer group"
                  >
                    <td className="py-3.5 px-4 font-bold text-cyan-400 group-hover:underline">
                      {item.video_id}
                    </td>
                    <td className="py-3.5 px-4 text-slate-200 font-sans font-medium truncate max-w-xs">
                      {item.filename}
                    </td>
                    <td className="py-3.5 px-4">
                      {getVerdictBadge(item.prediction)}
                    </td>
                    <td className="py-3.5 px-4 font-bold text-slate-100">
                      {Math.round((item.confidence || 0) * 100)}%
                    </td>
                    <td className="py-3.5 px-4 text-slate-400">
                      {item.duration_seconds ? `${item.duration_seconds}s` : 'N/A'}
                    </td>
                    <td className="py-3.5 px-4 text-slate-500">
                      {item.date?.replace('T', ' ').slice(0, 19) || 'N/A'}
                    </td>
                    <td className="py-3.5 px-4 text-right space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectVideo(item.video_id);
                        }}
                        className="p-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 transition-colors"
                        title="View Full Report"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={(e) => handleDelete(item.video_id, e)}
                        className="p-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 transition-colors"
                        title="Delete Report"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}
