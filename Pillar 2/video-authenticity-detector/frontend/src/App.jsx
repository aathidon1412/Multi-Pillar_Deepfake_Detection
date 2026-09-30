import React, { useState } from 'react';
import Navbar from './components/Navbar';
import UploadPage from './pages/UploadPage';
import ProcessingPage from './pages/ProcessingPage';
import ResultPage from './pages/ResultPage';
import HistoryPage from './pages/HistoryPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('unified');
  const [currentVideoId, setCurrentVideoId] = useState(null);
  const [currentFilename, setCurrentFilename] = useState('');

  React.useEffect(() => {
    const parseHash = () => {
      const hash = window.location.hash;
      if (hash.startsWith('#result/')) {
        const vid = hash.replace('#result/', '').trim();
        if (vid) {
          setCurrentVideoId(vid);
          setActiveTab('result');
        }
      }
    };
    parseHash();
    window.addEventListener('hashchange', parseHash);
    return () => window.removeEventListener('hashchange', parseHash);
  }, []);

  const handleStartProcessing = (videoId, filename) => {
    setCurrentVideoId(videoId);
    setCurrentFilename(filename);
    setActiveTab('processing');
  };

  const handleProcessingComplete = (videoId) => {
    setCurrentVideoId(videoId);
    window.location.hash = `result/${videoId}`;
    setActiveTab('result');
  };

  const handleAnalyzeAnother = () => {
    setCurrentVideoId(null);
    setCurrentFilename('');
    window.location.hash = '';
    setActiveTab('unified');
  };

  const handleSelectHistoricalVideo = (videoId) => {
    setCurrentVideoId(videoId);
    window.location.hash = `result/${videoId}`;
    setActiveTab('result');
  };

  return (
    <div className="min-h-screen bg-[#f8f9ff] flex flex-col justify-between text-slate-800">
      
      {/* Navigation Header */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 w-full pb-12">
        {/* All-in-One Universal Ingestion Workspace */}
        {(activeTab === 'unified' || activeTab === 'upload' || activeTab === 'video') && (
          <UploadPage 
            onStartProcessing={handleStartProcessing} 
            onSelectHistoricalVideo={handleSelectHistoricalVideo}
          />
        )}

        {/* Video Async Processing Screen */}
        {activeTab === 'processing' && (
          <ProcessingPage
            videoId={currentVideoId}
            filename={currentFilename}
            onProcessingComplete={handleProcessingComplete}
            onCancel={handleAnalyzeAnother}
          />
        )}

        {/* Video Forensic Dossier & Evidence Canvas */}
        {activeTab === 'result' && (
          <ResultPage
            videoId={currentVideoId}
            onAnalyzeAnother={handleAnalyzeAnother}
          />
        )}

        {/* Audit History Ledger */}
        {activeTab === 'history' && (
          <HistoryPage onSelectVideo={handleSelectHistoricalVideo} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-700">USMFE Core v4.2.1-SEC</span>
            <span className="text-slate-300">•</span>
            <span className="font-mono text-slate-500">Universal Multi-Pillar Media Forensics</span>
          </div>
          <div className="font-mono text-slate-400">
            NIST SP 800-86 Compliant Trace Logs • Unaltered Evidence Standard
          </div>
        </div>
      </footer>

    </div>
  );
}
