import React, { useState } from 'react';
import Navbar from './components/Navbar';
import UploadPage from './pages/UploadPage';
import ProcessingPage from './pages/ProcessingPage';
import ResultPage from './pages/ResultPage';
import HistoryPage from './pages/HistoryPage';
import ArchitecturePage from './pages/ArchitecturePage';
import Pillars1And5Page from './pages/Pillars1And5Page';
import Pillar3Page from './pages/Pillar3Page';
import Pillar4Page from './pages/Pillar4Page';

export default function App() {
  const [activeTab, setActiveTab] = useState('video');
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
    setActiveTab('video');
  };

  const handleSelectHistoricalVideo = (videoId) => {
    setCurrentVideoId(videoId);
    window.location.hash = `result/${videoId}`;
    setActiveTab('result');
  };

  return (
    <div className="min-h-screen bg-cyber-grid flex flex-col justify-between">
      
      {/* Navigation Header */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full pb-12">
        {/* Pillar 2 Video Forensics Flow */}
        {(activeTab === 'video' || activeTab === 'upload') && (
          <UploadPage onStartProcessing={handleStartProcessing} />
        )}

        {activeTab === 'processing' && (
          <ProcessingPage
            videoId={currentVideoId}
            filename={currentFilename}
            onProcessingComplete={handleProcessingComplete}
            onCancel={handleAnalyzeAnother}
          />
        )}

        {activeTab === 'result' && (
          <ResultPage
            videoId={currentVideoId}
            onAnalyzeAnother={handleAnalyzeAnother}
          />
        )}

        {/* Pillars 1 & 5: Universal Image Forensics */}
        {activeTab === 'image' && (
          <Pillars1And5Page />
        )}

        {/* Pillar 3: Audio & Speech Forensics */}
        {activeTab === 'audio' && (
          <Pillar3Page />
        )}

        {/* Pillar 4: Document & PDF Forensics */}
        {activeTab === 'document' && (
          <Pillar4Page />
        )}

        {/* History Registry */}
        {activeTab === 'history' && (
          <HistoryPage onSelectVideo={handleSelectHistoricalVideo} />
        )}

        {/* Architecture Specs */}
        {activeTab === 'architecture' && (
          <ArchitecturePage />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-md py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-slate-500 gap-2">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            <span className="text-slate-400">AuthentiGuard AI • All 5 Pillars Unified Forensics</span>
          </div>
          <div>
            Video (Pillar 2) • Image (P1 & P5) • Audio (Pillar 3) • Document (Pillar 4)
          </div>
        </div>
      </footer>

    </div>
  );
}
