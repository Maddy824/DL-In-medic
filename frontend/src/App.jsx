import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Scan,
  Brain,
  Box,
  Activity,
  Info,
  Wifi,
  WifiOff,
  Github,
} from 'lucide-react';

import Camera from './components/Camera';
import Results from './components/Results';
import ThreeDViewer from './components/ThreeDViewer';
import AugmentationViewer from './components/AugmentationViewer';
import { analyzeImage, augmentImage, healthCheck } from './utils/api';

const TABS = [
  { id: 'scan', label: 'Scan', icon: Scan },
  { id: '3d', label: '3D View', icon: Box },
  { id: 'augment', label: 'Augmentation', icon: Brain },
  { id: 'about', label: 'About', icon: Info },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('scan');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [augData, setAugData] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [lastBlob, setLastBlob] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [error, setError] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    const check = async () => {
      const ok = await healthCheck();
      setBackendOnline(ok);
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleCapture = useCallback(async (blob) => {
    setError(null);
    setResult(null);
    setAugData(null);
    setLastBlob(blob);

    const url = URL.createObjectURL(blob);
    setPreviewUrl(url);

    // Auto-analyze
    setIsAnalyzing(true);
    try {
      const res = await analyzeImage(blob);
      setResult(res);
    } catch (err) {
      setError(err.message || 'Analysis failed. Is the backend running?');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleAugment = useCallback(async () => {
    if (!lastBlob) return;
    setError(null);
    try {
      const data = await augmentImage(lastBlob);
      setAugData(data);
      setActiveTab('augment');
    } catch (err) {
      setError('Augmentation failed. Is the backend running?');
    }
  }, [lastBlob]);

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-medical-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-medical-500/20">
              <Activity size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-white to-medical-300 bg-clip-text text-transparent">
                DermScreen AI
              </h1>
              <p className="text-[10px] text-white/30 -mt-0.5">Real-time Dermatological Screening</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
              backendOnline
                ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                : 'bg-red-500/10 text-red-400 border border-red-500/20'
            }`}>
              {backendOnline ? <Wifi size={12} /> : <WifiOff size={12} />}
              {backendOnline ? 'Online' : 'Offline'}
            </div>
          </div>
        </div>
      </header>

      {/* Tab bar */}
      <nav className="sticky top-[57px] z-40 bg-gray-950/60 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-5xl mx-auto px-4 flex gap-1">
          {TABS.map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors relative ${
                  active ? 'text-medical-400' : 'text-white/40 hover:text-white/60'
                }`}
              >
                <Icon size={16} />
                {tab.label}
                {active && (
                  <motion.div
                    layoutId="tab-indicator"
                    className="absolute bottom-0 left-2 right-2 h-0.5 bg-medical-500 rounded-full"
                  />
                )}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-6">
        <AnimatePresence mode="wait">
          {/* ── SCAN TAB ── */}
          {activeTab === 'scan' && (
            <motion.div
              key="scan"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col lg:flex-row gap-6 items-start"
            >
              <div className="w-full lg:w-1/2 flex flex-col items-center gap-4">
                <Camera onCapture={handleCapture} isAnalyzing={isAnalyzing} />

                {/* Action buttons */}
                {lastBlob && result && (
                  <div className="flex gap-2">
                    <button
                      onClick={handleAugment}
                      className="px-4 py-2 bg-purple-600/20 border border-purple-500/30 text-purple-300 rounded-xl text-sm font-medium hover:bg-purple-600/30 transition-colors flex items-center gap-2"
                    >
                      <Brain size={16} />
                      View 3D Augmentation
                    </button>
                    <button
                      onClick={() => setActiveTab('3d')}
                      className="px-4 py-2 bg-cyan-600/20 border border-cyan-500/30 text-cyan-300 rounded-xl text-sm font-medium hover:bg-cyan-600/30 transition-colors flex items-center gap-2"
                    >
                      <Box size={16} />
                      3D Surface
                    </button>
                  </div>
                )}
              </div>

              <div className="w-full lg:w-1/2 flex flex-col items-center">
                {error && (
                  <div className="w-full max-w-md p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm mb-4">
                    {error}
                  </div>
                )}
                <Results result={result} />
                {!result && !isAnalyzing && (
                  <div className="text-center text-white/30 mt-8">
                    <Scan size={48} className="mx-auto mb-3 opacity-30" />
                    <p className="text-sm">Capture or upload an image to begin screening</p>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* ── 3D VIEW TAB ── */}
          {activeTab === '3d' && (
            <motion.div
              key="3d"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4"
            >
              <div className="text-center mb-2">
                <h2 className="text-xl font-bold text-white">3D Skin Surface Viewer</h2>
                <p className="text-sm text-white/40 mt-1">
                  Interactive 3D projection of captured skin lesion. Drag to rotate, scroll to zoom.
                </p>
              </div>
              <div className="w-full max-w-lg">
                <ThreeDViewer imageUrl={previewUrl} />
              </div>
              {!previewUrl && (
                <p className="text-sm text-white/30">Capture an image first to project it onto the 3D surface</p>
              )}
            </motion.div>
          )}

          {/* ── AUGMENTATION TAB ── */}
          {activeTab === 'augment' && (
            <motion.div
              key="augment"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4"
            >
              {augData ? (
                <AugmentationViewer augData={augData} onRerun={handleAugment} />
              ) : (
                <div className="text-center text-white/30 mt-12">
                  <Brain size={48} className="mx-auto mb-3 opacity-30" />
                  <p className="text-sm">Capture and analyze an image first, then view the 3D augmentation pipeline</p>
                  {lastBlob && (
                    <button
                      onClick={handleAugment}
                      className="mt-4 px-5 py-2.5 bg-purple-600 hover:bg-purple-500 rounded-xl font-medium transition-colors"
                    >
                      Run 3D Augmentation
                    </button>
                  )}
                </div>
              )}
            </motion.div>
          )}

          {/* ── ABOUT TAB ── */}
          {activeTab === 'about' && (
            <motion.div
              key="about"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-2xl mx-auto space-y-6"
            >
              <div className="glass-card p-6">
                <h2 className="text-xl font-bold text-white mb-4">About DermScreen AI</h2>
                <p className="text-white/60 text-sm leading-relaxed mb-4">
                  DermScreen AI is a real-time deep learning system for dermatological screening. It uses 
                  your phone's camera to capture skin lesion images, processes them through a MobileNetV2-based 
                  neural network with channel attention, and provides classification across 7 common skin 
                  conditions with risk assessment.
                </p>
                <h3 className="text-lg font-semibold text-white mb-2">Key Features</h3>
                <ul className="space-y-2 text-sm text-white/60">
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">Real-time Camera Input:</strong> WebRTC-powered camera access with front/rear switching</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">DL Classification:</strong> MobileNetV2 backbone with channel attention for 7 skin lesion types</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">3D Augmentation Pipeline:</strong> Perspective warping, surface-normal lighting, elastic deformation, color augmentation</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">3D Surface Viewer:</strong> Interactive Three.js visualization projecting lesion onto curved skin geometry</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">Risk Assessment:</strong> Color-coded risk levels (Critical, High, Moderate, Low) with clinical descriptions</span>
                  </li>
                </ul>
              </div>

              <div className="glass-card p-6">
                <h3 className="text-lg font-semibold text-white mb-2">Model Architecture</h3>
                <div className="bg-black/30 rounded-xl p-4 font-mono text-xs text-medical-300 space-y-1">
                  <p>MobileNetV2 (ImageNet Pretrained)</p>
                  <p>  → Channel Attention (SE-style)</p>
                  <p>  → Global Average Pooling</p>
                  <p>  → FC(1280→512) + BN + ReLU + Dropout(0.3)</p>
                  <p>  → FC(512→256) + BN + ReLU + Dropout(0.2)</p>
                  <p>  → FC(256→7) → Softmax</p>
                  <p className="text-white/30 mt-2">Auxiliary: FC(1280→128→1→Sigmoid) confidence</p>
                </div>
              </div>

              <div className="glass-card p-6">
                <h3 className="text-lg font-semibold text-white mb-2">Classification Categories</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {[
                    { name: 'Melanoma', risk: 'critical' },
                    { name: 'Basal Cell Carcinoma', risk: 'high' },
                    { name: 'Actinic Keratosis', risk: 'moderate' },
                    { name: 'Vascular Lesion', risk: 'moderate' },
                    { name: 'Benign Keratosis', risk: 'low' },
                    { name: 'Melanocytic Nevi', risk: 'low' },
                    { name: 'Dermatofibroma', risk: 'low' },
                  ].map(c => (
                    <div key={c.name} className={`px-3 py-2 rounded-lg border text-sm risk-${c.risk}`}>
                      {c.name}
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/20">
                <p className="text-xs text-amber-200/70 leading-relaxed">
                  <strong className="text-amber-300">⚕️ Medical Disclaimer:</strong> This tool is for educational 
                  and screening purposes only. It is NOT a substitute for professional medical diagnosis. 
                  The demo model uses ImageNet pretrained weights and has not been clinically validated. 
                  Always consult a board-certified dermatologist for any skin concerns.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-4 text-center text-xs text-white/20">
        DermScreen AI v1.0 — Deep Learning Dermatological Screening System
      </footer>
    </div>
  );
}
