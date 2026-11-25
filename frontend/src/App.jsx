import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Scan,
  Brain,
  Box,
  Activity,
  Info,
  Wifi,
  WifiOff,
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
  const [analysisError, setAnalysisError] = useState(null);
  const [augmentationError, setAugmentationError] = useState(null);
  const analysisRequest = useRef(0);
  const augmentationRequest = useRef(0);
  const selectedBlob = useRef(null);

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

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => () => {
    analysisRequest.current++;
    augmentationRequest.current++;
  }, []);

  const clearImage = useCallback(() => {
    analysisRequest.current++;
    augmentationRequest.current++;
    selectedBlob.current = null;
    setAnalysisError(null);
    setAugmentationError(null);
    setResult(null);
    setAugData(null);
    setLastBlob(null);
    setPreviewUrl(null);
    setIsAnalyzing(false);
  }, []);

  const handleCapture = useCallback(async (blob) => {
    clearImage();
    selectedBlob.current = blob;
    setLastBlob(blob);

    const url = URL.createObjectURL(blob);
    setPreviewUrl(url);

    // Auto-analyze
    const request = ++analysisRequest.current;
    setIsAnalyzing(true);
    try {
      const res = await analyzeImage(blob);
      if (request === analysisRequest.current) setResult(res);
    } catch (err) {
      if (request === analysisRequest.current) {
        setAnalysisError(err.message || 'Analysis failed. Is the backend running?');
      }
    } finally {
      if (request === analysisRequest.current) setIsAnalyzing(false);
    }
  }, [clearImage]);

  const handleAugment = useCallback(async () => {
    if (!lastBlob || lastBlob !== selectedBlob.current) return;
    const request = ++augmentationRequest.current;
    setAugmentationError(null);
    setActiveTab('augment');
    try {
      const data = await augmentImage(lastBlob);
      if (request === augmentationRequest.current) setAugData(data);
    } catch (err) {
      if (request === augmentationRequest.current) {
        setAugmentationError(err.message || 'Augmentation failed. Is the backend running?');
      }
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
              <p className="text-[10px] text-white/30 -mt-0.5">Educational Computer Vision</p>
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
                <Camera onCapture={handleCapture} onClear={clearImage} preview={previewUrl} isAnalyzing={isAnalyzing} />

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
                {analysisError && (
                  <div role="alert" className="w-full max-w-md p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm mb-4">
                    {analysisError}
                  </div>
                )}
                <Results result={result} />
                {!result && !isAnalyzing && (
                  <div className="text-center text-white/30 mt-8">
                    <Scan size={48} className="mx-auto mb-3 opacity-30" />
                    <p className="text-sm">Capture or upload an image to explore</p>
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
                  Image projected onto a simulated curved surface, not a reconstruction. Drag to rotate, scroll to zoom.
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
              {augmentationError && (
                <div role="alert" className="w-full max-w-md p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm">
                  {augmentationError}
                </div>
              )}
              {augData ? (
                <AugmentationViewer augData={augData} onRerun={handleAugment} />
              ) : (
                <div className="text-center text-white/30 mt-12">
                  <Brain size={48} className="mx-auto mb-3 opacity-30" />
                  <p className="text-sm">Capture or upload an image first, then view the 3D augmentation pipeline</p>
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
                  DermScreen AI is an educational computer-vision project. Explore how perspective,
                  lighting and color transforms change an image, or train a MobileNetV2 classifier
                  for seven dataset categories. No trained weights are included.
                </p>
                <h3 className="text-lg font-semibold text-white mb-2">Key Features</h3>
                <ul className="space-y-2 text-sm text-white/60">
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">Camera Input:</strong> Camera capture and image upload</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">Trainable Classifier:</strong> MobileNetV2 with channel attention for seven dataset labels</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">3D Augmentation Pipeline:</strong> Perspective warping, surface-normal lighting, elastic deformation, color augmentation</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-medical-400 mt-0.5">•</span>
                    <span><strong className="text-white/80">3D Surface Viewer:</strong> Interactive Three.js visualization projecting lesion onto curved skin geometry</span>
                  </li>
                </ul>
              </div>

              <div className="glass-card p-6">
                <h3 className="text-lg font-semibold text-white mb-2">Model Architecture</h3>
                <div className="bg-black/30 rounded-xl p-4 font-mono text-xs text-medical-300 space-y-1">
                  <p>MobileNetV2 (optional ImageNet initialization)</p>
                  <p>  → Channel Attention (SE-style)</p>
                  <p>  → Global Average Pooling</p>
                  <p>  → FC(1280→512) + BN + ReLU + Dropout(0.3)</p>
                  <p>  → FC(512→256) + BN + ReLU + Dropout(0.2)</p>
                  <p>  → FC(256→7) → Softmax</p>
                </div>
              </div>

              <div className="glass-card p-6">
                <h3 className="text-lg font-semibold text-white mb-2">Classification Categories</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {[
                    'Melanoma', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Vascular Lesion',
                    'Benign Keratosis', 'Melanocytic Nevi', 'Dermatofibroma',
                  ].map(name => (
                    <div key={name} className="px-3 py-2 rounded-lg border border-white/10 text-sm text-white/60">
                      {name}
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/20">
                <p className="text-xs text-amber-200/70 leading-relaxed">
                  <strong className="text-amber-300">Educational use only:</strong> This is not a medical device
                  and has not been clinically validated. Even with trained weights, class scores are not
                  estimates of medical risk and must not be used for diagnosis or treatment decisions.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-4 text-center text-xs text-white/20">
        DermScreen AI v1.0 — Educational computer vision, not a medical device
      </footer>
    </div>
  );
}
