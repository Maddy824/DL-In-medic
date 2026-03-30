import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Camera as CameraIcon, SwitchCamera, Aperture, Upload, X } from 'lucide-react';

export default function Camera({ onCapture, isAnalyzing }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [facingMode, setFacingMode] = useState('environment');
  const [cameraActive, setCameraActive] = useState(false);
  const [preview, setPreview] = useState(null);

  const startCamera = useCallback(async () => {
    try {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setStream(mediaStream);
      setCameraActive(true);
    } catch (err) {
      console.error('Camera access denied:', err);
      alert('Camera access is required for real-time screening. Please allow camera permissions.');
    }
  }, [facingMode, stream]);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      setStream(null);
    }
    setCameraActive(false);
  }, [stream]);

  const switchCamera = useCallback(() => {
    setFacingMode(prev => (prev === 'environment' ? 'user' : 'environment'));
  }, []);

  useEffect(() => {
    if (cameraActive) {
      startCamera();
    }
    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop());
    };
  }, [facingMode]);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      if (blob) {
        const url = URL.createObjectURL(blob);
        setPreview(url);
        onCapture(blob);
      }
    }, 'image/jpeg', 0.92);
  }, [onCapture]);

  const handleFileUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    onCapture(file);
    e.target.value = '';
  }, [onCapture]);

  const clearPreview = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
  }, [preview]);

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Camera / Preview area */}
      <div className="relative w-full max-w-md aspect-[3/4] rounded-2xl overflow-hidden bg-gray-900 border border-white/10">
        {preview ? (
          <>
            <img src={preview} alt="Captured" className="w-full h-full object-cover" />
            {!isAnalyzing && (
              <button
                onClick={clearPreview}
                className="absolute top-3 right-3 p-2 bg-black/60 rounded-full hover:bg-black/80 transition-colors"
              >
                <X size={18} />
              </button>
            )}
            {isAnalyzing && (
              <div className="absolute inset-0 bg-black/30 flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-12 h-12 border-4 border-medical-400 border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm font-medium text-medical-300">Analyzing...</span>
                </div>
                <div className="scan-line" />
              </div>
            )}
          </>
        ) : cameraActive ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover mirror"
              style={{ transform: facingMode === 'user' ? 'scaleX(-1)' : 'none' }}
            />
            {/* Viewfinder overlay */}
            <div className="viewfinder-corner tl" />
            <div className="viewfinder-corner tr" />
            <div className="viewfinder-corner bl" />
            <div className="viewfinder-corner br" />
            <div className="absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t from-black/60 to-transparent" />
            <div className="absolute bottom-3 left-0 right-0 flex items-center justify-center gap-6">
              <button
                onClick={switchCamera}
                className="p-3 bg-white/10 backdrop-blur rounded-full hover:bg-white/20 transition-colors"
                title="Switch camera"
              >
                <SwitchCamera size={20} />
              </button>
              <button
                onClick={captureFrame}
                className="p-4 bg-medical-500 rounded-full hover:bg-medical-400 transition-colors shadow-lg shadow-medical-500/30 ring-4 ring-white/20"
                title="Capture"
              >
                <Aperture size={28} />
              </button>
              <button
                onClick={stopCamera}
                className="p-3 bg-white/10 backdrop-blur rounded-full hover:bg-white/20 transition-colors"
                title="Stop camera"
              >
                <X size={20} />
              </button>
            </div>
          </>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-6 p-6">
            <div className="w-20 h-20 rounded-full bg-medical-500/10 flex items-center justify-center">
              <CameraIcon size={36} className="text-medical-400" />
            </div>
            <div className="text-center">
              <p className="text-white/80 font-medium mb-1">Capture or Upload</p>
              <p className="text-white/40 text-sm">Use your camera or upload an image of a skin lesion for AI screening</p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={startCamera}
                className="px-5 py-2.5 bg-medical-600 hover:bg-medical-500 rounded-xl font-medium transition-colors flex items-center gap-2"
              >
                <CameraIcon size={18} />
                Open Camera
              </button>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-5 py-2.5 bg-white/10 hover:bg-white/15 rounded-xl font-medium transition-colors flex items-center gap-2"
              >
                <Upload size={18} />
                Upload
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Hidden elements */}
      <canvas ref={canvasRef} className="hidden" />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileUpload}
        className="hidden"
      />

      {/* Quick upload when camera is active */}
      {cameraActive && !preview && (
        <button
          onClick={() => fileInputRef.current?.click()}
          className="text-sm text-white/50 hover:text-white/70 flex items-center gap-1.5 transition-colors"
        >
          <Upload size={14} />
          Or upload an image instead
        </button>
      )}
    </div>
  );
}
