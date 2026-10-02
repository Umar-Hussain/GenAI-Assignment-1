import React, { useState, useRef, useEffect } from 'react';
import { Upload, Camera, Play, Download, Zap, Palette, Video, VideoOff, Eye } from 'lucide-react';
import { faceToSketch } from '../utils/api';
import ImageComparisonSlider from './ImageComparisonSlider';

export default function FaceToSketch() {
  const [photo, setPhoto] = useState(null);
  const [sketch, setSketch] = useState(null);
  const [selectedStyle, setSelectedStyle] = useState(1);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState('gallery');

  // Webcam state
  const [webcamActive, setWebcamActive] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const styles = [
    { id: 1, name: 'Style 1', label: 'Classic Pencil Hatching', desc: 'Fine graphite lines, subtle contour shading' },
    { id: 2, name: 'Style 2', label: 'Charcoal & Wash', desc: 'Deep tonal contrast and textured gradients' },
    { id: 3, name: 'Style 3', label: 'Artistic Caricature Contour', desc: 'Bold expressive ink boundaries and clean strokes' },
  ];

  const sampleFaces = [
    { name: 'Portrait 1', url: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=256&h=256&fit=crop' },
    { name: 'Portrait 2', url: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=256&h=256&fit=crop' },
    { name: 'Portrait 3', url: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=256&h=256&fit=crop' },
  ];

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      stopWebcam();
      const reader = new FileReader();
      reader.onload = (event) => {
        setPhoto(event.target.result);
        setSketch(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSelectSample = async (url) => {
    stopWebcam();
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onload = (e) => {
        setPhoto(e.target.result);
        setSketch(null);
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error(err);
    }
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 256, height: 256 } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setWebcamActive(true);
    } catch (err) {
      alert('Camera access denied or unavailable: ' + err.message);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setWebcamActive(false);
  };

  const captureWebcam = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, 128, 128);
    const dataUrl = canvas.toDataURL('image/png');
    setPhoto(dataUrl);
    setSketch(null);
    stopWebcam();
  };

  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  const handleGenerateSketch = async () => {
    if (!photo) return;
    setLoading(true);
    try {
      const res = await fetch(photo);
      const blob = await res.blob();
      const data = await faceToSketch(blob, selectedStyle);
      setSketch(`data:image/png;base64,${data.sketch_image}`);
      setInferenceTime(data.inference_time_ms);
    } catch (err) {
      alert(`Sketch generation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center space-x-2">
            <span>Face-to-Sketch Generator</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-pink-500/10 text-pink-400 border border-pink-500/20">
              Task 4 (Conditional GAN)
            </span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            U-Net generator conditioned on FS2K style embeddings with PatchGAN discriminator supervision for authentic facial sketch synthesis.
          </p>
        </div>

        {inferenceTime !== null && (
          <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
            <Zap className="w-4 h-4 text-amber-400" />
            <span className="text-slate-400">Synthesis Latency:</span>
            <span className="font-semibold text-emerald-400">{inferenceTime.toFixed(1)} ms</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Card */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-5">
          <div className="space-y-3">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              1. Input Portrait Photo
            </label>

            <div className="grid grid-cols-3 gap-2">
              {sampleFaces.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSelectSample(s.url)}
                  className="group relative rounded-lg overflow-hidden border border-slate-700 hover:border-pink-500 transition-all aspect-square"
                >
                  <img src={s.url} alt={s.name} className="w-full h-full object-cover group-hover:scale-105 transition-all" />
                  <span className="absolute bottom-0 inset-x-0 bg-slate-950/80 text-[10px] py-0.5 text-center text-slate-300">
                    {s.name}
                  </span>
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col items-center justify-center border border-dashed border-slate-700 hover:border-slate-500 rounded-lg p-3 cursor-pointer bg-slate-950/40 transition-all">
                <Upload className="w-4 h-4 text-slate-400 mb-1" />
                <span className="text-[11px] text-slate-300">Upload Photo</span>
                <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
              </label>

              {!webcamActive ? (
                <button
                  onClick={startWebcam}
                  className="flex flex-col items-center justify-center border border-slate-700 hover:border-pink-500/50 rounded-lg p-3 bg-slate-950/40 text-slate-300 hover:text-white transition-all"
                >
                  <Video className="w-4 h-4 text-pink-400 mb-1" />
                  <span className="text-[11px]">Use Webcam</span>
                </button>
              ) : (
                <button
                  onClick={stopWebcam}
                  className="flex flex-col items-center justify-center border border-rose-700/50 rounded-lg p-3 bg-rose-950/30 text-rose-300 transition-all"
                >
                  <VideoOff className="w-4 h-4 mb-1" />
                  <span className="text-[11px]">Stop Cam</span>
                </button>
              )}
            </div>

            {webcamActive && (
              <div className="space-y-2 p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                <div className="aspect-square rounded-lg overflow-hidden bg-black flex items-center justify-center">
                  <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                </div>
                <button
                  onClick={captureWebcam}
                  className="w-full py-1.5 bg-pink-600 hover:bg-pink-500 text-white rounded text-xs font-semibold flex items-center justify-center space-x-1"
                >
                  <Camera className="w-3.5 h-3.5" />
                  <span>Capture Snapshot</span>
                </button>
              </div>
            )}
          </div>

          {/* Style Selector */}
          <div className="space-y-3 pt-3 border-t border-slate-800">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                2. Select FS2K Sketch Style
              </label>
              <Palette className="w-4 h-4 text-pink-400" />
            </div>

            <div className="space-y-2">
              {styles.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedStyle(s.id)}
                  className={`w-full p-3 rounded-lg border text-left transition-all ${
                    selectedStyle === s.id
                      ? 'border-pink-500 bg-pink-950/20 text-white shadow-sm'
                      : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-200">{s.name}: {s.label}</span>
                    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                      selectedStyle === s.id ? 'bg-pink-500/30 text-pink-300' : 'bg-slate-800 text-slate-500'
                    }`}>
                      s = {s.id}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 mt-1">{s.desc}</p>
                </button>
              ))}
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800">
            <button
              onClick={handleGenerateSketch}
              disabled={!photo || loading}
              className="w-full py-2.5 bg-pink-600 hover:bg-pink-500 disabled:opacity-50 text-white rounded-lg text-sm font-semibold flex items-center justify-center space-x-2 shadow-lg shadow-pink-600/30 transition-all"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>{loading ? 'Synthesizing Sketch...' : 'Generate Sketch (Task 4)'}</span>
            </button>
          </div>
        </div>

        {/* Visual Display */}
        <div className="lg:col-span-2 bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Photorealistic Paired Translation
            </span>
            {sketch && photo && (
              <div className="flex bg-slate-950 border border-slate-800 rounded-lg p-0.5 text-xs">
                <button
                  onClick={() => setActiveView('gallery')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    activeView === 'gallery' ? 'bg-pink-600 text-white' : 'text-slate-400'
                  }`}
                >
                  Side-by-Side
                </button>
                <button
                  onClick={() => setActiveView('slider')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    activeView === 'slider' ? 'bg-pink-600 text-white' : 'text-slate-400'
                  }`}
                >
                  Split Slider
                </button>
              </div>
            )}
          </div>

          {activeView === 'slider' && sketch && photo ? (
            <div className="py-4">
              <ImageComparisonSlider
                beforeImage={photo}
                afterImage={sketch}
                beforeLabel="Input Photograph"
                afterLabel={`Synthesized Sketch (${styles.find((s) => s.id === selectedStyle)?.name})`}
              />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <span className="text-xs text-slate-400 font-medium block">Facial Photograph (x)</span>
                <div className="aspect-square rounded-xl border border-slate-800 bg-slate-950/60 overflow-hidden flex items-center justify-center">
                  {photo ? (
                    <img src={photo} alt="Photo" className="w-full h-full object-cover" />
                  ) : (
                    <Eye className="w-8 h-8 text-slate-700" />
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-xs text-slate-400 font-medium block">
                  Synthesized Sketch G(x, s)
                </span>
                <div className="aspect-square rounded-xl border border-slate-800 bg-slate-950/60 overflow-hidden flex items-center justify-center relative group">
                  {sketch ? (
                    <>
                      <img src={sketch} alt="Sketch" className="w-full h-full object-cover bg-white" />
                      <a
                        href={sketch}
                        download={`sketch_style_${selectedStyle}.png`}
                        className="absolute bottom-3 right-3 p-2 rounded-lg bg-slate-900/90 text-white opacity-0 group-hover:opacity-100 transition-all shadow-lg flex items-center space-x-1.5 text-xs"
                      >
                        <Download className="w-4 h-4" />
                        <span>Save PNG</span>
                      </a>
                    </>
                  ) : (
                    <Eye className="w-8 h-8 text-slate-700" />
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
