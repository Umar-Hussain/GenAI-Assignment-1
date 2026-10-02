import React, { useState } from 'react';
import { Upload, Play, RefreshCw, Zap, Sliders, Eye, Download } from 'lucide-react';
import { universalRestore, corruptImage } from '../utils/api';
import ImageComparisonSlider from './ImageComparisonSlider';

export default function UniversalRestoration() {
  const [cleanImage, setCleanImage] = useState(null);
  const [corruptedImage, setCorruptedImage] = useState(null);
  const [restoredImage, setRestoredImage] = useState(null);
  const [errorMap, setErrorMap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [activeView, setActiveView] = useState('grid');

  // Corruption controls
  const [corruptionType, setCorruptionType] = useState('salt_pepper');
  const [saltProb, setSaltProb] = useState(0.08);
  const [blurKernel, setBlurKernel] = useState(5);
  const [blurSigma, setBlurSigma] = useState(1.5);
  const [occlusionRects, setOcclusionRects] = useState(2);
  const [occlusionCoverage, setOcclusionCoverage] = useState(0.20);

  const sampleImages = [
    { name: 'Abyssinian Cat', url: 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=256&h=256&fit=crop' },
    { name: 'Golden Retriever', url: 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=256&h=256&fit=crop' },
    { name: 'Pug Dog', url: 'https://images.unsplash.com/photo-1517849845537-4d257902454a?w=256&h=256&fit=crop' },
  ];

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setCleanImage(event.target.result);
        setCorruptedImage(null);
        setRestoredImage(null);
        setErrorMap(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSelectSample = async (url) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onload = (e) => {
        setCleanImage(e.target.result);
        setCorruptedImage(null);
        setRestoredImage(null);
        setErrorMap(null);
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error('Failed to load sample image', err);
    }
  };

  const handleApplyCorruption = async () => {
    if (!cleanImage) return;
    setLoading(true);
    try {
      const res = await fetch(cleanImage);
      const blob = await res.blob();
      const params = {};
      if (corruptionType === 'salt_pepper') params.prob = saltProb;
      if (corruptionType === 'gaussian_blur') {
        params.kernel_size = blurKernel;
        params.sigma = blurSigma;
      }
      if (corruptionType === 'occlusion') {
        params.num_rects = occlusionRects;
        params.coverage = occlusionCoverage;
      }

      const data = await corruptImage(blob, corruptionType, params);
      setCorruptedImage(`data:image/png;base64,${data.corrupted_image}`);
      setRestoredImage(null);
      setErrorMap(null);
    } catch (err) {
      alert(`Corruption failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunRestoration = async () => {
    const targetImage = corruptedImage || cleanImage;
    if (!targetImage) return;
    setLoading(true);
    try {
      const res = await fetch(targetImage);
      const blob = await res.blob();
      const data = await universalRestore(blob);
      setRestoredImage(`data:image/png;base64,${data.restored_image}`);
      setInferenceTime(data.inference_time_ms);

      // Generate Error Map (residual between clean and restored)
      if (cleanImage) {
        generateErrorMapCanvas(cleanImage, `data:image/png;base64,${data.restored_image}`);
      }
    } catch (err) {
      alert(`Restoration failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const generateErrorMapCanvas = (img1Src, img2Src) => {
    const img1 = new Image();
    const img2 = new Image();
    let loaded = 0;
    const onBothLoaded = () => {
      loaded += 1;
      if (loaded < 2) return;
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img1, 0, 0, 128, 128);
      const d1 = ctx.getImageData(0, 0, 128, 128);
      ctx.drawImage(img2, 0, 0, 128, 128);
      const d2 = ctx.getImageData(0, 0, 128, 128);

      const out = ctx.createImageData(128, 128);
      for (let i = 0; i < d1.data.length; i += 4) {
        const diffR = Math.abs(d1.data[i] - d2.data[i]);
        const diffG = Math.abs(d1.data[i + 1] - d2.data[i + 1]);
        const diffB = Math.abs(d1.data[i + 2] - d2.data[i + 2]);
        const diff = (diffR + diffG + diffB) / 3;
        // Heatmap color map: yellow-red
        out.data[i] = Math.min(255, diff * 3.5);
        out.data[i + 1] = Math.min(255, diff * 1.5);
        out.data[i + 2] = Math.max(0, 50 - diff);
        out.data[i + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
      setErrorMap(canvas.toDataURL());
    };
    img1.onload = onBothLoaded;
    img2.onload = onBothLoaded;
    img1.src = img1Src;
    img2.src = img2Src;
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center space-x-2">
            <span>Universal Restoration</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
              Task 1 (Autoencoder)
            </span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Single shared convolutional bottleneck autoencoder capable of reconstructing clean images from all corruptions.
          </p>
        </div>

        {inferenceTime !== null && (
          <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
            <Zap className="w-4 h-4 text-amber-400" />
            <span className="text-slate-400">Inference Latency:</span>
            <span className="font-semibold text-emerald-400">{inferenceTime.toFixed(1)} ms</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Card */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-5">
          <div className="space-y-3">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              1. Select or Upload Image
            </label>
            <div className="grid grid-cols-3 gap-2">
              {sampleImages.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSelectSample(s.url)}
                  className="group relative rounded-lg overflow-hidden border border-slate-700 hover:border-indigo-500 transition-all aspect-square"
                >
                  <img src={s.url} alt={s.name} className="w-full h-full object-cover group-hover:scale-105 transition-all" />
                  <span className="absolute bottom-0 inset-x-0 bg-slate-950/80 text-[10px] py-0.5 text-center text-slate-300 truncate px-1">
                    {s.name}
                  </span>
                </button>
              ))}
            </div>

            <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-lg p-3 cursor-pointer bg-slate-950/40 transition-all">
              <Upload className="w-5 h-5 text-slate-400 mb-1" />
              <span className="text-xs text-slate-300">Upload Custom Image</span>
              <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>

          <div className="space-y-3 pt-3 border-t border-slate-800">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                2. Runtime Corruption Pipeline
              </label>
              <Sliders className="w-4 h-4 text-slate-500" />
            </div>

            <select
              value={corruptionType}
              onChange={(e) => setCorruptionType(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="clean">Clean (No artificial corruption)</option>
              <option value="salt_pepper">Salt & Pepper Noise</option>
              <option value="gaussian_blur">Gaussian Blur</option>
              <option value="occlusion">Rectangular Occlusion</option>
            </select>

            {corruptionType === 'salt_pepper' && (
              <div className="space-y-1 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span>Corruption Probability (p)</span>
                  <span className="font-semibold text-indigo-400">{saltProb}</span>
                </div>
                <input
                  type="range"
                  min="0.02"
                  max="0.15"
                  step="0.01"
                  value={saltProb}
                  onChange={(e) => setSaltProb(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>
            )}

            {corruptionType === 'gaussian_blur' && (
              <div className="space-y-2 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span>Kernel Size:</span>
                  <div className="space-x-1">
                    {[3, 5, 7].map((k) => (
                      <button
                        key={k}
                        onClick={() => setBlurKernel(k)}
                        className={`px-2 py-0.5 rounded text-[11px] ${
                          blurKernel === k ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {k}x{k}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Sigma (&sigma;):</span>
                  <span className="font-semibold text-indigo-400">{blurSigma}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2.5"
                  step="0.1"
                  value={blurSigma}
                  onChange={(e) => setBlurSigma(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>
            )}

            {corruptionType === 'occlusion' && (
              <div className="space-y-2 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span>Rectangles (1-3):</span>
                  <div className="space-x-1">
                    {[1, 2, 3].map((r) => (
                      <button
                        key={r}
                        onClick={() => setOcclusionRects(r)}
                        className={`px-2 py-0.5 rounded text-[11px] ${
                          occlusionRects === r ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {r}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Area Coverage:</span>
                  <span className="font-semibold text-indigo-400">{(occlusionCoverage * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.35"
                  step="0.05"
                  value={occlusionCoverage}
                  onChange={(e) => setOcclusionCoverage(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>
            )}

            <button
              onClick={handleApplyCorruption}
              disabled={!cleanImage || loading}
              className="w-full py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 rounded-lg text-xs font-medium flex items-center justify-center space-x-1 transition-all"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Apply Artificial Corruption</span>
            </button>
          </div>

          <div className="pt-3 border-t border-slate-800">
            <button
              onClick={handleRunRestoration}
              disabled={(!cleanImage && !corruptedImage) || loading}
              className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-semibold flex items-center justify-center space-x-2 shadow-lg shadow-indigo-600/30 transition-all"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>{loading ? 'Running Inference...' : 'Restore Image (Task 1)'}</span>
            </button>
          </div>
        </div>

        {/* Visual Inspection Display */}
        <div className="lg:col-span-2 bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Visual Inspection & Error Analysis
            </span>
            {restoredImage && corruptedImage && (
              <div className="flex bg-slate-950 border border-slate-800 rounded-lg p-0.5 text-xs">
                <button
                  onClick={() => setActiveView('grid')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    activeView === 'grid' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                  }`}
                >
                  Gallery View
                </button>
                <button
                  onClick={() => setActiveView('slider')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    activeView === 'slider' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                  }`}
                >
                  Interactive Split Slider
                </button>
              </div>
            )}
          </div>

          {activeView === 'slider' && restoredImage && corruptedImage ? (
            <div className="py-4">
              <ImageComparisonSlider
                beforeImage={corruptedImage}
                afterImage={restoredImage}
                beforeLabel="Corrupted Input"
                afterLabel="Universal Restored"
              />
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <span className="text-xs text-slate-400 block font-medium">Clean Target</span>
                <div className="aspect-square rounded-lg border border-slate-800 bg-slate-950/60 overflow-hidden flex items-center justify-center">
                  {cleanImage ? (
                    <img src={cleanImage} alt="Clean" className="w-full h-full object-cover" />
                  ) : (
                    <Eye className="w-6 h-6 text-slate-700" />
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-xs text-slate-400 block font-medium">Corrupted Input</span>
                <div className="aspect-square rounded-lg border border-slate-800 bg-slate-950/60 overflow-hidden flex items-center justify-center">
                  {corruptedImage ? (
                    <img src={corruptedImage} alt="Corrupted" className="w-full h-full object-cover" />
                  ) : cleanImage ? (
                    <span className="text-xs text-slate-600 px-2 text-center">Click 'Apply Corruption'</span>
                  ) : (
                    <Eye className="w-6 h-6 text-slate-700" />
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-xs text-slate-400 block font-medium">Reconstructed Output</span>
                <div className="aspect-square rounded-lg border border-slate-800 bg-slate-950/60 overflow-hidden flex items-center justify-center relative group">
                  {restoredImage ? (
                    <>
                      <img src={restoredImage} alt="Restored" className="w-full h-full object-cover" />
                      <a
                        href={restoredImage}
                        download="universal_restored.png"
                        className="absolute bottom-2 right-2 p-1.5 rounded-md bg-slate-900/90 text-white opacity-0 group-hover:opacity-100 transition-all shadow"
                        title="Download Image"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </a>
                    </>
                  ) : (
                    <Eye className="w-6 h-6 text-slate-700" />
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-xs text-slate-400 block font-medium">Absolute Error Map</span>
                <div className="aspect-square rounded-lg border border-slate-800 bg-slate-950/60 overflow-hidden flex items-center justify-center">
                  {errorMap ? (
                    <img src={errorMap} alt="Error Map" className="w-full h-full object-cover" />
                  ) : (
                    <span className="text-xs text-slate-600 px-2 text-center">Computed post-inference</span>
                  )}
                </div>
              </div>
            </div>
          )}

          {errorMap && (
            <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/80">
              <span>Error Heatmap Calibration:</span>
              <div className="flex items-center space-x-2">
                <span>0.0 (Zero Error)</span>
                <div className="w-24 h-2 rounded bg-gradient-to-r from-black via-amber-600 to-yellow-300" />
                <span>Max Error</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
