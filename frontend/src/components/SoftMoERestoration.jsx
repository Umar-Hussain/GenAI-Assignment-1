import React, { useState } from 'react';
import { Upload, Play, RefreshCw, Zap, Cpu, Sparkles, Sliders, Eye, Download } from 'lucide-react';
import { softMoE, corruptImage } from '../utils/api';
import ImageComparisonSlider from './ImageComparisonSlider';

export default function SoftMoERestoration() {
  const [image, setImage] = useState(null);
  const [restoredImage, setRestoredImage] = useState(null);
  const [routingWeights, setRoutingWeights] = useState(null);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState('gallery');

  const [corruptionType, setCorruptionType] = useState('salt_pepper');

  const branches = [
    { id: 0, name: 'Identity (Clean) Branch', desc: 'Direct skip connection bypass', color: 'from-emerald-500 to-teal-400' },
    { id: 1, name: 'Salt & Pepper Expert', desc: 'Non-linear impulse restoration', color: 'from-amber-500 to-orange-400' },
    { id: 2, name: 'Gaussian Blur Expert', desc: 'High-frequency edge deblurring', color: 'from-cyan-500 to-blue-400' },
    { id: 3, name: 'Occlusion Inpainter', desc: 'Contextual spatial hole reconstruction', color: 'from-rose-500 to-pink-400' },
  ];

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setRestoredImage(null);
        setRoutingWeights(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleApplyCorruption = async () => {
    if (!image) return;
    setLoading(true);
    try {
      const res = await fetch(image);
      const blob = await res.blob();
      const params = {};
      if (corruptionType === 'salt_pepper') params.prob = 0.08;
      if (corruptionType === 'gaussian_blur') {
        params.kernel_size = 5;
        params.sigma = 1.5;
      }
      if (corruptionType === 'occlusion') {
        params.num_rects = 2;
        params.coverage = 0.20;
      }

      const data = await corruptImage(blob, corruptionType, params);
      setImage(`data:image/png;base64,${data.corrupted_image}`);
      setRestoredImage(null);
      setRoutingWeights(null);
    } catch (err) {
      alert(`Corruption failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunMoE = async () => {
    if (!image) return;
    setLoading(true);
    try {
      const res = await fetch(image);
      const blob = await res.blob();
      const data = await softMoE(blob);
      setRestoredImage(`data:image/png;base64,${data.restored_image}`);
      setRoutingWeights(data.routing_weights);
      setInferenceTime(data.inference_time_ms);
    } catch (err) {
      alert(`Soft-MoE failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center space-x-2">
            <span>Soft Mixture-of-Experts Restoration</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20">
              Task 3 (Differentiable MoE)
            </span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Gating network calculates continuous softmax weights across 3 specialist models and an identity clean bypass, enabling smooth blended restoration.
          </p>
        </div>

        {inferenceTime !== null && (
          <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
            <Zap className="w-4 h-4 text-amber-400" />
            <span className="text-slate-400">Total Latency:</span>
            <span className="font-semibold text-emerald-400">{inferenceTime.toFixed(1)} ms</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-5">
          <div className="space-y-3">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              1. Input Image
            </label>
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-lg p-4 cursor-pointer bg-slate-950/40 transition-all">
              <Upload className="w-6 h-6 text-slate-400 mb-1" />
              <span className="text-xs text-slate-300 font-medium">Select Image File</span>
              <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>

          <div className="space-y-3 pt-3 border-t border-slate-800">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              2. Inject Runtime Corruption
            </label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                onClick={() => setCorruptionType('salt_pepper')}
                className={`py-2 px-2.5 rounded-lg border text-left transition-all ${
                  corruptionType === 'salt_pepper' ? 'border-amber-500 bg-amber-500/10 text-amber-300' : 'border-slate-800 bg-slate-950 text-slate-400'
                }`}
              >
                Salt & Pepper
              </button>
              <button
                onClick={() => setCorruptionType('gaussian_blur')}
                className={`py-2 px-2.5 rounded-lg border text-left transition-all ${
                  corruptionType === 'gaussian_blur' ? 'border-cyan-500 bg-cyan-500/10 text-cyan-300' : 'border-slate-800 bg-slate-950 text-slate-400'
                }`}
              >
                Gaussian Blur
              </button>
              <button
                onClick={() => setCorruptionType('occlusion')}
                className={`py-2 px-2.5 rounded-lg border text-left transition-all col-span-2 ${
                  corruptionType === 'occlusion' ? 'border-rose-500 bg-rose-500/10 text-rose-300' : 'border-slate-800 bg-slate-950 text-slate-400'
                }`}
              >
                Rectangular Occlusion
              </button>
            </div>
            <button
              onClick={handleApplyCorruption}
              disabled={!image || loading}
              className="w-full py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 rounded-lg text-xs font-medium flex items-center justify-center space-x-1"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Apply Corruption</span>
            </button>
          </div>

          <div className="pt-3 border-t border-slate-800">
            <button
              onClick={handleRunMoE}
              disabled={!image || loading}
              className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-lg text-sm font-semibold flex items-center justify-center space-x-2 shadow-lg shadow-purple-600/30 transition-all"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>{loading ? 'Executing Soft MoE...' : 'Run Soft MoE (Task 3)'}</span>
            </button>
          </div>
        </div>

        {/* Routing Weights & Inspection */}
        <div className="lg:col-span-2 space-y-6">
          {/* Continuous Expert Weights */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
                <Cpu className="w-4 h-4 text-purple-400" />
                <span>Gating Network Routing Weights &omega; (Continuous Blend)</span>
              </span>
              <span className="text-xs text-slate-500 font-mono">&tau; = 1.0</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {branches.map((b) => {
                const weight = routingWeights ? routingWeights[b.id] || 0 : 0.25;
                const isDominant = routingWeights && Math.max(...routingWeights) === weight && weight > 0.35;

                return (
                  <div
                    key={b.id}
                    className={`p-3.5 rounded-lg border transition-all ${
                      isDominant
                        ? 'border-purple-500/60 bg-purple-950/20 shadow-md shadow-purple-500/10'
                        : 'border-slate-800/80 bg-slate-950/40'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center space-x-1.5">
                        <span className="text-xs font-semibold text-slate-200">{b.name}</span>
                        {isDominant && (
                          <span className="flex items-center space-x-1 text-[10px] text-purple-300 font-semibold px-1.5 py-0.5 rounded bg-purple-500/20 border border-purple-500/30">
                            <Sparkles className="w-3 h-3" />
                            <span>Dominant</span>
                          </span>
                        )}
                      </div>
                      <span className="font-mono text-xs font-bold text-slate-300">
                        {(weight * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                      <div
                        className={`h-full bg-gradient-to-r ${b.color} transition-all duration-500`}
                        style={{ width: `${weight * 100}%` }}
                      />
                    </div>

                    <p className="text-[11px] text-slate-500 mt-1">{b.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Visual Outputs */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Visual Inspection
              </span>
              {restoredImage && image && (
                <div className="flex bg-slate-950 border border-slate-800 rounded-lg p-0.5 text-xs">
                  <button
                    onClick={() => setActiveView('gallery')}
                    className={`px-2.5 py-1 rounded-md transition-all ${
                      activeView === 'gallery' ? 'bg-purple-600 text-white' : 'text-slate-400'
                    }`}
                  >
                    Side-by-Side
                  </button>
                  <button
                    onClick={() => setActiveView('slider')}
                    className={`px-2.5 py-1 rounded-md transition-all ${
                      activeView === 'slider' ? 'bg-purple-600 text-white' : 'text-slate-400'
                    }`}
                  >
                    Split Comparison
                  </button>
                </div>
              )}
            </div>

            {activeView === 'slider' && restoredImage && image ? (
              <div className="py-2">
                <ImageComparisonSlider
                  beforeImage={image}
                  afterImage={restoredImage}
                  beforeLabel="Degraded Input"
                  afterLabel="Soft MoE Blended Output"
                />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <span className="text-xs text-slate-400 font-medium block">Input Image</span>
                  <div className="aspect-square rounded-lg border border-slate-800 bg-slate-950/60 overflow-hidden flex items-center justify-center">
                    {image ? (
                      <img src={image} alt="Input" className="w-full h-full object-cover" />
                    ) : (
                      <Eye className="w-6 h-6 text-slate-700" />
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <span className="text-xs text-slate-400 font-medium block">
                    Soft MoE Blended Output
                  </span>
                  <div className="aspect-square rounded-lg border border-slate-800 bg-slate-950/60 overflow-hidden flex items-center justify-center relative group">
                    {restoredImage ? (
                      <>
                        <img src={restoredImage} alt="Restored" className="w-full h-full object-cover" />
                        <a
                          href={restoredImage}
                          download="soft_moe_restored.png"
                          className="absolute bottom-2 right-2 p-1.5 rounded-md bg-slate-900/90 text-white opacity-0 group-hover:opacity-100 transition-all shadow"
                        >
                          <Download className="w-3.5 h-3.5" />
                        </a>
                      </>
                    ) : (
                      <Eye className="w-6 h-6 text-slate-700" />
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
