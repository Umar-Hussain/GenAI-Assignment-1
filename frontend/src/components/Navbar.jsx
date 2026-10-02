import React from 'react';
import { Layers, GitBranch, Cpu, Palette, BarChart3, Activity } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, backendStatus }) {
  const tabs = [
    { id: 'task1', label: 'Universal AE', icon: Layers, subtitle: 'Task 1: Denoising Autoencoder' },
    { id: 'task2', label: 'Hard-Routed', icon: GitBranch, subtitle: 'Task 2: Specialist Routing' },
    { id: 'task3', label: 'Soft MoE', icon: Cpu, subtitle: 'Task 3: Mixture-of-Experts' },
    { id: 'task4', label: 'Face-to-Sketch', icon: Palette, subtitle: 'Task 4: Conditional GAN' },
    { id: 'benchmarks', label: 'Telemetry & Benchmarks', icon: BarChart3, subtitle: 'System Performance' },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="font-bold text-lg bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                GenAI Suite
              </span>
              <span className="ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                v1.0
              </span>
            </div>
          </div>

          <nav className="flex space-x-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="flex items-center space-x-3 text-xs">
            <span className="text-slate-400">Backend:</span>
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus === 'ok' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'
                }`}
              />
              <span className={backendStatus === 'ok' ? 'text-emerald-400 font-medium' : 'text-rose-400 font-medium'}>
                {backendStatus === 'ok' ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
