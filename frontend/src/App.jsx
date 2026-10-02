import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import UniversalRestoration from './components/UniversalRestoration';
import HardRoutedRestoration from './components/HardRoutedRestoration';
import SoftMoERestoration from './components/SoftMoERestoration';
import FaceToSketch from './components/FaceToSketch';
import BenchmarkStats from './components/BenchmarkStats';
import { checkHealth } from './utils/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('task1');
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    const ping = async () => {
      const res = await checkHealth();
      setBackendStatus(res.status === 'ok' ? 'ok' : 'offline');
    };
    ping();
    const interval = setInterval(ping, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col text-slate-100">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} backendStatus={backendStatus} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'task1' && <UniversalRestoration />}
        {activeTab === 'task2' && <HardRoutedRestoration />}
        {activeTab === 'task3' && <SoftMoERestoration />}
        {activeTab === 'task4' && <FaceToSketch />}
        {activeTab === 'benchmarks' && <BenchmarkStats backendStatus={backendStatus} />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/60 py-4 text-center text-xs text-slate-500">
        <p>GenAI Multi-Task Restoration & Synthesis System | Oxford-IIIT Pet & FS2K Benchmark Suite</p>
      </footer>
    </div>
  );
}
