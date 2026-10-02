const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return await res.json();
  } catch (err) {
    return { status: 'offline', error: err.message };
  }
}

export async function universalRestore(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/universal-restore`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function hardRoute(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/hard-route`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function softMoE(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/soft-moe`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function faceToSketch(file, style) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('style', style.toString());
  const res = await fetch(`${API_BASE}/api/face-to-sketch`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function corruptImage(file, corruptionType, params = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('corruption_type', corruptionType);
  formData.append('params', JSON.stringify(params));
  const res = await fetch(`${API_BASE}/api/corrupt`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
