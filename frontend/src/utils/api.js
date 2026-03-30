const API_BASE = '/api';

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Analysis failed');
  }

  return res.json();
}

export async function augmentImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/augment`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error('Augmentation failed');
  }

  return res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  return res.json();
}

export async function healthCheck() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export function blobFromDataUrl(dataUrl) {
  const parts = dataUrl.split(',');
  const mime = parts[0].match(/:(.*?);/)[1];
  const b64 = atob(parts[1]);
  const arr = new Uint8Array(b64.length);
  for (let i = 0; i < b64.length; i++) arr[i] = b64.charCodeAt(i);
  return new Blob([arr], { type: mime });
}
