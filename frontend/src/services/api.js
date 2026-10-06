import axios from 'axios';

// Dev (Vite): same-origin + proxy to :8000. Production / direct: port 8000 on current host.
const getBackendBase = () => {
  if (typeof window !== 'undefined' && window.location) {
    const { protocol, hostname, port } = window.location;
    if (port === '5173' || port === '3000') {
      return `${protocol}//${hostname}:${port}`;
    }
    return `${protocol}//${hostname}:8000`;
  }
  return 'http://127.0.0.1:8000';
};

const BACKEND_BASE = getBackendBase();

const api = axios.create({
  baseURL: `${BACKEND_BASE}/api`,
  timeout: 300000,
});

const isRetriableRequestError = (err) => {
  if (!err) return false;
  if (err.code === 'ERR_NETWORK' || err.message === 'Network Error') return true;
  const status = err.response?.status;
  return status === 502 || status === 503 || status === 504;
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const withBackendRetry = async (requestFn, { retries = 6, delayMs = 2000 } = {}) => {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await requestFn();
    } catch (err) {
      lastError = err;
      if (!isRetriableRequestError(err) || attempt >= retries) {
        throw err;
      }
      await sleep(delayMs);
    }
  }
  throw lastError;
};

export const uploadVideo = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await withBackendRetry(() => api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onUploadProgress && progressEvent.total) {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onUploadProgress(percentCompleted);
      }
    },
  }));
  return response.data;
};

export const startAnalysis = async (videoId) => {
  const response = await withBackendRetry(() => api.post(`/analyze/${videoId}`));
  return response.data;
};

export const checkStatus = async (videoId) => {
  const response = await api.get(`/status/${videoId}`);
  return response.data;
};

export const getResult = async (videoId) => {
  const response = await api.get(`/result/${videoId}`);
  return response.data;
};

export const getHistory = async () => {
  const response = await api.get('/history');
  return response.data;
};

export const deleteHistoryItem = async (videoId) => {
  const response = await api.delete(`/history/${videoId}`);
  return response.data;
};

export const analyzeUniversal = async (file, p4Enabled = true, audioMode = 'spoken') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('p4_benford_enabled', p4Enabled);
  formData.append('audio_mode', audioMode);
  const response = await api.post('/pillars/universal', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000 // 5 minutes for long multi-minute audio and multi-pillar analysis
  });
  return response.data;
};

export const analyzeImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/pillars/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const analyzeAudio = async (file, mode = 'spoken') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mode', mode);
  const response = await api.post('/pillars/audio', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const analyzeDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/pillars/document', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const getMediaUrl = (storagePath) => {
  if (!storagePath) return '';
  if (storagePath.startsWith('http://') || storagePath.startsWith('https://')) {
    return storagePath;
  }
  const cleanPath = storagePath.startsWith('/') ? storagePath.slice(1) : storagePath;
  return `${BACKEND_BASE}/storage/${cleanPath}`;
};

export default api;

