import axios from 'axios';

// Base API configuration (Vite proxy forwards /api and /storage to backend)
const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
});

export const uploadVideo = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onUploadProgress && progressEvent.total) {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onUploadProgress(percentCompleted);
      }
    },
  });
  return response.data;
};

export const startAnalysis = async (videoId) => {
  const response = await api.post(`/analyze/${videoId}`);
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
  return `/storage/${cleanPath}`;
};

export default api;

