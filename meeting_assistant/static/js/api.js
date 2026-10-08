const BASE = '/api/jobs';
const isDemo = () => new URLSearchParams(window.location.search).get('demo') === '1';

export async function uploadJob(file, keywords) {
    const formData = new FormData();
    formData.append('file', file);
    if (keywords) formData.append('keywords', keywords);
    
    // In demo mode, if we wanted to mock the upload, we could intercept here.
    // But since the Python backend supports DEMO_FIXTURE env var, we'll just hit it normally.
    
    const res = await fetch(BASE, { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');
    return data.job_id;
}

export async function pollJob(jobId) {
    const res = await fetch(`${BASE}/${jobId}`);
    if (!res.ok) throw new Error('Polling failed');
    return await res.json();
}

export async function getJobResult(jobId) {
    const res = await fetch(`${BASE}/${jobId}/result`);
    if (!res.ok) throw new Error('Failed to fetch results');
    return await res.json();
}

export function getAudioUrl(jobId) {
    return `${BASE}/${jobId}/audio`;
}

export function getDownloadUrl(jobId, kind) {
    return `${BASE}/${jobId}/download/${kind}`;
}
