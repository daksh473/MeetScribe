import { uploadJob } from '../api.js';

export function renderUpload(container) {
    container.innerHTML = `
        <div class="upload-hero">
            <h1>Turn any meeting into a trustworthy record.</h1>
        </div>
        <div class="card" style="max-width: 600px; margin: 0 auto;">
            <div id="upload-error" class="banner error-banner hidden"></div>
            
            <label id="dropzone" class="dropzone flex-col items-center justify-center">
                <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" style="color: var(--muted); margin-bottom: 1rem;"><path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                <div style="font-weight: 500; font-size: 1.1rem; margin-bottom: 0.5rem;">Click or drag to upload</div>
                <div class="text-sm text-muted">Supported formats: MP3, WAV, M4A, MP4, OGG, FLAC</div>
                <input type="file" id="file-input" accept=".mp3,.wav,.m4a,.mp4,.ogg,.flac">
            </label>
            
            <div id="file-info" class="hidden mt-4 p-4 items-center justify-between" style="background: var(--bg); border: 1px solid var(--border); border-radius: 8px; display: flex;">
                <span id="file-name" class="font-medium"></span>
                <button id="clear-file" class="btn-icon">&times;</button>
            </div>

            <details class="mt-4" style="border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem 1rem;">
                <summary style="cursor: pointer; font-size: 0.875rem; color: var(--text);">Advanced options</summary>
                <div class="mt-4 flex-col gap-2">
                    <label class="text-sm text-muted">Domain keywords (comma separated)</label>
                    <input type="text" id="keywords" placeholder="e.g. Kubernetes, Jira, Acme Corp">
                </div>
            </details>

            <div class="mt-8 flex justify-center">
                <button id="start-btn" class="btn btn-primary" disabled style="width: 100%; padding: 0.75rem; font-size: 1rem;">Start processing</button>
            </div>
        </div>
        
        <div class="step-strip">
            <div><span>1</span> Transcribe</div>
            <div><span>2</span> Refine terminology</div>
            <div><span>3</span> Generate minutes & tasks</div>
        </div>
    `;

    const dropzone = container.querySelector('#dropzone');
    const input = container.querySelector('#file-input');
    const errorEl = container.querySelector('#upload-error');
    const fileInfo = container.querySelector('#file-info');
    const fileName = container.querySelector('#file-name');
    const clearBtn = container.querySelector('#clear-file');
    const startBtn = container.querySelector('#start-btn');
    const keywordsInput = container.querySelector('#keywords');

    let selectedFile = null;

    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('dragover'); });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        input.value = '';
        fileInfo.classList.add('hidden');
        fileInfo.style.display = 'none';
        dropzone.style.display = 'flex';
        startBtn.disabled = true;
    });

    function handleFile(file) {
        errorEl.classList.add('hidden');
        const exts = ['mp3', 'wav', 'm4a', 'mp4', 'ogg', 'flac'];
        const ext = file.name.split('.').pop().toLowerCase();
        
        if (!exts.includes(ext)) {
            errorEl.textContent = 'Invalid file format. Please upload a supported audio file.';
            errorEl.classList.remove('hidden');
            return;
        }
        
        selectedFile = file;
        fileName.textContent = file.name;
        dropzone.style.display = 'none';
        fileInfo.classList.remove('hidden');
        fileInfo.style.display = 'flex';
        startBtn.disabled = false;
    }

    startBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        startBtn.disabled = true;
        startBtn.textContent = 'Uploading...';
        
        try {
            const jobId = await uploadJob(selectedFile, keywordsInput.value);
            window.location.hash = `#/job/${jobId}`;
        } catch (err) {
            errorEl.textContent = err.message;
            errorEl.classList.remove('hidden');
            startBtn.disabled = false;
            startBtn.textContent = 'Start processing';
        }
    });
}
