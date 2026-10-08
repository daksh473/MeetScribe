import { renderUpload } from './views/upload.js';
import { renderProcessing } from './views/processing.js';
import { renderResults } from './views/results.js';
import { getJobResult } from './api.js';
import { setResult, setJobId } from './state.js';

const appRoot = document.getElementById('app-root');
const audioDock = document.getElementById('audio-dock-container');
const overlay = document.getElementById('overlay-container');
const topBar = document.getElementById('top-bar');

async function handleRoute() {
    let hash = window.location.hash.slice(1);
    
    if (hash.startsWith('/job/')) {
        const jobId = hash.split('/')[2];
        topBar.classList.remove('hidden');
        
        try {
            // First check if it's done by attempting to get the result
            const res = await getJobResult(jobId);
            setJobId(jobId);
            setResult(res);
            renderResults(appRoot, audioDock, overlay);
            
            if (res.demo) {
                document.getElementById('demo-banner').classList.remove('hidden');
            }
        } catch (e) {
            // Not done yet, show processing
            audioDock.innerHTML = '';
            overlay.innerHTML = '';
            renderProcessing(appRoot, jobId);
        }
    } 
    else if (hash === '/results') {
        // Transition from processing
        topBar.classList.remove('hidden');
        renderResults(appRoot, audioDock, overlay);
    }
    else {
        // Default to upload
        topBar.classList.add('hidden');
        audioDock.innerHTML = '';
        overlay.innerHTML = '';
        renderUpload(appRoot);
    }
}

window.addEventListener('hashchange', handleRoute);

// Init
document.getElementById('nav-new-meeting').addEventListener('click', () => {
    window.location.hash = '#/upload';
});

if (!window.location.hash || window.location.hash === '#/') {
    window.location.hash = '#/upload';
} else {
    handleRoute();
}

// Global hook for popovers etc.
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.popover, .drawer, .overlay').forEach(el => el.style.display = 'none');
    }
});
