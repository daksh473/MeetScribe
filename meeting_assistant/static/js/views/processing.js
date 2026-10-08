import { pollJob, getJobResult } from '../api.js';
import { setJobId, setResult } from '../state.js';
import { esc } from '../utils.js';
import { renderResults } from './results.js';

let interval = null;
let seenMessages = new Set();

export function renderProcessing(container, jobId) {
    setJobId(jobId);
    seenMessages.clear();
    
    container.innerHTML = `
        <div class="processing-box card" id="proc-card">
            <h2 class="text-center mb-4">Processing Meeting</h2>
            <div id="proc-error" class="banner error-banner hidden"></div>
            <div id="proc-warnings"></div>
            
            <div class="progress-track">
                <div id="prog-fill" class="progress-fill"></div>
            </div>
            <div class="text-center text-sm mono mb-8" id="prog-text">0% - Initializing...</div>
            
            <div class="stepper">
                <div class="stepper-item" id="step-queued"><div class="stepper-dot"></div> Validating</div>
                <div class="stepper-item" id="step-transcribing"><div class="stepper-dot"></div> Transcribing</div>
                <div class="stepper-item" id="step-consensus"><div class="stepper-dot"></div> Building Consensus</div>
                <div class="stepper-item" id="step-refining"><div class="stepper-dot"></div> Refining Transcript</div>
                <div class="stepper-item" id="step-documenting"><div class="stepper-dot"></div> Extracting Dialogue Acts</div>
                <div class="stepper-item" id="step-compiling"><div class="stepper-dot"></div> Compiling Ledger</div>
                <div class="stepper-item" id="step-verifying"><div class="stepper-dot"></div> Verifying Evidence</div>
                <div class="stepper-item" id="step-finalizing"><div class="stepper-dot"></div> Drafting Final Minutes</div>
            </div>
            
            <details class="mt-4" id="log-details">
                <summary style="cursor: pointer; font-size: 0.875rem; color: var(--muted);">Activity Log</summary>
                <div id="activity-log" class="log-box"></div>
            </details>
            
            <div class="mt-8 flex justify-center hidden" id="retry-box">
                <button class="btn" onclick="window.location.hash='#/upload'">Try again</button>
            </div>
        </div>
        <div id="partial-results-container"></div>
    `;

    if (interval) clearInterval(interval);
    
    interval = setInterval(async () => {
        try {
            const data = await pollJob(jobId);
            
            if (data.demo) {
                const b = document.getElementById('demo-banner');
                if (b) b.classList.remove('hidden');
            }
            
            updateUI(container, data);
            
            if (data.status === 'done') {
                clearInterval(interval);
                const result = await getJobResult(jobId);
                setResult(result);
                window.location.hash = `#/results`; // Let router handle the switch
            } else if (data.status === 'failed') {
                clearInterval(interval);
                container.querySelector('#proc-error').textContent = data.error;
                container.querySelector('#proc-error').classList.remove('hidden');
                container.querySelector('#retry-box').classList.remove('hidden');
                
                // Fetch partial results if they exist
                try {
                    const result = await getJobResult(jobId);
                    if (result && result.raw_transcript) {
                        setResult(result);
                        const partialContainer = container.querySelector('#partial-results-container');
                        partialContainer.classList.add('mt-8');
                        const dock = document.getElementById('dock');
                        const overlay = document.getElementById('overlay');
                        renderResults(partialContainer, dock, overlay);
                    }
                } catch(err) {
                    console.error("No partial results available", err);
                }
            }
        } catch (e) {
            console.error(e);
        }
    }, 1500);
}

function updateUI(container, data) {
    const fill = container.querySelector('#prog-fill');
    const txt = container.querySelector('#prog-text');
    const p = Math.round(data.progress * 100);
    fill.style.width = `${p}%`;
    txt.textContent = `${p}% - ${data.stage_label || data.stage}`;
    
    const steps = container.querySelectorAll('.stepper-item');
    steps.forEach(s => s.classList.remove('active', 'failed'));
    
    const currentStep = container.querySelector(`#step-${data.stage}`);
    if (currentStep) {
        currentStep.classList.add('active');
        if (data.status === 'failed') {
            currentStep.classList.add('failed');
            currentStep.style.color = 'var(--crit)';
            const dot = currentStep.querySelector('.stepper-dot');
            if (dot) {
                dot.style.background = 'var(--crit)';
                dot.style.borderColor = 'var(--crit)';
            }
            const logBox = container.querySelector('#log-details');
            if (logBox) logBox.open = true;
        }
        let prev = currentStep.previousElementSibling;
        while(prev) {
            prev.classList.add('done');
            prev = prev.previousElementSibling;
        }
    }
    
    if (data.messages && data.messages.length > 0) {
        const log = container.querySelector('#activity-log');
        data.messages.forEach(msg => {
            if (!seenMessages.has(msg)) {
                seenMessages.add(msg);
                const entry = document.createElement('div');
                entry.textContent = `> ${msg}`;
                log.appendChild(entry);
            }
        });
        log.scrollTop = log.scrollHeight;
    }
    
    if (data.warnings && data.warnings.length) {
        const wBox = container.querySelector('#proc-warnings');
        wBox.innerHTML = data.warnings.map(w => `<div class="banner warn-banner">${esc(w)}</div>`).join('');
    }
}
