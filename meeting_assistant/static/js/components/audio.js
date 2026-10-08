import { getAudioUrl } from '../api.js';
import { formatTime } from '../utils.js';

let audioEl = null;
let playBtn = null;
let fill = null;
let thumb = null;
let timeCur = null;

export function renderDock(container, jobId, uncertainSpans = []) {
    container.innerHTML = `
        <div class="player-spacer"></div>
        <div class="player-dock">
            <audio id="audio-el" src="${getAudioUrl(jobId)}"></audio>
            <button id="dock-play" class="btn-icon" style="background:var(--primary);color:#fff;border-radius:50%;width:36px;height:36px;">▶</button>
            <span id="dock-cur" class="mono text-xs">0:00</span>
            <div id="dock-seek" class="seek-bar-wrapper">
                <div class="seek-track">
                    <div id="dock-fill" class="seek-fill"></div>
                    <div id="dock-thumb" class="seek-thumb" style="left:0%"></div>
                    <!-- Markers -->
                    ${uncertainSpans.map(s => {
                        // We will position these after metadata loads
                        const color = s.category === 'CRITICAL' ? 'var(--crit)' : (s.category === 'HIGH' ? 'var(--warn)' : 'var(--muted)');
                        return `<div class="seek-marker" data-start="${s.start}" style="background:${color};"></div>`;
                    }).join('')}
                </div>
            </div>
            <span id="dock-tot" class="mono text-xs">0:00</span>
            <select id="dock-speed" class="speed-select">
                <option value="1">1x</option>
                <option value="1.25">1.25x</option>
                <option value="1.5">1.5x</option>
            </select>
        </div>
    `;

    audioEl = container.querySelector('#audio-el');
    playBtn = container.querySelector('#dock-play');
    fill = container.querySelector('#dock-fill');
    thumb = container.querySelector('#dock-thumb');
    timeCur = container.querySelector('#dock-cur');
    const timeTot = container.querySelector('#dock-tot');
    const seekWrap = container.querySelector('#dock-seek');
    const speedSel = container.querySelector('#dock-speed');
    const markers = container.querySelectorAll('.seek-marker');

    audioEl.addEventListener('loadedmetadata', () => {
        timeTot.textContent = formatTime(audioEl.duration);
        markers.forEach(m => {
            const pct = (parseFloat(m.dataset.start) / audioEl.duration) * 100;
            m.style.left = `${pct}%`;
        });
    });

    audioEl.addEventListener('timeupdate', () => {
        const pct = (audioEl.currentTime / audioEl.duration) * 100;
        fill.style.width = `${pct}%`;
        thumb.style.left = `${pct}%`;
        timeCur.textContent = formatTime(audioEl.currentTime);
    });

    playBtn.addEventListener('click', () => {
        if (audioEl.paused) { audioEl.play(); playBtn.textContent = '⏸'; }
        else { audioEl.pause(); playBtn.textContent = '▶'; }
    });

    seekWrap.addEventListener('click', (e) => {
        const rect = seekWrap.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        audioEl.currentTime = pct * audioEl.duration;
    });

    speedSel.addEventListener('change', () => {
        audioEl.playbackRate = parseFloat(speedSel.value);
    });
}

export function seekAudio(secs) {
    if (!audioEl) return;
    const t = Math.max(0, secs - 1);
    audioEl.currentTime = t;
    audioEl.play();
    if(playBtn) playBtn.textContent = '⏸';
}
