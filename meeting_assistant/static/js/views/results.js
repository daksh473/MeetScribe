import { state } from '../state.js';
import { esc, formatTime, createEl } from '../utils.js';
import { initTabs } from '../components/tabs.js';
import { getDownloadUrl } from '../api.js';
import { renderDock, seekAudio } from '../components/audio.js';

export function renderResults(container, dockContainer, overlayContainer) {
    try {
        const r = state.result;
    
    // Header
    const html = `
        <div class="card mb-4 flex items-center justify-between">
            <div>
                <h2 style="margin:0">${esc(r.metadata?.file_name || 'Meeting Recording')}</h2>
                <div class="flex gap-2 mt-4 flex-wrap">
                    <span class="chip"><svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="margin-right:4px"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> ${formatTime(r.metadata?.audio_duration_s || 0)}</span>
                    <span class="chip badge-info">Models used: ${r.metadata?.models_used?.length || 0} calls</span>
                    <span class="chip badge-warn">${r.raw_transcript?.uncertain_spans?.length || 0} Uncertainties</span>
                    <span class="chip badge-ok">${r.minutes?.decisions?.length || 0} Decisions</span>
                    <span class="chip badge-crit">${r.minutes?.action_items?.length || 0} Tasks</span>
                </div>
                ${r.raw_transcript?.metadata?.engine_statuses ? `
                <div class="mt-3 text-sm text-muted">
                    <strong>STT Engines:</strong> 
                    ${Object.entries(r.raw_transcript.metadata.engine_statuses).map(([name, status]) => {
                        let shortStatus = status;
                        let detailHtml = '';
                        const match = status.match(/^(.*?)\((.*)\)$/);
                        if (match) {
                            shortStatus = match[1].trim();
                            const fullError = match[2];
                            let readableShort = fullError;
                            if (fullError.includes('free tier') || fullError.includes('free_tier') || fullError.includes('authorization') || fullError.includes('401') || fullError.includes('403')) {
                                readableShort = "account blocked: free tier disabled";
                                shortStatus = "unavailable";
                            } else if (fullError.includes('Provider message:')) {
                                readableShort = fullError.split('Provider message:')[0].trim();
                            } else if (fullError.includes('{')) {
                                readableShort = "API error";
                                shortStatus = "unavailable";
                            }
                            shortStatus = `${shortStatus} (${esc(readableShort)})`;
                            detailHtml = `<details style="margin-top: 4px;"><summary style="cursor:pointer; font-size: 0.85em;">Technical details</summary><pre style="font-size: 0.8em; margin-top: 4px; white-space: pre-wrap;">${esc(fullError)}</pre></details>`;
                        } else {
                            shortStatus = esc(status);
                        }
                        return `<div style="margin-bottom: 8px;"><strong style="color:var(--text)">${esc(name)}:</strong> ${shortStatus}${detailHtml}</div>`;
                    }).join('')}
                </div>
                ` : ''}
            </div>
            
            <div class="dropdown">
                <button class="btn btn-primary" onclick="this.nextElementSibling.classList.toggle('show')">
                    Download ▾
                </button>
                <div class="dropdown-menu">
                    <a class="dropdown-item" href="${getDownloadUrl(state.jobId, 'record_md')}">Minutes (.md)</a>
                    <a class="dropdown-item" href="${getDownloadUrl(state.jobId, 'record_json')}">Structured Data (.json)</a>
                    <a class="dropdown-item" href="${getDownloadUrl(state.jobId, 'raw_txt')}">Raw Transcript (.txt)</a>
                    <a class="dropdown-item" href="${getDownloadUrl(state.jobId, 'refined_txt')}">Refined Transcript (.txt)</a>
                    <a class="dropdown-item" href="${getDownloadUrl(state.jobId, 'uncertainty_md')}">Uncertainty Report (.md)</a>
                    <div class="dropdown-divider"></div>
                    <a class="dropdown-item" href="${getDownloadUrl(state.jobId, 'all_zip')}">Download All (.zip)</a>
                </div>
            </div>
        </div>

        <div class="tabs" role="tablist">
            <button class="tab-btn" role="tab" aria-selected="true" aria-controls="tab-raw">Raw transcript</button>
            <button class="tab-btn" role="tab" aria-selected="false" aria-controls="tab-refined">Refined</button>
            <button class="tab-btn" role="tab" aria-selected="false" aria-controls="tab-min">Minutes</button>
            <button class="tab-btn" role="tab" aria-selected="false" aria-controls="tab-dec">Decisions</button>
            <button class="tab-btn" role="tab" aria-selected="false" aria-controls="tab-act">Action items</button>
            <button class="tab-btn" role="tab" aria-selected="false" aria-controls="tab-unc">Uncertainty report</button>
        </div>

        <div id="tab-raw" class="tab-pane" role="tabpanel"></div>
        <div id="tab-refined" class="tab-pane hidden" role="tabpanel"></div>
        <div id="tab-min" class="tab-pane hidden" role="tabpanel"></div>
        <div id="tab-dec" class="tab-pane hidden" role="tabpanel"></div>
        <div id="tab-act" class="tab-pane hidden" role="tabpanel"></div>
        <div id="tab-unc" class="tab-pane hidden" role="tabpanel"></div>
    `;
    
    container.innerHTML = html;
    initTabs(container);
    
    // Close dropdowns on outside click
    document.addEventListener('click', e => {
        if (!e.target.closest('.dropdown')) {
            container.querySelectorAll('.dropdown-menu').forEach(m => m.classList.remove('show'));
        }
    });

    function safeRender(el, renderFn, fallback = '') {
        try {
            renderFn();
        } catch (e) {
            console.error(e);
            el.innerHTML = `
                <div class="card">
                    <div class="banner error-banner">This section could not be displayed</div>
                    <details style="margin-top: 8px;">
                        <summary style="cursor:pointer; font-size:0.85em; color:var(--muted)">Technical details</summary>
                        <pre style="font-size:0.8em; margin-top:4px; white-space:pre-wrap;">${esc(e.name)}: ${esc(e.message)}</pre>
                    </details>
                    ${fallback}
                </div>
            `;
        }
    }

    safeRender(container.querySelector('#tab-raw'), () => renderRaw(container.querySelector('#tab-raw'), overlayContainer), r.raw_transcript?.raw_text ? `<div class="mt-4"><div class="banner warn-banner mb-2">Displaying raw text fallback</div><p style="white-space:pre-wrap">${esc(r.raw_transcript.raw_text)}</p></div>` : '');
    safeRender(container.querySelector('#tab-refined'), () => renderRefined(container.querySelector('#tab-refined')));
    safeRender(container.querySelector('#tab-min'), () => renderMinutes(container.querySelector('#tab-min')));
    safeRender(container.querySelector('#tab-dec'), () => renderDecisions(container.querySelector('#tab-dec')));
    safeRender(container.querySelector('#tab-act'), () => renderActions(container.querySelector('#tab-act')));
    safeRender(container.querySelector('#tab-unc'), () => renderUncertainty(container.querySelector('#tab-unc')));

    renderDock(dockContainer, state.jobId, r.raw_transcript?.uncertain_spans || []);
    } catch (err) {
        console.error("Critical rendering error:", err);
        container.innerHTML = `
            <div class="card">
                <div class="banner error-banner">The UI encountered a critical error during rendering.</div>
                <details style="margin-top: 8px;">
                    <summary style="cursor:pointer; font-size:0.85em; color:var(--muted)">Technical details</summary>
                    <pre style="font-size:0.8em; margin-top:4px; white-space:pre-wrap;">${esc(err.name)}: ${esc(err.message)}\n${err.stack ? esc(err.stack) : ''}</pre>
                </details>
            </div>
        `;
    }
}

function renderRaw(el, overlay) {
    const segs = state.result.raw_transcript?.segments || [];
    const spans = state.result.raw_transcript?.uncertain_spans || [];
    
    let html = `
        <div class="flex items-center justify-between mb-4">
            <input type="text" id="raw-search" placeholder="Search transcript..." style="width: 300px;">
            <label class="toggle"><input type="checkbox" id="raw-toggle" checked> Show uncertainty highlights</label>
        </div>
        <div class="card" id="raw-content">
    `;
    
    segs.forEach(seg => {
        let text = esc(seg.text);
        
        spans.forEach(span => {
            if (span.start >= seg.start && span.end <= seg.end + 1) {
                let cls = 'hl-low';
                if (span.category === 'CRITICAL') cls = 'hl-crit';
                if (span.category === 'HIGH') cls = 'hl-high';
                
                const spanStr = esc(span.chosen_text || '');
                if (spanStr && text.includes(spanStr)) {
                    const altObj = span.alternatives || {};
                    const alts = esc(Object.entries(altObj).map(([eng, txt]) => `${eng}: ${txt}`).join(' | '));
                    const badgeClass = span.category === 'CRITICAL' ? 'badge-crit' : 'badge-warn';
                    text = text.replace(spanStr, `<span class="hl ${cls}" data-alts="${alts}" data-badge="${span.category}" data-bclass="${badgeClass}" data-start="${span.start}">${spanStr}</span>`);
                }
            }
        });
        
        html += `<div class="utterance">
            <div class="u-meta">${seg.speaker ? `<span class="spk">${esc(seg.speaker)}</span>` : ''}<span class="ts" onclick="window.seekAudio(${seg.start})">${formatTime(seg.start)}</span></div>
            <div>${text}</div>
        </div>`;
    });
    
    html += `</div>`;
    el.innerHTML = html;
    
    // Toggle
    el.querySelector('#raw-toggle').addEventListener('change', e => {
        const hls = el.querySelectorAll('.hl');
        hls.forEach(h => {
            if (e.target.checked) h.style.cssText = '';
            else h.style.cssText = 'background:transparent; border:none;';
        });
    });
    
    // Search
    el.querySelector('#raw-search').addEventListener('input', e => {
        const term = e.target.value.toLowerCase();
        el.querySelectorAll('.utterance').forEach(u => {
            u.style.display = u.textContent.toLowerCase().includes(term) ? 'block' : 'none';
        });
    });

    // Popover overlay
    overlay.innerHTML = `
        <div id="hl-popover" class="popover">
            <div class="flex justify-between items-center mb-2">
                <span id="hl-badge" class="badge"></span>
                <button id="hl-close" class="btn-icon">&times;</button>
            </div>
            <div class="text-sm mb-4">Alternatives: <span id="hl-alts" class="mono text-muted"></span></div>
            <button id="hl-play" class="btn btn-sm w-full">▶ Play this moment</button>
        </div>
    `;
    
    const pop = overlay.querySelector('#hl-popover');
    
    el.querySelectorAll('.hl').forEach(hl => {
        hl.addEventListener('click', e => {
            const rect = hl.getBoundingClientRect();
            pop.style.top = `${rect.bottom + window.scrollY + 8}px`;
            pop.style.left = `${rect.left + window.scrollX}px`;
            
            overlay.querySelector('#hl-badge').textContent = hl.dataset.badge;
            overlay.querySelector('#hl-badge').className = `badge ${hl.dataset.bclass}`;
            overlay.querySelector('#hl-alts').textContent = hl.dataset.alts;
            overlay.querySelector('#hl-play').onclick = () => window.seekAudio(parseFloat(hl.dataset.start));
            
            pop.style.display = 'block';
            e.stopPropagation();
        });
    });
    
    document.addEventListener('click', e => { if (!pop.contains(e.target)) pop.style.display = 'none'; });
    overlay.querySelector('#hl-close').addEventListener('click', () => pop.style.display = 'none');
}

function renderRefined(el) {
    const raw = state.result.raw_transcript?.raw_text || '';
    const resols = state.result.refined_transcript?.resolutions || [];
    const flags = state.result.refined_transcript?.flags || [];
    
    let diffHtml = esc(raw);
    resols.forEach(res => {
        if (res.original_span && diffHtml.includes(esc(res.original_span))) {
            const block = `<span class="diff-del">${esc(res.original_span)}</span> <span class="diff-ins">${esc(res.refined_span)}</span> <span class="chip" style="font-size:10px; margin-left:4px;">${esc(res.reason)}</span>`;
            diffHtml = diffHtml.replace(esc(res.original_span), block);
        }
    });

    let html = `
        ${flags.length ? `<div class="banner warn-banner mb-4">${flags.length} edit(s) rejected by safety guards.</div>` : ''}
        <div class="mb-4">
            <label class="toggle"><input type="checkbox" id="diff-toggle"> Show only changes</label>
        </div>
        <div class="card diff-grid" id="diff-view">
            <div>
                <h3>Diff View</h3>
                <div style="line-height:1.7;">${diffHtml}</div>
            </div>
        </div>
    `;
    el.innerHTML = html;
    
    el.querySelector('#diff-toggle').addEventListener('change', e => {
        const v = el.querySelector('#diff-view');
        if (e.target.checked) v.classList.add('diff-only-mode');
        else v.classList.remove('diff-only-mode');
    });
}

function renderMinutes(el) {
    const m = state.result.minutes || {};
    el.innerHTML = `
        <div class="card mb-4">
            <h3>Executive Summary</h3>
            <p class="text-muted mt-2" style="white-space:pre-wrap;">${esc(m.summary)}</p>
        </div>
        <div class="grid-2">
            ${m.sections ? m.sections.map(s => `
                <div class="card mb-4">
                    <div class="flex items-center gap-2 mb-2">
                        <h3>${esc(s.title)}</h3>
                        ${s.start ? `<span class="chip badge-muted mono">${formatTime(s.start)}</span>` : ''}
                    </div>
                    <ul style="padding-left:1.5rem; color:var(--text); line-height:1.6;" class="mt-2 text-sm">
                        ${s.points.map(p => `<li class="mb-1">${esc(p)}</li>`).join('')}
                    </ul>
                </div>
            `).join('') : ''}
        </div>
    `;
}

// Maps utterance [u0] tags to timestamps
function playEvidence(evs) {
    if (!evs || !evs.length) return;
    const match = evs[0].match(/\[(u\d+)\]/);
    if (match && match[1]) {
        const utt = state.result.refined_transcript?.utterances?.find(u => u.id === match[1]);
        if (utt) window.seekAudio(utt.start);
    }
}

function renderDecisions(el) {
    const decs = state.result.minutes?.decisions || [];
    window.playEvidence = playEvidence; // global for inline onclick
    
    let html = `<div style="display:grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap:1rem;">`;
    if (decs.length === 0) {
        html = `<div class="card text-center text-muted">No decisions were found in this recording.</div>`;
    } else {
        decs.forEach(d => {
            let bCls = 'badge-muted';
            if (d.status === 'AGREED') bCls = 'badge-ok';
            if (d.status === 'REJECTED') bCls = 'badge-crit';
            if (d.status === 'PROPOSED') bCls = 'badge-warn';
            
            html += `
            <div class="card">
                <div class="flex justify-between items-start mb-2">
                    <span class="badge ${bCls}">${esc(d.status)}</span>
                    ${d.low_confidence_audio ? `<span class="badge badge-warn">Low-confidence audio</span>` : ''}
                </div>
                <p class="mb-4 text-sm">${esc(d.text)}</p>
                <div class="flex justify-between items-center text-xs mt-auto pt-4" style="border-top:1px solid var(--border);">
                    <span class="text-muted italic" style="max-width:200px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">"${esc(d.evidence[0] || 'No quote')}"</span>
                    <button class="btn btn-icon" onclick='window.playEvidence(${JSON.stringify(d.evidence)})'>▶ Play</button>
                </div>
            </div>`;
        });
        html += `</div>`;
    }
    el.innerHTML = html;
}

function renderActions(el) {
    const acts = state.result.minutes?.action_items || [];
    
    if (acts.length === 0) {
        el.innerHTML = `<div class="card text-center text-muted">No action items were found.</div>`;
        return;
    }
    
    let html = `
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Task</th>
                        <th>Owner</th>
                        <th>Deadline</th>
                        <th>Status</th>
                        <th>Evidence</th>
                    </tr>
                </thead>
                <tbody>
    `;
    acts.forEach(a => {
        let own = a.owner ? esc(a.owner) : `<span class="chip badge-muted" title="Not mentioned in the recording, so left blank on purpose">Unspecified</span>`;
        let dead = a.deadline ? esc(a.deadline) : `<span class="chip badge-muted" title="Not mentioned in the recording, so left blank on purpose">Unspecified</span>`;
        let stat = a.status === 'CONFIRMED' ? 'badge-ok' : 'badge-warn';
        
        html += `
            <tr>
                <td>${esc(a.task)} ${a.low_confidence_audio ? `<span class="badge badge-warn ml-2">Low Conf</span>` : ''}</td>
                <td>${own}</td>
                <td>${dead}</td>
                <td><span class="badge ${stat}">${esc(a.status)}</span></td>
                <td><button class="btn btn-sm" onclick='window.playEvidence(${JSON.stringify(a.evidence)})'>▶ Play</button></td>
            </tr>
        `;
    });
    html += `</tbody></table></div>`;
    el.innerHTML = html;
}

function renderUncertainty(el) {
    const spans = state.result.raw_transcript?.uncertain_spans || [];
    
    let c = 0, h = 0, l = 0;
    spans.forEach(s => { if(s.category==='CRITICAL') c++; else if(s.category==='HIGH') h++; else l++; });
    
    let html = `
        <div class="flex gap-4 mb-8 justify-center">
            <div class="card text-center flex-col items-center justify-center" style="width:150px;"><div style="font-size:2rem; color:var(--crit);">${c}</div><div class="text-sm text-muted">Critical</div></div>
            <div class="card text-center flex-col items-center justify-center" style="width:150px;"><div style="font-size:2rem; color:var(--warn);">${h}</div><div class="text-sm text-muted">High</div></div>
            <div class="card text-center flex-col items-center justify-center" style="width:150px;"><div style="font-size:2rem; color:var(--text);">${l}</div><div class="text-sm text-muted">Low</div></div>
        </div>
    `;
    
    if (spans.length === 0) {
        html += `<div class="card text-center text-muted">No uncertain spans detected.</div>`;
    } else {
        html += `
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Category</th>
                            <th>Alternatives</th>
                            <th>Resolved As</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        spans.forEach(s => {
            let cls = s.category==='CRITICAL' ? 'badge-crit' : (s.category==='HIGH' ? 'badge-warn' : 'badge-muted');
            html += `
                <tr>
                    <td class="mono text-muted">${formatTime(s.start)}</td>
                    <td><span class="badge ${cls}">${s.category}</span></td>
                    <td class="text-sm">${esc((s.alternatives||[]).join(' | '))}</td>
                    <td class="text-sm text-muted">${esc(s.chosen_text || s.resolved_as || '')}</td>
                    <td><button class="btn btn-sm" onclick="window.seekAudio(${s.start})">▶ Play</button></td>
                </tr>
            `;
        });
        html += `</tbody></table></div>`;
    }
    el.innerHTML = html;
}
