export function esc(str) {
    if (str == null) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

export function formatTime(secs) {
    if (secs == null || isNaN(secs)) return '--:--';
    const s = Math.floor(secs);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const rs = s % 60;
    if (h > 0) {
        return `${h}:${m < 10 ? '0' : ''}${m}:${rs < 10 ? '0' : ''}${rs}`;
    }
    return `${m}:${rs < 10 ? '0' : ''}${rs}`;
}

export function createEl(tag, className = '', html = '') {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (html) el.innerHTML = html;
    return el;
}
