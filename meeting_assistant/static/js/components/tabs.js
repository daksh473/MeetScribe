export function initTabs(container) {
    const tabs = container.querySelectorAll('.tab-btn');
    const panes = container.querySelectorAll('.tab-pane');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.setAttribute('aria-selected', 'false'));
            panes.forEach(p => p.classList.add('hidden'));
            
            tab.setAttribute('aria-selected', 'true');
            const targetId = tab.getAttribute('aria-controls');
            container.querySelector(`#${targetId}`).classList.remove('hidden');
        });
    });
}
