// Regression test for JS rendering
import { renderResults } from './meeting_assistant/static/js/views/results.js';
import { state } from './meeting_assistant/static/js/state.js';

// Mock DOM
global.document = {
    createElement: (tag) => {
        return { 
            textContent: '',
            get innerHTML() { return this.textContent.replace(/</g, "&lt;").replace(/>/g, "&gt;"); },
            style: {},
            classList: { add: () => {}, remove: () => {} }
        };
    },
    addEventListener: () => {}
};
global.window = { location: { href: '' } };

const mockContainer = {
    innerHTML: '',
    querySelector: () => mockContainer,
    querySelectorAll: () => [], addEventListener: () => {}
};

const mockOverlay = { innerHTML: '' };
const mockDock = { innerHTML: '' };

const testCases = [
    {
        name: "Normal segments",
        segments: [{ text: "Hello", start: 0.0, end: 1.0, speaker: "spk1" }]
    },
    {
        name: "Missing speaker",
        segments: [{ text: "No speaker", start: 0.0, end: 1.0 }]
    },
    {
        name: "Empty segments",
        segments: []
    },
    {
        name: "Missing start/end",
        segments: [{ text: "Missing times" }]
    }
];

let failed = false;

for (const tc of testCases) {
    console.log(`Running test: ${tc.name}`);
    state.result = {
        metadata: { file_name: "test", audio_duration_s: 10, models_used: [] },
        raw_transcript: { 
            segments: tc.segments,
            raw_text: tc.segments.map(s => s.text).join(' ')
        },
        minutes: {}
    };
    
    try {
        renderResults(mockContainer, mockDock, mockOverlay);
        if (mockContainer.innerHTML.includes('error-banner')) {
            console.error(`  Failed: error banner shown!`);
            failed = true;
        } else {
            console.log(`  Passed!`);
        }
    } catch (e) {
        console.error(`  Uncaught Exception!`, e);
        failed = true;
    }
}

if (failed) process.exit(1);
console.log("All UI render tests passed!");
