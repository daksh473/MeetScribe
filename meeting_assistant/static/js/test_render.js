import { renderResults } from './views/results.js';
import { state } from './state.js';

// Mock state
state.result = {
    metadata: {
        file_name: "test.mp3",
        audio_duration_s: 100,
        models_used: ["A", "B"]
    },
    raw_transcript: {
        raw_text: "Hello world",
        segments: [],
        uncertain_spans: [],
        metadata: {
            engine_statuses: {
                "whisper": "OK",
                "elevenlabs": "blocked / failed (Provider message: Error)"
            }
        }
    },
    minutes: {
        decisions: [],
        action_items: []
    }
};

const container = {
    innerHTML: '',
    querySelector: () => ({ addEventListener: () => {} }),
    querySelectorAll: () => []
};

try {
    renderResults(container, {}, {});
    console.log("Render Success! HTML length:", container.innerHTML.length);
} catch (e) {
    console.error("CRASH:", e);
}
