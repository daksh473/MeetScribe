export const state = {
    jobId: null,
    result: null,
    audioDuration: 0
};

export function setJobId(id) {
    state.jobId = id;
}

export function setResult(res) {
    state.result = res;
}
