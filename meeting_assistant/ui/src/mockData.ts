export const MOCK_DATA = {
  metadata: {
    title: "Q3 Strategy & Roadmap Sync",
    duration: "45:12",
    date: "2026-10-07",
    file: "roadmap_sync_v2.wav"
  },
  transcript: [
    {
      id: "u1",
      speaker: "Speaker 1",
      timestamp: "04:12",
      text: "We need to allocate the budget for the new marketing campaign. I'm thinking we start with ",
      disputed: {
        id: "s1",
        category: "CRITICAL",
        engineA: "$15,000",
        engineB: "$50,000",
        resolvedText: "$15,000",
        resolution: "Modality Guard Blocked Edit",
        confidence: 0.92
      },
      textAfter: " just to test the waters."
    },
    {
      id: "u2",
      speaker: "Speaker 2",
      timestamp: "04:18",
      text: "Sounds good. Should we also consider updating the landing page?",
    },
    {
      id: "u3",
      speaker: "Speaker 1",
      timestamp: "04:22",
      text: "Yes, definitely. I'll take care of the landing page updates.",
    },
    {
      id: "u4",
      speaker: "Speaker 3",
      timestamp: "04:28",
      text: "Maybe we should hire an external agency for the design?",
    },
    {
      id: "u5",
      speaker: "Speaker 2",
      timestamp: "04:32",
      text: "Let's stick to the internal team for now. Also, can someone send out the meeting notes?",
    }
  ],
  decisions: [
    {
      id: "D-1",
      status: "AGREED",
      title: "Marketing Budget",
      proposalQuote: "allocate the budget for the new marketing campaign. I'm thinking we start with $15,000",
      agreementQuote: "Sounds good.",
      speaker: "Speaker 2",
      timestamp: "04:18",
      confidence: "HIGH CONFIDENCE"
    },
    {
      id: "D-2",
      status: "PROPOSED",
      title: "External Agency",
      proposalQuote: "Maybe we should hire an external agency for the design?",
      agreementQuote: null,
      speaker: "Speaker 3",
      timestamp: "04:28",
      confidence: "HIGH CONFIDENCE"
    }
  ],
  actionItems: [
    {
      id: "T-1",
      description: "Send out the meeting notes",
      owner: "Unspecified",
      deadline: "Unspecified",
      status: "TENTATIVE",
      quote: "can someone send out the meeting notes?"
    },
    {
      id: "T-2",
      description: "Take care of the landing page updates",
      owner: "Speaker 1 (self-assigned)",
      deadline: "Unspecified",
      status: "CONFIRMED",
      quote: "I'll take care of the landing page updates."
    }
  ],
  audit: {
    totalWords: 4520,
    editBudgetUsed: "1.2%",
    editBudgetMax: "3.0%",
    totalDisputedSpans: 4,
    falseConfirmationRate: "0%"
  }
};
