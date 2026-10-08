import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileAudio, CheckCircle2, ChevronDown, Download, Play, 
  Settings, User, Clock, FileText, Activity
} from 'lucide-react';
import { cn } from './lib/utils';
import { MOCK_DATA } from './mockData';

function TopBar() {
  return (
    <header className="h-16 border-b border-divider bg-surface backdrop-blur-xl flex items-center justify-between px-6 shrink-0 z-10 relative">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30">
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <span className="font-semibold text-zinc-100 tracking-tight">MeetScribe</span>
        </div>
        
        <div className="h-4 w-px bg-divider mx-2" />
        
        <div className="flex flex-col">
          <span className="text-sm font-medium text-zinc-200">{MOCK_DATA.metadata.title}</span>
          <span className="text-xs text-zinc-500 flex items-center gap-2">
            <FileAudio className="w-3 h-3" /> {MOCK_DATA.metadata.file} • {MOCK_DATA.metadata.duration}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-8">
        <div className="flex items-center gap-4 text-xs font-medium text-zinc-400">
          <div className="flex items-center gap-2 text-emerald-400">
            <CheckCircle2 className="w-4 h-4" />
            <span>1. STT Consensus</span>
          </div>
          <div className="h-px w-4 bg-divider" />
          <div className="flex items-center gap-2 text-emerald-400">
            <CheckCircle2 className="w-4 h-4" />
            <span>2. Cloze Refinement</span>
          </div>
          <div className="h-px w-4 bg-divider" />
          <div className="flex items-center gap-2 text-zinc-200">
            <div className="w-4 h-4 rounded-full border-2 border-zinc-500 border-t-zinc-200 animate-spin" />
            <span>3. Ledger Verification</span>
          </div>
        </div>

        <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-hover border border-divider text-sm text-zinc-300 hover:text-white transition-colors">
          <Download className="w-4 h-4" />
          Export
          <ChevronDown className="w-3 h-3 ml-1" />
        </button>
      </div>
    </header>
  );
}

function TranscriptChip({ disputed }: { disputed: any }) {
  const [open, setOpen] = useState(false);

  const isCritical = disputed.category === 'CRITICAL';
  const chipClass = isCritical 
    ? 'bg-amber-500/10 text-amber-300 border-amber-500/25 hover:bg-amber-500/20' 
    : 'bg-indigo-500/10 text-indigo-300 border-indigo-500/20 hover:bg-indigo-500/20';

  return (
    <span className="relative inline-block mx-1">
      <button 
        onClick={() => setOpen(!open)}
        className={cn(
          "px-2 py-0.5 rounded-md border text-sm transition-all duration-200", 
          chipClass
        )}
      >
        {disputed.resolvedText}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 5, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 5, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-64 p-3 rounded-xl bg-obsidian border border-divider shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] z-50"
          >
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs uppercase tracking-wider text-zinc-500">ID: {disputed.id}</span>
              <span className={cn("text-[10px] px-1.5 py-0.5 rounded border uppercase", isCritical ? 'text-amber-400 border-amber-500/30' : 'text-indigo-400 border-indigo-500/30')}>
                {disputed.category}
              </span>
            </div>
            
            <div className="space-y-2 text-sm text-zinc-300 mb-3">
              <div className="flex justify-between">
                <span className="text-zinc-500">Engine A:</span>
                <span>{disputed.engineA}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Engine B:</span>
                <span>{disputed.engineB}</span>
              </div>
            </div>

            <div className="pt-2 border-t border-divider">
              <p className="text-xs text-zinc-400 mb-1">Resolution: {disputed.resolution}</p>
              <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-500/50 rounded-full" style={{ width: `${disputed.confidence * 100}%` }} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  );
}

function Transcript() {
  return (
    <div className="flex flex-col h-full bg-obsidian/50 border-r border-divider relative">
      <div className="flex items-center justify-between p-4 border-b border-divider shrink-0">
        <h2 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
          <FileText className="w-4 h-4" />
          Transcript Inspector
        </h2>
        <div className="flex bg-surface rounded-lg p-1">
          <button className="px-3 py-1 rounded-md bg-white/10 text-xs font-medium text-white shadow-sm">Refined</button>
          <button className="px-3 py-1 rounded-md text-xs font-medium text-zinc-400 hover:text-zinc-200 transition-colors">Raw</button>
        </div>
      </div>
      
      <div className="h-1 bg-zinc-800 w-full relative">
        <div className="absolute top-0 left-[20%] w-1 h-1 bg-amber-500 rounded-full shadow-[0_0_8px_rgba(245,158,11,0.6)]" />
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 p-6 space-y-6">
        {MOCK_DATA.transcript.map(utt => (
          <div key={utt.id} className="flex gap-4 group">
            <div className="w-16 shrink-0 text-right">
              <span className="text-xs text-zinc-500 font-mono">{utt.timestamp}</span>
            </div>
            <div className="flex-1">
              <div className="text-xs font-medium text-zinc-400 mb-1">{utt.speaker}</div>
              <p className="text-zinc-200 leading-relaxed text-sm">
                {utt.text}
                {utt.disputed && <TranscriptChip disputed={utt.disputed} />}
                {utt.textAfter}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Ledger() {
  const [activeTab, setActiveTab] = useState('decisions');

  const tabs = [
    { id: 'summary', label: 'Summary' },
    { id: 'decisions', label: 'Decisions (LEDGER)' },
    { id: 'tasks', label: 'Action Items' },
    { id: 'audit', label: 'Dispute Audit' },
  ];

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-divider shrink-0">
        <div className="flex gap-2">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                activeTab === t.id 
                  ? "bg-surface-hover text-zinc-100 border border-divider shadow-sm" 
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-surface"
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 p-6">
        {activeTab === 'summary' && (
          <div className="space-y-6">
            <h3 className="text-lg font-medium text-zinc-100">Executive Summary</h3>
            <div className="text-sm text-zinc-300 leading-relaxed space-y-4">
              <p>
                The team discussed the upcoming Q3 strategy and marketing roadmap. A primary focus was the budget allocation for the new campaign, where it was agreed to start with an initial budget of $15,000 <span className="inline-block px-1.5 py-0.5 rounded bg-surface border border-divider text-xs font-mono text-emerald-400 cursor-pointer hover:bg-surface-hover">[D-1]</span> to test the waters before scaling up.
              </p>
              <p>
                There was also a proposal to hire an external design agency <span className="inline-block px-1.5 py-0.5 rounded bg-surface border border-divider text-xs font-mono text-zinc-400 cursor-pointer hover:bg-surface-hover">[D-2]</span>, but the team decided to stick with internal resources for now. Action items were assigned, including sending out the meeting notes <span className="inline-block px-1.5 py-0.5 rounded bg-surface border border-divider text-xs font-mono text-zinc-400 cursor-pointer hover:bg-surface-hover">[T-1]</span> and updating the landing page <span className="inline-block px-1.5 py-0.5 rounded bg-surface border border-divider text-xs font-mono text-emerald-400 cursor-pointer hover:bg-surface-hover">[T-2]</span>.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'decisions' && (
          <div className="space-y-4">
            {MOCK_DATA.decisions.map(d => (
              <div key={d.id} className="p-4 rounded-xl bg-surface border border-divider hover:bg-surface-hover transition-all duration-200">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-zinc-500 text-xs font-mono">{d.id}</span>
                    <span className={cn(
                      "text-[10px] px-2 py-0.5 rounded-full border uppercase tracking-wide font-semibold",
                      d.status === 'AGREED' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-zinc-800/60 text-zinc-300 border-zinc-700/50'
                    )}>
                      {d.status}
                    </span>
                    <span className={cn(
                      "text-[10px] px-2 py-0.5 rounded-full border uppercase tracking-wide",
                      d.confidence === 'HIGH CONFIDENCE' ? 'text-zinc-400 border-zinc-700/50' : 'bg-amber-500/10 text-amber-300 border-amber-500/25'
                    )}>
                      {d.confidence}
                    </span>
                  </div>
                </div>
                
                <h3 className="text-zinc-100 font-medium mb-2">{d.title}</h3>
                
                <div className="space-y-2 mt-4 pl-3 border-l-2 border-divider">
                  <div className="text-sm text-zinc-300 italic">"{d.proposalQuote}"</div>
                  {d.agreementQuote && (
                    <div className="text-sm text-emerald-400/80 italic flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3" /> "{d.agreementQuote}" <span className="text-zinc-500 text-xs not-italic">- {d.speaker} at {d.timestamp}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'tasks' && (
          <div className="space-y-4">
            {MOCK_DATA.actionItems.map(t => (
              <div key={t.id} className="p-4 rounded-xl bg-surface border border-divider hover:bg-surface-hover transition-all duration-200">
                <div className="flex items-center gap-3 mb-3">
                  <span className="text-zinc-500 text-xs font-mono">{t.id}</span>
                  <span className={cn(
                    "text-[10px] px-2 py-0.5 rounded-full border uppercase tracking-wide font-semibold",
                    t.status === 'CONFIRMED' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-zinc-800/60 text-zinc-300 border-zinc-700/50'
                  )}>
                    {t.status}
                  </span>
                </div>
                
                <p className="text-zinc-200 font-medium mb-4">{t.description}</p>
                
                <div className="flex gap-4">
                  <div className="flex items-center gap-2 text-xs">
                    <User className="w-3.5 h-3.5 text-zinc-500" />
                    <span className={cn(
                      "px-2 py-1 rounded-md border",
                      t.owner === 'Unspecified' ? 'border-dashed border-zinc-700 text-zinc-500' : 'border-zinc-700/50 text-zinc-300 bg-zinc-800/40'
                    )}>
                      {t.owner}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <Clock className="w-3.5 h-3.5 text-zinc-500" />
                    <span className={cn(
                      "px-2 py-1 rounded-md border",
                      t.deadline === 'Unspecified' ? 'border-dashed border-zinc-700 text-zinc-500' : 'border-zinc-700/50 text-zinc-300 bg-zinc-800/40'
                    )}>
                      {t.deadline}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'audit' && (
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-surface border border-divider">
              <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Edit Budget Used</div>
              <div className="text-2xl font-light text-zinc-100">{MOCK_DATA.audit.editBudgetUsed} <span className="text-sm text-zinc-500">/ {MOCK_DATA.audit.editBudgetMax}</span></div>
            </div>
            <div className="p-4 rounded-xl bg-surface border border-divider">
              <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Total Words</div>
              <div className="text-2xl font-light text-zinc-100">{MOCK_DATA.audit.totalWords}</div>
            </div>
            <div className="p-4 rounded-xl bg-surface border border-divider">
              <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Disputed Spans</div>
              <div className="text-2xl font-light text-amber-300">{MOCK_DATA.audit.totalDisputedSpans}</div>
            </div>
            <div className="p-4 rounded-xl bg-surface border border-divider">
              <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">False Confirmation Rate</div>
              <div className="text-2xl font-light text-emerald-400">{MOCK_DATA.audit.falseConfirmationRate}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function AudioPlayer() {
  return (
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-96 bg-surface-hover backdrop-blur-2xl border border-white/10 shadow-[0_12px_40px_0_rgba(0,0,0,0.5)] rounded-2xl p-3 flex items-center justify-between z-20">
      <div className="flex items-center gap-3">
        <button className="w-8 h-8 rounded-full bg-zinc-100 flex items-center justify-center hover:bg-zinc-300 transition-colors">
          <Play className="w-4 h-4 text-zinc-900 ml-0.5" />
        </button>
        <span className="text-xs font-medium text-zinc-300 font-mono">04:12 / 45:12</span>
      </div>
      <div className="flex items-center gap-2">
        <button className="text-[10px] font-bold text-zinc-400 hover:text-zinc-200 px-2">1.25x</button>
        <div className="h-4 w-px bg-divider mx-1" />
        <button className="text-zinc-400 hover:text-zinc-200">
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

function App() {
  return (
    <div className="flex flex-col h-screen w-full relative">
      <TopBar />
      <div className="flex flex-1 overflow-hidden">
        <div className="w-[45%] h-full">
          <Transcript />
        </div>
        <div className="w-[55%] h-full">
          <Ledger />
        </div>
      </div>
      <AudioPlayer />
    </div>
  );
}

export default App;
