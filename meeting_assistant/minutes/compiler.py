"""Pass B: Compiler. Deterministic rules to build Ledger items from Dialogue Acts."""

from typing import Dict, List, Tuple
from llm.schemas import ActionItem, Decision, DialogueAct, Utterance


class Cluster:
    def __init__(self):
        self.acts: List[DialogueAct] = []
        self.last_idx: int = -1


def _find_utterance_idx(uid: str, utts: List[Utterance]) -> int:
    for i, u in enumerate(utts):
        if u.id == uid:
            return i
    return -1


def _find_speaker(uid: str, utts: List[Utterance]) -> str:
    for u in utts:
        if u.id == uid:
            return u.speaker
    return "Unknown"


def _quote_in_utterance(quote: str, uid: str, utts: List[Utterance]) -> bool:
    for u in utts:
        if u.id == uid:
            # Must be an exact substring
            return quote.strip() in u.text
    return False


def compile_ledger(acts: List[DialogueAct], utterances: List[Utterance], window: int = 5) -> Tuple[List[Decision], List[ActionItem]]:
    """Compile raw dialogue acts into strict Decisions and Action Items."""
    
    # 1. Cluster acts
    clusters: List[Cluster] = []
    act_to_cluster: Dict[str, Cluster] = {}  # target_uid -> Cluster (based on the initial proposal/task)
    
    # Sort acts by utterance index just in case
    sorted_acts = sorted(acts, key=lambda a: _find_utterance_idx(a.utterance_id, utterances))
    
    for act in sorted_acts:
        idx = _find_utterance_idx(act.utterance_id, utterances)
        if idx == -1:
            continue
            
        added = False
        
        # Check in_reply_to chain
        if act.in_reply_to:
            for c in clusters:
                if any(a.utterance_id == act.in_reply_to for a in c.acts):
                    c.acts.append(act)
                    c.last_idx = max(c.last_idx, idx)
                    added = True
                    break
                    
        # Check proximity if not added
        if not added:
            # If it's a seed act (PROPOSAL, TASK), start new cluster
            if act.label in {"PROPOSAL", "TASK"}:
                c = Cluster()
                c.acts.append(act)
                c.last_idx = idx
                clusters.append(c)
            else:
                # Attach to most recent active cluster
                active = [c for c in clusters if idx - c.last_idx <= window]
                if active:
                    active[-1].acts.append(act)
                    active[-1].last_idx = max(active[-1].last_idx, idx)
                else:
                    # Orphan act, just start a cluster (might be useless but safe)
                    c = Cluster()
                    c.acts.append(act)
                    c.last_idx = idx
                    clusters.append(c)

    decisions: List[Decision] = []
    action_items: List[ActionItem] = []
    
    d_counter = 1
    t_counter = 1
    
    # 2. Evaluate clusters
    for c in clusters:
        proposals = [a for a in c.acts if a.label == "PROPOSAL"]
        tasks = [a for a in c.acts if a.label == "TASK"]
        
        # Helper to check evidence
        def build_evidence(al: List[DialogueAct]) -> Tuple[List[str], bool]:
            ev = []
            low_conf = False
            for a in al:
                if not a.quote or not a.utterance_id:
                    continue
                if not _quote_in_utterance(a.quote, a.utterance_id, utterances):
                    low_conf = True
                ev.append(f"{a.quote} [{a.utterance_id}]")
            return ev, low_conf
            
        # Process Decisions
        for prop in proposals:
            # Find related agreements/rejections/deferrals in this cluster
            # strictly, we can just look if they exist in the cluster since we clustered them
            agreements = [a for a in c.acts if a.label == "AGREEMENT"]
            rejections = [a for a in c.acts if a.label == "REJECTION"]
            deferrals = [a for a in c.acts if a.label == "DEFERRAL"]
            
            # Find the latest relevant act index
            prop_idx = _find_utterance_idx(prop.utterance_id, utterances)
            
            status = "PROPOSED"
            
            if deferrals:
                status = "DEFERRED"
            elif rejections:
                # If there's a rejection, is it after an agreement? Doesn't matter, rejection wins.
                status = "REJECTED"
            elif agreements:
                status = "AGREED"
                
            related_acts = [prop] + agreements + rejections + deferrals
            ev, low_conf = build_evidence(related_acts)
            
            if not ev:
                continue
                
            decisions.append(Decision(
                id=f"D{d_counter}",
                text=prop.quote,
                status=status,
                evidence=ev,
                low_confidence=low_conf
            ))
            d_counter += 1
            
        # Process Tasks
        for t in tasks:
            agreements = [a for a in c.acts if a.label == "AGREEMENT"]
            owners = [a for a in c.acts if a.label == "OWNER"]
            timeframes = [a for a in c.acts if a.label == "TIMEFRAME"]
            
            status = "TENTATIVE"
            owner = "Unspecified"
            deadline = "Unspecified"
            
            # Owner rules
            if owners:
                o_act = owners[0]
                spk = _find_speaker(o_act.utterance_id, utterances)
                
                # Check for tentative language in all owner/task acts
                tentative_words = {"might", "maybe", "should", "could", "can", "probably"}
                is_tentative = False
                for a in [t] + owners:
                    for u in utterances:
                        if u.id == a.utterance_id:
                            text_lower = u.text.lower()
                            if "?" in text_lower or any(w in text_lower.split() for w in tentative_words):
                                is_tentative = True
                            
                quote_lower = o_act.quote.lower()
                if "i'll" in quote_lower or "i will" in quote_lower or "i can" in quote_lower or "let me" in quote_lower:
                    owner = f"{spk} (self-assigned)"
                else:
                    owner = o_act.quote
                    
                status = "TENTATIVE" if is_tentative else "CONFIRMED"
            elif agreements:
                status = "CONFIRMED"
                
            if timeframes:
                deadline = timeframes[0].quote
                
            related_acts = [t] + agreements + owners + timeframes
            ev, low_conf = build_evidence(related_acts)
            
            if not ev:
                continue
                
            action_items.append(ActionItem(
                id=f"T{t_counter}",
                task=t.quote,
                status=status,
                owner=owner,
                deadline=deadline,
                evidence=ev,
                low_confidence=low_conf
            ))
            t_counter += 1
            
    return decisions, action_items
