import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple
from collections import deque
import numpy as np

_FLOOR_WORDS = {"floor", "carpet", "rug", "mat", "ground"}
_POSTURES    = {"standing_on", "sitting_on", "lying_on"}

def is_person(lbl: str) -> bool:
    return str(lbl).lower().strip().startswith("person")

def is_floor_lbl(lbl: str) -> bool:
    L = str(lbl).lower()
    return any(w in L for w in _FLOOR_WORDS)

def people_set(graph: List[Dict]) -> List[str]:
    ids = set()
    for tr in graph:
        s, r, o = tr["subject"], tr["relation"], tr["object"]
        if is_person(s):
            ids.add(s)
        if is_person(o):
            ids.add(o)
    return sorted(ids)

def edge_multiset(graph: List[Dict]) -> List[Tuple[str, str, str]]:
    return [(tr["subject"], tr["relation"], tr["object"]) for tr in graph]

def posture_of(graph: List[Dict], person_id: str) -> Tuple[str, Optional[str]]:
    posture, support = "unknown", None
    for tr in graph:
        if tr["subject"] == person_id and tr["relation"] in _POSTURES:
            posture, support = tr["relation"], tr["object"]
    return posture, support

def short_name(lbl: str) -> str:
    m = re.match(r"^\s*([A-Za-z]+)\s*[_ ]?(\d+)?", str(lbl).strip())
    if not m:
        return str(lbl).strip()
    base = m.group(1).lower()
    num = m.group(2) or ""
    return f"p{num}" if base == "person" else f"{base}{num}"

class DeltaNotes(NamedTuple):
    posture_drop: List[str]
    floor_contact: List[str]
    support_flip: List[Tuple[str, str, str]]
    births: List[str]
    deaths: List[str]
    rel_flips: List[Tuple[str, str, str]]
    falling: List[str]

def delta_score(prev_graph: List[Dict], cur_graph: List[Dict], weights: Dict[str, float]) -> Tuple[float, DeltaNotes]:
    score = 0.0
    prev_people = set(people_set(prev_graph))
    cur_people  = set(people_set(cur_graph))

    births = sorted(cur_people - prev_people)
    deaths = sorted(prev_people - cur_people)

    if births:
        score += weights["person_birth"] * len(births)
    if deaths:
        score += weights["person_death"] * len(deaths)

    posture_drop, floor_contact, support_flip = [], [], []

    for pid in sorted(prev_people & cur_people):
        p_prev, s_prev = posture_of(prev_graph, pid)
        p_cur,  s_cur  = posture_of(cur_graph,  pid)

        if p_prev in {"standing_on", "sitting_on"} and p_cur == "lying_on":
            posture_drop.append(pid)
            score += weights["posture_drop"]

        if (s_prev is None or not is_floor_lbl(s_prev)) and (s_cur is not None and is_floor_lbl(s_cur)):
            floor_contact.append(pid)
            score += weights["floor_contact"]

        if s_prev and s_cur and s_prev != s_cur:
            support_flip.append((pid, s_prev, s_cur))
            score += weights["support_flip"]

    prev_edges = set(edge_multiset(prev_graph))
    cur_edges  = set(edge_multiset(cur_graph))
    flips = list(((prev_edges - cur_edges) | (cur_edges - prev_edges)))
    flips = [f for f in flips if f[1] in {"to the left of", "to the right of", "above", "below"}]
    k = min(6, len(flips))
    rel_flips = flips[:k]
    if k > 0:
        score += weights["lr_above_below"] * k

    falling_events = [
        s for (s, r, o) in cur_edges
        if r == "falling_towards"
        and (s, r, o) not in prev_edges
        and is_person(s)
        and is_floor_lbl(o)
    ]
    if falling_events:
        score += weights["falling"] * len(falling_events)

    notes = DeltaNotes(
        posture_drop=posture_drop,
        floor_contact=floor_contact,
        support_flip=support_flip,
        births=births,
        deaths=deaths,
        rel_flips=rel_flips,
        falling=falling_events,
    )
    return score, notes

def derive_reason(notes: DeltaNotes, score: float, thr: float) -> str:
    parts = []
    if notes.posture_drop:
        parts.append(f"{', '.join(short_name(x) for x in notes.posture_drop)} posture drop")
    if notes.floor_contact:
        parts.append(f"{', '.join(short_name(x) for x in notes.floor_contact)} floor contact")
    if notes.falling:
        parts.append(f"{', '.join(short_name(x) for x in notes.falling)} falling_towards floor")
    if notes.support_flip:
        flips = "; ".join(f"{short_name(pid)} {short_name(old)}→{short_name(new)}" for (pid, old, new) in notes.support_flip)
        parts.append(f"support flip: {flips}")
    if not parts and notes.rel_flips:
        rels = "; ".join(f"{short_name(s)} {r} {short_name(o)}" for (s, r, o) in notes.rel_flips[:2])
        parts.append(f"spatial changes: {rels}")
    if not parts:
        parts.append("no salient deltas")

    parts.append(f"score={score:.2f}≥{thr:.2f}" if score >= thr else f"score={score:.2f}<{thr:.2f}")
    return "; ".join(parts)

class KOvN:
    def __init__(self, k: int, n: int, thr: float):
        self.k = k
        self.n = n
        self.thr = thr
        self.buf: List[float] = []

    def push(self, s: float) -> bool:
        self.buf = (self.buf + [s])[-self.n:]
        return sum(x >= self.thr for x in self.buf) >= self.k

def translate_delta(t: int, notes: DeltaNotes, cur_graph: List[Dict], max_lines: int):
    lines: List[str] = []
    for pid in notes.posture_drop:
        lines.append(f"- {short_name(pid)}: posture standing/sitting→lying")
    for pid in notes.floor_contact:
        lines.append(f"- {short_name(pid)}: support → floor")
    for pid in notes.falling:
        lines.append(f"- {short_name(pid)}: falling_towards floor")
    for (pid, old, new) in notes.support_flip:
        lines.append(f"- {short_name(pid)}: support {short_name(old)} → {short_name(new)}")
    if notes.births:
        lines.append(f"- new persons: {', '.join(short_name(x) for x in notes.births)}")
    if notes.deaths:
        lines.append(f"- persons left: {', '.join(short_name(x) for x in notes.deaths)}")
    for (s, r, o) in notes.rel_flips[:2]:
        lines.append(f"- {short_name(s)} {r} {short_name(o)}")

    ctx_pid = notes.posture_drop[:1] or notes.floor_contact[:1] or notes.falling[:1] or ([notes.support_flip[0][0]] if notes.support_flip else [])
    if ctx_pid:
        post, sup = posture_of(cur_graph, ctx_pid[0])
        sup_s = short_name(sup) if sup else "none"
        lines.append(f"- context {short_name(ctx_pid[0])}: posture={post}, support={sup_s}, on_floor={str(bool(sup and is_floor_lbl(sup)))}")

    lines = lines[:max_lines]
    return f"Frame {t} summary:\n" + ("\n".join(lines) if lines else "- no salient deltas")

class SlidingWindowCounter:
    def __init__(self, window_sec: float):
        self.window = window_sec
        self.events = deque()

    def push(self, t: float):
        self.events.append(t)
        self._gc(t)

    def count(self, now_t: Optional[float] = None) -> int:
        if now_t is None:
            now_t = time.time()
        self._gc(now_t)
        return len(self.events)

    def _gc(self, now_t: float):
        while self.events and (now_t - self.events[0]) > self.window:
            self.events.popleft()

class ConstraintManager:
    def __init__(self, max_alarms_per_minute: int = 6, budget_per_minute: float = 6.0):
        self.counter = SlidingWindowCounter(60.0)
        self.max_per_min = max_alarms_per_minute
        self.budget_per_min = budget_per_minute
        self.cost_counter = SlidingWindowCounter(60.0)

    def can_fire(self) -> bool:
        now = time.time()
        return not (self.counter.count(now) >= self.max_per_min or self.cost_counter.count(now) >= self.budget_per_min)

    def on_fire(self):
        t = time.time()
        self.counter.push(t)
        self.cost_counter.push(t)

class DynamicThreshold:
    def __init__(self, base=2.0, min_thr=1.0, max_thr=3.0, lr=0.05):
        self.value = base
        self.min_thr = min_thr
        self.max_thr = max_thr
        self.lr = lr

    def update(self, outcome: str):
        if outcome == "TP":
            self.value -= self.lr * 0.5
        elif outcome == "FP":
            self.value += self.lr
        elif outcome == "FN":
            self.value -= self.lr * 1.5
        elif outcome == "TN":
            self.value += self.lr * 0.2
        self.value = float(np.clip(self.value, self.min_thr, self.max_thr))

class ConstrainedLinUCB:
    def __init__(self, d: int, alpha: float = 1.5, seed: int = 42, eps_greedy: float = 0.05):
        self.d = d
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        self.A = [np.eye(d), np.eye(d)]
        self.b = [np.zeros((d,)), np.zeros((d,))]
        self.eps = eps_greedy

    def _theta(self, a: int) -> np.ndarray:
        return np.linalg.solve(self.A[a], self.b[a])

    def _ucb(self, x: np.ndarray, a: int) -> float:
        Ainv = np.linalg.inv(self.A[a])
        theta = self._theta(a)
        mu = float(theta @ x)
        sigma = float(np.sqrt(x @ Ainv @ x))
        return mu + self.alpha * sigma

    def decide(self, x: np.ndarray, constraints: ConstraintManager, cooldown_ok: bool):
        if not cooldown_ok or not constraints.can_fire():
            u0 = self._ucb(x, 0)
            u1 = self._ucb(x, 1)
            conf = float(1 / (1 + np.exp(-(u1 - u0))))
            return 0, conf, "constraints-blocked"

        if self.rng.rand() < self.eps:
            a = int(self.rng.rand() < 0.5)
        else:
            u0 = self._ucb(x, 0)
            u1 = self._ucb(x, 1)
            a = 1 if u1 > u0 else 0

        u0 = self._ucb(x, 0)
        u1 = self._ucb(x, 1)
        conf = float(1 / (1 + np.exp(-(u1 - u0))))
        return a, conf, "linucb"

    def update(self, x: np.ndarray, a: int, reward: float):
        self.A[a] += np.outer(x, x)
        self.b[a] += reward * x

def feature_vector_from_deltas(score: float, notes: DeltaNotes, cur_graph: List[Dict]) -> np.ndarray:
    p_drop   = len(notes.posture_drop)
    f_contact= len(notes.floor_contact)
    s_flip   = len(notes.support_flip)
    births   = len(notes.births)
    deaths   = len(notes.deaths)
    relflip  = len(notes.rel_flips)
    falling  = len(notes.falling)
    people   = len(people_set(cur_graph))

    rel1 = (p_drop + f_contact) / max(1, people)
    rel2 = (s_flip + relflip) / max(1, people)

    has_floor = 1.0 if any(is_floor_lbl(tr["object"]) for tr in cur_graph if tr["relation"] in _POSTURES) else 0.0

    return np.array(
        [1.0, score, p_drop, f_contact, s_flip, relflip, births, deaths, people, rel1, rel2, has_floor, falling],
        dtype=np.float32,
    )

@dataclass
class TwoFrameState:
    smoother: KOvN
    last_fire_ts: float = -1e9
    dynamic_thr: DynamicThreshold = field(default_factory=lambda: DynamicThreshold(base=2.0))

    def cooldown_ok(self, cooldown_s: float) -> bool:
        return (time.time() - self.last_fire_ts) >= cooldown_s

    def mark_fired(self):
        self.last_fire_ts = time.time()
