"""
Block World Solver — Algorithm Infrastructure
===============================================
This module defines the state representation, validation, successor generation,
and the plug-in infrastructure for search algorithms and heuristics.

Algorithms (A*, IDA*, etc.) and heuristics will be added in a later phase.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import (
    Callable, Dict, List, Optional, Tuple,
)


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

# A stack is a tuple of block names, ordered bottom-to-top.
Stack = Tuple[str, ...]

# A state is a frozenset of stacks (order on the table is irrelevant).
State = frozenset  # frozenset[Stack]


def make_state(stacks: List[List[str]]) -> State:
    """Create a canonical State from a list of stacks (each bottom→top)."""
    return frozenset(tuple(s) for s in stacks if s)


def state_to_sorted_lists(state: State) -> List[List[str]]:
    """Convert a State back to a sorted list of lists (for deterministic display)."""
    return sorted([list(s) for s in state], key=lambda s: (len(s), s))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when the input is invalid."""
    pass


def validate_state(stacks: List[List[str]], label: str = "מצב") -> State:
    """
    Validate a single state description.
    - No duplicate blocks.
    - No empty names.
    - 1 ≤ total blocks ≤ 10.
    Returns a canonical State on success.
    """
    all_blocks: List[str] = []
    for stack in stacks:
        for block in stack:
            if not block or not block.strip():
                raise ValidationError(f"{label}: שם קובייה ריק אינו חוקי")
            all_blocks.append(block.strip())

    if len(all_blocks) == 0:
        raise ValidationError(f"{label}: יש להזין לפחות קובייה אחת")
    if len(all_blocks) > 10:
        raise ValidationError(
            f"{label}: מספר הקוביות ({len(all_blocks)}) חורג מהמקסימום (10)"
        )

    seen = set()
    for b in all_blocks:
        if b in seen:
            raise ValidationError(f"{label}: הקובייה '{b}' מופיעה יותר מפעם אחת")
        seen.add(b)

    return make_state(stacks)


def validate_pair(
    start_stacks: List[List[str]], goal_stacks: List[List[str]]
) -> Tuple[State, State]:
    """
    Validate a (start, goal) pair:
    - Each must be individually valid.
    - Both must contain exactly the same set of blocks.
    """
    start = validate_state(start_stacks, "מצב התחלה")
    goal = validate_state(goal_stacks, "מצב יעד")

    start_blocks = {b for s in start for b in s}
    goal_blocks = {b for s in goal for b in s}

    if start_blocks != goal_blocks:
        missing_in_goal = start_blocks - goal_blocks
        missing_in_start = goal_blocks - start_blocks
        parts = []
        if missing_in_goal:
            parts.append(f"חסרות ביעד: {', '.join(sorted(missing_in_goal))}")
        if missing_in_start:
            parts.append(f"חסרות בהתחלה: {', '.join(sorted(missing_in_start))}")
        raise ValidationError(
            "קבוצות הקוביות אינן זהות בין מצב ההתחלה למצב היעד. " +
            "; ".join(parts)
        )

    return start, goal


# ---------------------------------------------------------------------------
# Successor generation
# ---------------------------------------------------------------------------

@dataclass
class Move:
    """Describes a single block move."""
    block: str
    source_desc: str   # human-readable description of source stack
    dest_desc: str     # human-readable description of destination
    source_stack: Stack
    dest_stack: Optional[Stack]  # None → new stack on table

    def describe(self) -> str:
        return f"{self.block}: {self.source_desc} → {self.dest_desc}"


def get_successors(state: State) -> List[Tuple[State, Move]]:
    """
    Generate all valid successor states from a given state.
    Each successor is paired with the Move that produced it.
    """
    stacks = list(state)
    successors: List[Tuple[State, Move]] = []

    for i, src in enumerate(stacks):
        if not src:
            continue
        block = src[-1]  # top block
        remaining_src = src[:-1]  # source after removal

        # --- Move to table (new stack) ---
        # Only valid if block is not already alone on table (src has >1 block)
        if len(src) > 1:
            new_stacks = []
            for j, s in enumerate(stacks):
                if j == i:
                    if remaining_src:
                        new_stacks.append(remaining_src)
                else:
                    new_stacks.append(s)
            new_stacks.append((block,))
            new_state = frozenset(new_stacks)

            src_desc = _stack_desc(src)
            move = Move(
                block=block,
                source_desc=f"ראש ערימה {src_desc}",
                dest_desc="שולחן",
                source_stack=src,
                dest_stack=None,
            )
            successors.append((new_state, move))

        # --- Move to top of another stack ---
        for j, dest in enumerate(stacks):
            if j == i:
                continue
            new_stacks = []
            for k, s in enumerate(stacks):
                if k == i:
                    if remaining_src:
                        new_stacks.append(remaining_src)
                elif k == j:
                    new_stacks.append(s + (block,))
                else:
                    new_stacks.append(s)
            new_state = frozenset(new_stacks)

            src_desc = _stack_desc(src)
            dest_desc = _stack_desc(dest)
            move = Move(
                block=block,
                source_desc=f"ראש ערימה {src_desc}",
                dest_desc=f"ראש ערימה {dest_desc}",
                source_stack=src,
                dest_stack=dest,
            )
            successors.append((new_state, move))

    return successors


def _stack_desc(stack: Stack) -> str:
    """Human-readable description of a stack (bottom→top)."""
    return "-".join(stack)


# ---------------------------------------------------------------------------
# Search result data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchMetrics:
    """Metrics collected during a search run."""
    runtime_seconds: float = 0.0
    states_generated: int = 0       # states produced before duplicate check
    states_expanded: int = 0        # states from which children were generated
    max_queue_size: int = 0         # A*: max priority queue size
    path_improvements: int = 0      # times a shorter path to a state was found
    ida_iterations: int = 0         # IDA*: number of threshold iterations


@dataclass
class SearchLogEntry:
    """One entry in the A* search log (for visualization)."""
    step: int
    state: State
    g: int          # accumulated cost
    h: float        # heuristic estimate
    f: float        # g + h
    queue_size: int
    action: Optional[str]  # move description or None for initial
    children_info: Optional[List[Dict]] = None  # info about successors


@dataclass
class IDAIterationLog:
    """Log for one IDA* iteration."""
    iteration: int
    threshold: float
    expansions: int
    prunings: int
    next_threshold: float
    sample_expansions: List[Dict] = field(default_factory=list)
    sample_prunings: List[Dict] = field(default_factory=list)


@dataclass
class SearchResult:
    """Complete result of a search run."""
    success: bool
    path: List[Tuple[State, Optional[Move]]]  # (state, move_that_led_here)
    cost: int
    metrics: SearchMetrics
    search_log: List[SearchLogEntry] = field(default_factory=list)        # A*
    ida_logs: List[IDAIterationLog] = field(default_factory=list)          # IDA*
    error_message: str = ""


# ---------------------------------------------------------------------------
# Algorithm & Heuristic registries
# ---------------------------------------------------------------------------

# Registry: name → callable
_algorithms: Dict[str, Callable] = {}
_heuristics: Dict[str, Callable] = {}


def register_algorithm(name: str):
    """Decorator to register a search algorithm."""
    def decorator(func):
        _algorithms[name] = func
        return func
    return decorator


def register_heuristic(name: str):
    """Decorator to register a heuristic function."""
    def decorator(func):
        _heuristics[name] = func
        return func
    return decorator


def get_algorithms() -> List[str]:
    """Return list of registered algorithm names."""
    return list(_algorithms.keys())


def get_heuristics() -> List[str]:
    """Return list of registered heuristic names."""
    return list(_heuristics.keys())


def run_search(
    algorithm_name: str,
    heuristic_name: str,
    start: State,
    goal: State,
) -> SearchResult:
    """
    Run a search algorithm with a given heuristic.
    Returns a SearchResult.
    """
    if algorithm_name not in _algorithms:
        return SearchResult(
            success=False, path=[], cost=0,
            metrics=SearchMetrics(),
            error_message=f"אלגוריתם '{algorithm_name}' לא נמצא",
        )
    if heuristic_name not in _heuristics:
        return SearchResult(
            success=False, path=[], cost=0,
            metrics=SearchMetrics(),
            error_message=f"היוריסטיקה '{heuristic_name}' לא נמצאה",
        )

    algo_func = _algorithms[algorithm_name]
    heuristic_func = _heuristics[heuristic_name]

    t0 = time.perf_counter()
    result = algo_func(start, goal, heuristic_func)
    result.metrics.runtime_seconds = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Placeholder heuristic (zero — makes A* act as UCS / Dijkstra)
# ---------------------------------------------------------------------------

@register_heuristic("h=0 (ללא היוריסטיקה)")
def heuristic_zero(state: State, goal: State) -> float:
    """Trivial zero heuristic — admissible but uninformative."""
    return 0.0


# ---------------------------------------------------------------------------
# Serialization helpers (for JSON API)
# ---------------------------------------------------------------------------

def state_to_json(state: State) -> List[List[str]]:
    """Convert a State to a JSON-serializable list of lists."""
    return state_to_sorted_lists(state)


def move_to_json(move: Optional[Move]) -> Optional[Dict]:
    """Convert a Move to a JSON-serializable dict."""
    if move is None:
        return None
    return {
        "block": move.block,
        "source_desc": move.source_desc,
        "dest_desc": move.dest_desc,
        "description": move.describe(),
    }


def search_result_to_json(result: SearchResult) -> Dict:
    """Convert a SearchResult to a JSON-serializable dict."""
    path_json = []
    for i, (state, move) in enumerate(result.path):
        path_json.append({
            "step": i,
            "state": state_to_json(state),
            "move": move_to_json(move),
            "cumulative_cost": i,  # each move costs 1
        })

    log_json = []
    for entry in result.search_log:
        log_entry = {
            "step": entry.step,
            "state": state_to_json(entry.state),
            "g": entry.g,
            "h": entry.h,
            "f": entry.f,
            "queue_size": entry.queue_size,
            "action": entry.action,
        }
        if entry.children_info is not None:
            log_entry["children_info"] = entry.children_info
        log_json.append(log_entry)

    ida_json = []
    for il in result.ida_logs:
        ida_json.append({
            "iteration": il.iteration,
            "threshold": il.threshold,
            "expansions": il.expansions,
            "prunings": il.prunings,
            "next_threshold": il.next_threshold,
            "sample_expansions": il.sample_expansions,
            "sample_prunings": il.sample_prunings,
        })

    return {
        "success": result.success,
        "cost": result.cost,
        "error_message": result.error_message,
        "path": path_json,
        "metrics": {
            "runtime_seconds": round(result.metrics.runtime_seconds, 4),
            "states_generated": result.metrics.states_generated,
            "states_expanded": result.metrics.states_expanded,
            "max_queue_size": result.metrics.max_queue_size,
            "path_improvements": result.metrics.path_improvements,
            "ida_iterations": result.metrics.ida_iterations,
        },
        "search_log": log_json,
        "ida_logs": ida_json,
    }
