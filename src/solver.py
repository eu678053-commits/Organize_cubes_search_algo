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
import heapq


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

# A stack is a tuple of block names, ordered bottom-to-top.
Stack = Tuple[str, ...]

# A state is a frozenset of stacks (order on the table is irrelevant).
State = frozenset  # frozenset[Stack]

# Run time limit of algorithms - A*, IDA*
RUN_TIME_LIMIT = 30.0

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
    expansions: int = 0
    prunings: int = 0
    next_threshold: float = float("inf")
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

class _SearchTimeout(Exception):
    pass

@register_algorithm("IDA*")
def idastar(
    start: State,
    goal: State,
    heuristic: Callable[[State, State], float],
) -> SearchResult:

    t0 = time.perf_counter()
    metrics = SearchMetrics()
    ida_logs: List[IDAIterationLog] = []

    # Initial threshold
    threshold = heuristic(start, goal)
    def dfs_search(current_state: State, g: int, threshold: float) -> Tuple[Optional[List[Tuple[State, Optional[Move]]]], float]:
        if time.perf_counter() - t0 > RUN_TIME_LIMIT:
            iter_log.next_threshold = iter_log.threshold
            ida_logs.append(iter_log)
            raise _SearchTimeout
        f = g + heuristic(current_state, goal)
        if f > threshold:
            iter_log.prunings += 1
            if len(iter_log.sample_prunings) < 10:
                iter_log.sample_prunings.append({
                    "state": state_to_json(current_state),
                    "g": g,
                    "h": f - g,
                    "f": f,
                })
            return None, f
        if current_state == goal:
            return list(current_path), f

        metrics.states_expanded += 1
        iter_log.expansions += 1
        if len(iter_log.sample_expansions) < 10:
            iter_log.sample_expansions.append({
                "state": state_to_json(current_state),
                "g": g,
                "h": f - g,
                "f": f,
            })

        min_next_threshold = float("inf")

        successors = sorted(
            get_successors(current_state),
            key=lambda item: (g + 1 + heuristic(item[0], goal), _canonical_key(item[0])),
        )
        for child, move in successors:
            metrics.states_generated += 1
            if child in path_set:
                continue
            path_set.add(child)
            current_path.append((child, move))

            res_path, child_next_threshold = dfs_search(child, g + 1, threshold)
            path_set.remove(child)
            current_path.pop()

            if res_path is not None:
                return (res_path, child_next_threshold)
            else:
                min_next_threshold = min(min_next_threshold, child_next_threshold)
        return (None, min_next_threshold)

    try:
        while True:
            metrics.ida_iterations += 1

            iter_log = IDAIterationLog(
                iteration=metrics.ida_iterations,
                threshold=threshold
            )
            current_path = [(start, None)]
            path_set = {start}

            found_path, next_threshold = dfs_search(start, 0, threshold) # Start from the beginning
            iter_log.next_threshold = next_threshold
            ida_logs.append(iter_log)

            if found_path is not None:
                return SearchResult(
                    success=True,
                    path=found_path,
                    cost=len(found_path) - 1,
                    metrics=metrics,
                    ida_logs=ida_logs,
                )
            elif next_threshold == float("inf"):
                return SearchResult(
                    success=False,
                    path=[],
                    cost=0,
                    metrics=metrics,
                    ida_logs=ida_logs,
                    error_message="לא קיים פתרון"
                )
            else:
                threshold = next_threshold
    except _SearchTimeout:
        return SearchResult(
            success=False,
            path=[],
            cost=0,
            metrics=metrics,
            ida_logs=ida_logs,
            error_message="הריצה הופסקה במגבלה",
        )

@register_algorithm("A*")
def astar(
    start: State,
    goal: State,
    heuristic: Callable[[State, State], float],
) -> SearchResult:
    t0 = time.perf_counter()
    metrics = SearchMetrics()
    log_entries: List[SearchLogEntry] = []

    parent: Dict[State, Tuple[Optional[State], Optional[Move]]] = {start: (None, None)}
    best_cost: Dict[State, int] = {start: 0}

    open_heap = []
    _push_open(open_heap, heuristic(start, goal), 0, start)
    metrics.max_queue_size = 1

    try:
        while open_heap:
            if time.perf_counter() - t0 > RUN_TIME_LIMIT:
                raise _SearchTimeout

            f, negative_g, _, current_state = heapq.heappop(open_heap)
            g = -negative_g

            if g > best_cost[current_state]:
                continue

            _, incoming_move = parent[current_state]
            action_desc = incoming_move.describe() if incoming_move else None

            if current_state == goal:
                log_entries.append(
                    SearchLogEntry(
                        step=len(log_entries) + 1,
                        state=current_state,
                        g=g,
                        h=f - g,
                        f=f,
                        queue_size=len(open_heap),
                        action=action_desc,
                        children_info=[],
                    )
                )
                return SearchResult(
                    success=True,
                    path=_reconstruct_path(current_state, parent),
                    cost=g,
                    metrics=metrics,
                    search_log=log_entries,
                )

            metrics.states_expanded += 1
            children_info = []

            for child, move in get_successors(current_state):
                metrics.states_generated += 1
                info = _process_successor(
                    current_state, child, move, g + 1, goal,
                    heuristic, best_cost, parent, open_heap, metrics,
                )
                children_info.append(info)

            metrics.max_queue_size = max(metrics.max_queue_size, len(open_heap))
            log_entries.append(
                SearchLogEntry(
                    step=len(log_entries) + 1,
                    state=current_state,
                    g=g,
                    h=f - g,
                    f=f,
                    queue_size=len(open_heap),
                    action=action_desc,
                    children_info=children_info,
                )
            )

        return SearchResult(
            success=False,
            path=[],
            cost=0,
            metrics=metrics,
            search_log=log_entries,
            error_message="לא קיים פתרון",
        )
    except _SearchTimeout:
        return SearchResult(
            success=False,
            path=[],
            cost=0,
            metrics=metrics,
            search_log=log_entries,
            error_message="הריצה הופסקה במגבלה",
        )

def _process_successor(
    current_state: State,
    child: State,
    move: Move,
    new_g: int,
    goal: State,
    heuristic: Callable[[State, State], float],
    best_cost: Dict[State, int],
    parent: Dict[State, Tuple[Optional[State], Optional[Move]]],
    open_heap: list,
    metrics: SearchMetrics,
) -> Dict:
    """בדיקת יורש בודד (נוסף / שופר / נדחה), עדכון מבני הנתונים והחזרת רשומה ליומן."""
    h_child = heuristic(child, goal)
    f_child = new_g + h_child

    if child not in best_cost:
        best_cost[child] = new_g
        parent[child] = (current_state, move)
        _push_open(open_heap, f_child, new_g, child)
        status = "added"
    elif new_g < best_cost[child]:
        metrics.path_improvements += 1
        best_cost[child] = new_g
        parent[child] = (current_state, move)
        _push_open(open_heap, f_child, new_g, child)
        status = "improved"
    else:
        status = "rejected"

    return {
        "state": state_to_json(child),
        "move": move.describe(),
        "g": new_g,
        "h": h_child,
        "f": f_child,
        "status": status,
    }

def _reconstruct_path(
    current_state: State,
    parent: Dict[State, Tuple[Optional[State], Optional[Move]]]
    )-> List[Tuple[State, Optional[Move]]]:
    path =[]
    while current_state is not None:
        previous_state, move = parent[current_state]  # extract (parent_state, move)
        path.append((current_state, move))
        current_state = previous_state

    # Flip the list in order the initial state will be in start
    path.reverse()
    return path

def _canonical_key(state: State) -> Tuple[Tuple[str, ...], ...]:
    """ייצוג קנוני כ-tuple מקונן לצורך שבירת שוויון מילונית בתור העדיפויות."""
    return tuple(tuple(s) for s in state_to_sorted_lists(state))


def _push_open(open_heap: list, f: float, g: int, state: State) -> None:
    """דחיפת מצב לתור העדיפויות לפי f מינימלי, g מקסימלי וסדר מילוני."""
    heapq.heappush(open_heap, (f, -g, _canonical_key(state), state))


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

@register_heuristic("h1")
def heuristic_one(state: State, goal: State) -> float:
    sup_s = _get_direct_supports(state)
    sup_g = _get_direct_supports(goal)

    wrong_count = sum(1 for block in sup_s if sup_s[block] != sup_g[block])
    return float(wrong_count)

@register_heuristic("h2")
def heuristic_two(state: State, goal: State) -> float:
    sup_s = _get_direct_supports(state)
    sup_g = _get_direct_supports(goal)

    wrong_count = 0

    for stack in state:
        wrong_below = False
        for block in stack:
            if wrong_below:
                wrong_count += 1
            elif sup_s[block] != sup_g[block]:
                wrong_below = True
                wrong_count += 1
    return float(wrong_count)


def _get_direct_supports(state: State) -> Dict[str, Optional[str]]:
    supports: Dict[str, Optional[str]] = {}
    for stack in state:
        for i, block in enumerate(stack):
            if i == 0:
                supports[block] = None
            else:
                supports[block] = stack[i - 1]
    return supports

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