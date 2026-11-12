"""
Block World Solver — Flask Web Application
============================================
Serves the HTML UI and provides JSON API endpoints for:
- Validating start/goal states
- Listing available algorithms and heuristics
- Running a search and returning results
"""

from flask import Flask, jsonify, render_template, request
from solver import (
    SearchResult,
    SearchMetrics,
    ValidationError,
    get_algorithms,
    get_heuristics,
    make_state,
    move_to_json,
    run_search,
    search_result_to_json,
    state_to_json,
    validate_pair,
)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.route("/api/validate", methods=["POST"])
def api_validate():
    """
    Validate a (start, goal) pair.
    Expects JSON: { start: [[...], ...], goal: [[...], ...] }
    """
    data = request.get_json(force=True)
    start_stacks = data.get("start", [])
    goal_stacks = data.get("goal", [])

    try:
        start, goal = validate_pair(start_stacks, goal_stacks)
    except ValidationError as e:
        return jsonify({"valid": False, "error": str(e)}), 400

    return jsonify({
        "valid": True,
        "start": state_to_json(start),
        "goal": state_to_json(goal),
        "blocks": sorted({b for s in start for b in s}),
    })


@app.route("/api/options", methods=["GET"])
def api_options():
    """Return available algorithms and heuristics."""
    return jsonify({
        "algorithms": get_algorithms(),
        "heuristics": get_heuristics(),
    })


@app.route("/api/solve", methods=["POST"])
def api_solve():
    """
    Run a search.
    Expects JSON: {
        start: [[...], ...],
        goal: [[...], ...],
        algorithm: "...",
        heuristic: "..."
    }
    """
    data = request.get_json(force=True)
    start_stacks = data.get("start", [])
    goal_stacks = data.get("goal", [])
    algo_name = data.get("algorithm", "")
    heur_name = data.get("heuristic", "")

    # Validate first
    try:
        start, goal = validate_pair(start_stacks, goal_stacks)
    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    # Check if start == goal
    if start == goal:
        result = SearchResult(
            success=True,
            path=[(start, None)],
            cost=0,
            metrics=SearchMetrics(),
        )
        return jsonify(search_result_to_json(result))

    # Run the search
    result = run_search(algo_name, heur_name, start, goal)

    if not result.success:
        return jsonify(search_result_to_json(result)), 400 if result.error_message else 200

    return jsonify(search_result_to_json(result))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  Block World Solver")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
