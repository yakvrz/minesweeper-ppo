# Avoidability Diagnostics

The evaluation CLI now reports a set of deterministic metrics that capture when a reveal was
*forced* (no provably safe frontier remained) versus when the agent could have revealed a safe
cell but chose otherwise.

All deductions match the zero-flood closure implemented in the environment. For each decision:

1. **Constraint propagation** applies unit constraints and subset rules to mark obvious safe/mine
   cells.
2. **Component feasibility** runs a backtracking solver (or 0–1 feasibility checks for larger
   components) on each connected frontier. If any assignment proves that a cell cannot be a mine,
   it is marked as a guaranteed-safe option.

A step is labeled *forced* when no guaranteed-safe frontier cells exist. Otherwise it is labeled
*safe-option available*.

The metrics printed in `eval.py` are:

- `forced_guess_rate`: Fraction of reveal decisions that were truly forced.
- `forced_guess_success_rate`: Success rate (revealed safe) on those forced guesses.
- `forced_guess_episode_rate`: Share of episodes that encountered at least one forced guess.
- `safe_option_rate`: Complement of `forced_guess_rate`; how often a safe choice existed.
- `safe_option_pick_rate`: Among those safe-option steps, how often the agent actually picked a
  proven-safe cell (higher is better).
- `safe_option_miss_rate`: The flip side—avoidable mistakes where the agent ignored a safe cell.
- `avg_safe_options_per_turn`: Average number of guaranteed-safe cells available when a safe option
  existed.
- `avg_frontier_component_size`: Mean size of the frontier components considered by the solver.
- `avg_selected_component_size`: Mean size of the component containing the cell the agent picked.

These diagnostics are purely observational; the belief head is detached from the policy trunk, and
the solver’s results only affect logging.
