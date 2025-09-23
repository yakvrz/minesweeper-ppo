from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .env import MinesweeperEnv


_NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


@dataclass
class AvoidabilityResult:
    avoidable: bool
    forced_safe_cells: Set[int]
    component_sizes: List[int]
    chosen_is_forced_safe: bool
    chosen_component_size: Optional[int]

    @property
    def count_forced_safe_cells(self) -> int:
        return len(self.forced_safe_cells)


@dataclass(frozen=True)
class _Constraint:
    vars: Tuple[int, ...]
    target: int


@dataclass(frozen=True)
class _LocalConstraint:
    vars: Tuple[int, ...]
    target: int


class _ConstraintSolver:
    def __init__(self, num_vars: int, constraints: Sequence[_LocalConstraint]):
        self.num_vars = num_vars
        self.constraints = list(constraints)
        self.targets = [c.target for c in self.constraints]
        self.constraint_vars = [list(c.vars) for c in self.constraints]
        self.var_to_constraints: List[List[int]] = [[] for _ in range(num_vars)]
        for c_idx, vars_list in enumerate(self.constraint_vars):
            for var in vars_list:
                self.var_to_constraints[var].append(c_idx)
        # Branch higher-degree variables first for faster pruning.
        self.order = sorted(range(num_vars), key=lambda v: len(self.var_to_constraints[v]), reverse=True)

    def is_feasible(self, forced: Optional[Dict[int, int]] = None) -> bool:
        assignment = [None] * self.num_vars
        assigned_sum = [0] * len(self.constraints)
        unknown_count = [len(vars_list) for vars_list in self.constraint_vars]

        forced = forced or {}
        forced_stack: List[Tuple[int, List[Tuple[int, int, int]]]] = []
        for var, value in forced.items():
            ok, changes = self._assign(var, value, assignment, assigned_sum, unknown_count)
            if not ok:
                return False
            forced_stack.append((var, changes))

        def dfs(pos: int) -> bool:
            if pos == len(self.order):
                return True
            var = self.order[pos]
            if assignment[var] is not None:
                return dfs(pos + 1)
            for value in (0, 1):
                ok, changes = self._assign(var, value, assignment, assigned_sum, unknown_count)
                if not ok:
                    continue
                if dfs(pos + 1):
                    return True
                self._revert(var, assignment, assigned_sum, unknown_count, changes)
            return False

        feasible = dfs(0)

        # Revert forced assignments to keep solver reusable.
        for var, changes in reversed(forced_stack):
            self._revert(var, assignment, assigned_sum, unknown_count, changes)

        return feasible

    def forced_safe_variables(self) -> Set[int]:
        forced_safe: Set[int] = set()
        for var in range(self.num_vars):
            if not self.is_feasible({var: 1}):
                forced_safe.add(var)
        return forced_safe

    def _assign(
        self,
        var: int,
        value: int,
        assignment: List[Optional[int]],
        assigned_sum: List[int],
        unknown_count: List[int],
    ) -> Tuple[bool, List[Tuple[int, int, int]]]:
        current = assignment[var]
        if current is not None:
            return current == value, []

        changes: List[Tuple[int, int, int]] = []
        for c_idx in self.var_to_constraints[var]:
            prev_assigned = assigned_sum[c_idx]
            prev_unknown = unknown_count[c_idx]
            new_assigned = prev_assigned + (1 if value == 1 else 0)
            new_unknown = prev_unknown - 1
            assigned_sum[c_idx] = new_assigned
            unknown_count[c_idx] = new_unknown
            changes.append((c_idx, prev_assigned, prev_unknown))
            target = self.targets[c_idx]
            if new_assigned > target or new_assigned + new_unknown < target:
                for idx, a_prev, u_prev in reversed(changes):
                    assigned_sum[idx] = a_prev
                    unknown_count[idx] = u_prev
                return False, []

        assignment[var] = value
        return True, changes

    @staticmethod
    def _revert(
        var: int,
        assignment: List[Optional[int]],
        assigned_sum: List[int],
        unknown_count: List[int],
        changes: Iterable[Tuple[int, int, int]],
    ) -> None:
        assignment[var] = None
        for c_idx, prev_assigned, prev_unknown in reversed(list(changes)):
            assigned_sum[c_idx] = prev_assigned
            unknown_count[c_idx] = prev_unknown


def analyze_avoidability(
    env: MinesweeperEnv,
    chosen_cell: Optional[int],
    *,
    component_threshold: int = 22,
) -> AvoidabilityResult:
    H, W = env.H, env.W

    if not env.first_click_done:
        return AvoidabilityResult(
            avoidable=True,
            forced_safe_cells=set(),
            component_sizes=[],
            chosen_is_forced_safe=False,
            chosen_component_size=None,
        )

    revealed = env.revealed
    hidden = (~env.revealed) & (~env.flags)

    frontier_mask = np.zeros_like(revealed, dtype=bool)
    for r in range(H):
        for c in range(W):
            if not hidden[r, c]:
                continue
            for dr, dc in _NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and revealed[nr, nc]:
                    frontier_mask[r, c] = True
                    break

    frontier_coords = np.argwhere(frontier_mask)
    if frontier_coords.size == 0:
        chosen_component_size = None
        if chosen_cell is not None:
            chosen_component_size = 1 if not env.revealed.flat[chosen_cell] else None
        return AvoidabilityResult(
            avoidable=False,
            forced_safe_cells=set(),
            component_sizes=[],
            chosen_is_forced_safe=False,
            chosen_component_size=chosen_component_size,
        )

    frontier_cells: List[Tuple[int, int]] = []
    var_index: Dict[Tuple[int, int], int] = {}
    for idx, (r, c) in enumerate(frontier_coords):
        r_i = int(r)
        c_i = int(c)
        frontier_cells.append((r_i, c_i))
        var_index[(r_i, c_i)] = idx

    constraints: List[_Constraint] = []
    for r in range(H):
        for c in range(W):
            if not revealed[r, c]:
                continue
            if env.mine_mask[r, c]:
                continue
            vars_for_constraint: List[int] = []
            for dr, dc in _NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and frontier_mask[nr, nc]:
                    vars_for_constraint.append(var_index[(nr, nc)])
            if vars_for_constraint:
                constraints.append(_Constraint(tuple(vars_for_constraint), int(env.adjacent_counts[r, c])))

    # Build component map on the full frontier (prior to deductions).
    adjacency: List[Set[int]] = [set() for _ in range(len(frontier_cells))]
    for constraint in constraints:
        vars_tuple = constraint.vars
        for i, var in enumerate(vars_tuple):
            for other in vars_tuple[i + 1 :]:
                adjacency[var].add(other)
                adjacency[other].add(var)

    component_ids: Dict[int, int] = {}
    components: List[List[int]] = []
    for var in range(len(frontier_cells)):
        if var in component_ids:
            continue
        queue = [var]
        component_ids[var] = len(components)
        comp_vars: List[int] = []
        while queue:
            current = queue.pop()
            comp_vars.append(current)
            for neigh in adjacency[current]:
                if neigh not in component_ids:
                    component_ids[neigh] = component_ids[var]
                    queue.append(neigh)
        components.append(comp_vars)

    component_sizes = [len(comp) for comp in components]

    chosen_is_forced_safe = False
    chosen_component_size: Optional[int] = None
    chosen_var: Optional[int] = None
    if chosen_cell is not None:
        row, col = divmod(int(chosen_cell), W)
        chosen_var = var_index.get((row, col))
        if chosen_var is not None:
            comp_idx = component_ids.get(chosen_var)
            if comp_idx is not None:
                chosen_component_size = component_sizes[comp_idx]

    assignments: Dict[int, int] = {}

    def _constraint_remaining(constraint: _Constraint) -> Tuple[List[int], int]:
        remaining: List[int] = []
        target = constraint.target
        for var in constraint.vars:
            value = assignments.get(var)
            if value is None:
                remaining.append(var)
            elif value == 1:
                target -= 1
        return remaining, target

    forced_safe_vars: Set[int] = set()
    forced_mine_vars: Set[int] = set()

    changed = True
    while changed:
        changed = False
        # Unit propagation.
        for constraint in constraints:
            remaining, target = _constraint_remaining(constraint)
            if target < 0 or target > len(remaining):
                continue
            if target == 0:
                for var in remaining:
                    if var not in assignments:
                        assignments[var] = 0
                        forced_safe_vars.add(var)
                        changed = True
            elif target == len(remaining):
                for var in remaining:
                    if var not in assignments:
                        assignments[var] = 1
                        forced_mine_vars.add(var)
                        changed = True
        if changed:
            continue

        # Subset rule.
        for i, constraint_a in enumerate(constraints):
            remaining_a, target_a = _constraint_remaining(constraint_a)
            if not remaining_a:
                continue
            set_a = set(remaining_a)
            for j, constraint_b in enumerate(constraints):
                if i == j:
                    continue
                remaining_b, target_b = _constraint_remaining(constraint_b)
                if not remaining_b:
                    continue
                set_b = set(remaining_b)
                if not set_a.issubset(set_b):
                    continue
                diff = set_b - set_a
                if not diff:
                    continue
                if target_a == target_b:
                    for var in diff:
                        if var not in assignments:
                            assignments[var] = 0
                            forced_safe_vars.add(var)
                            changed = True
                    if changed:
                        break
                elif target_b - target_a == len(diff):
                    for var in diff:
                        if var not in assignments:
                            assignments[var] = 1
                            forced_mine_vars.add(var)
                            changed = True
                    if changed:
                        break
            if changed:
                break

    if forced_safe_vars and chosen_var is not None and chosen_var in forced_safe_vars:
        chosen_is_forced_safe = True

    remaining_constraints: List[_Constraint] = []
    for constraint in constraints:
        remaining, target = _constraint_remaining(constraint)
        if not remaining:
            continue
        remaining_constraints.append(_Constraint(tuple(remaining), target))

    if forced_safe_vars:
        forced_safe_cells = {frontier_cells[var][0] * W + frontier_cells[var][1] for var in forced_safe_vars}
        return AvoidabilityResult(
            avoidable=True,
            forced_safe_cells=forced_safe_cells,
            component_sizes=component_sizes,
            chosen_is_forced_safe=chosen_is_forced_safe,
            chosen_component_size=chosen_component_size,
        )

    additional_forced_safe: Set[int] = set()

    constraints_by_component: Dict[int, List[_Constraint]] = {}
    for constraint in remaining_constraints:
        comp_idx = component_ids[constraint.vars[0]]
        constraints_by_component.setdefault(comp_idx, []).append(constraint)

    for comp_idx, comp_vars in enumerate(components):
        free_vars = [var for var in comp_vars if var not in assignments]
        if not free_vars:
            continue

        relevant_constraints = constraints_by_component.get(comp_idx)
        if not relevant_constraints:
            continue

        var_map = {var: i for i, var in enumerate(free_vars)}
        local_constraints = [
            _LocalConstraint(tuple(var_map[var] for var in constraint.vars), constraint.target)
            for constraint in relevant_constraints
        ]

        solver = _ConstraintSolver(len(free_vars), local_constraints)

        if len(free_vars) <= component_threshold:
            forced_local = solver.forced_safe_variables()
            additional_forced_safe.update(free_vars[idx] for idx in forced_local)
        else:
            for local_idx, global_var in enumerate(free_vars):
                if not solver.is_feasible({local_idx: 1}):
                    additional_forced_safe.add(global_var)

    total_forced_safe = forced_safe_vars.union(additional_forced_safe)

    if chosen_var is not None and chosen_var in total_forced_safe:
        chosen_is_forced_safe = True

    forced_safe_cells = {frontier_cells[var][0] * W + frontier_cells[var][1] for var in total_forced_safe}

    avoidable = bool(forced_safe_cells)

    return AvoidabilityResult(
        avoidable=avoidable,
        forced_safe_cells=forced_safe_cells,
        component_sizes=component_sizes,
        chosen_is_forced_safe=chosen_is_forced_safe,
        chosen_component_size=chosen_component_size,
    )
