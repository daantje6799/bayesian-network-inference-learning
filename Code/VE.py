"""Variable elimination algorithm for Bayesian networks."""

import time
import pandas as pd
from pathlib import Path
from read_bayesnet import BayesNet


class VariableElimination:

    def __init__(self, network):
        self.network = network

    def run(self, query, observed, elim_order):
        factors = [self._restrict_factor(self.network.probabilities[n].copy(), observed)
                   for n in self.network.nodes]
        for var in elim_order:
            if var == query or var in observed: continue
            bucket = [f for f in factors if var in f.columns]
            if not bucket: continue
            factors = [f for f in factors if var not in f.columns]
            prod = self._multiply_factors(bucket)
            if prod is not None: factors.append(self._sum_out(prod, var))

        result = self._multiply_factors(factors)
        if result is None or result.empty:
            return pd.DataFrame(columns=[query, 'prob'])
        result['prob'] /= result['prob'].sum()
        return result[[query, 'prob']]

    @staticmethod
    def _restrict_factor(factor, observed):
        for var, val in observed.items():
            if var in factor.columns:
                factor = factor[factor[var] == val].drop(columns=[var])
        return factor

    @staticmethod
    def _multiply_two_factors(f1, f2):
        """Optimized factor multiplication using pandas operations."""
        vars1 = [c for c in f1.columns if c != 'prob']
        vars2 = [c for c in f2.columns if c != 'prob']
        common = [c for c in vars1 if c in vars2]
        
        if common:
            # Merge on common columns
            merged = pd.merge(f1, f2, on=common, suffixes=('_x', '_y'))
        else:
            # Cartesian product for non-overlapping factors
            merged = pd.merge(f1.assign(_key=1), f2.assign(_key=1), on='_key').drop('_key', axis=1)
        
        # Multiply probabilities (handle both single and dual prob columns from merge)
        prob_cols = [c for c in merged.columns if c.startswith('prob')]
        if len(prob_cols) == 2:
            merged['prob'] = merged[prob_cols[0]] * merged[prob_cols[1]]
            merged = merged.drop(columns=prob_cols)
        
        return merged

    def _multiply_factors(self, factors):
        if not factors: return None
        prod = factors[0]
        for f in factors[1:]: prod = self._multiply_two_factors(prod, f)
        return prod

    @staticmethod
    def _sum_out(factor, variable):
        group_cols = [c for c in factor.columns if c not in (variable, 'prob')]
        if not group_cols:
            return pd.DataFrame({'prob': [factor['prob'].sum()]})
        return factor.groupby(group_cols)['prob'].sum().reset_index()

    def run_joint(self, query_vars, observed, elim_order, return_evidence_prob=False):
        query_set = set(query_vars)
        factors = [self._restrict_factor(self.network.probabilities[n].copy(), observed)
                   for n in self.network.nodes]

        for Z in elim_order:
            if Z in query_set or Z in observed: continue
            bucket = [f for f in factors if Z in f.columns]
            if not bucket: continue
            factors = [f for f in factors if Z not in f.columns]
            prod = self._multiply_factors(bucket)
            if prod is not None: factors.append(self._sum_out(prod, Z))

        result = self._multiply_factors(factors) if factors else None
        if result is None or result.empty:
            return pd.DataFrame({'prob': []})
        if not query_vars:
            return pd.DataFrame({'prob': [result['prob'].sum()]})

        total = float(result['prob'].sum())
        if total: result['prob'] /= total
        posterior = result[[v for v in query_vars if v in result.columns] + ['prob']]
        return (posterior, total) if return_evidence_prob else posterior

    def evidence_prob(self, observed, elim_order):
        result = self.run_joint([], observed, elim_order)
        factor = result[0] if isinstance(result, tuple) else result
        return 0.0 if factor.empty else float(factor['prob'].iloc[0])


# --- Heuristics ---

def _neighbors(network, var):
    """Get all neighbors of a variable (variables in the same factors)."""
    neighbors = set()
    for node in network.nodes:
        if var in network.probabilities[node].columns:
            for col in network.probabilities[node].columns:
                if col not in (var, 'prob'):
                    neighbors.add(col)
    return neighbors


def min_fill_heuristic(network, remaining_vars):
    """Min-fill heuristic: choose variable that creates fewest new edges."""
    def fill(var):
        nbrs = list(_neighbors(network, var))
        # Count pairs of neighbors that aren't already connected
        count = 0
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                # Check if nbrs[i] and nbrs[j] appear together in any factor
                connected = False
                for node in network.nodes:
                    cpt_cols = {c for c in network.probabilities[node].columns if c != 'prob'}
                    if nbrs[i] in cpt_cols and nbrs[j] in cpt_cols:
                        connected = True
                        break
                if not connected:
                    count += 1
        return count
    return min(remaining_vars, key=fill)


def least_degree_heuristic(network, remaining_vars):
    """Choose variable with fewest neighbors."""
    return min(remaining_vars, key=lambda v: len(_neighbors(network, v)))


def fewest_factors_heuristic(network, remaining_vars):
    """Choose variable that appears in fewest factors."""
    return min(remaining_vars, key=lambda v: sum(1 for n in network.nodes
                                                  if v in network.probabilities[n].columns))

def pick_heuristic(name):
    return {'least_degree': least_degree_heuristic,
            'fewest_factors': fewest_factors_heuristic}.get(name, min_fill_heuristic)

def build_elim_order(network, protected, heuristic='min_fill'):
    remaining = set(network.nodes) - protected
    choose = pick_heuristic(heuristic)
    order = []
    while remaining:
        var = choose(network, remaining)
        order.append(var)
        remaining.remove(var)
    return order


# --- Part 1 entry points ---

def run_part1_ve(net: BayesNet, heuristic: str = 'min_fill') -> tuple[str, dict]:
    ve = VariableElimination(net)
    query, evidence = 'Alarm', {'Burglary': 'True'}
    result = ve.run(query, evidence, build_elim_order(net, {query}, heuristic=heuristic))
    return (f"Part 1 (VE): query={query}, evidence={evidence}, "
            f"distribution={result.to_dict(orient='records')}, heuristic={heuristic}"), evidence


def run_part1_bonus(dataset_path: Path, earthquake_bif_path: Path) -> str:
    lines = ["=== BONUS: Part 1 checks ==="]

    bonus_net = BayesNet(str(dataset_path))
    multi_var = next((n for n in bonus_net.nodes if len(bonus_net.values[n]) > 2), None)
    if multi_var:
        dist = VariableElimination(bonus_net).run(
            multi_var, {}, build_elim_order(bonus_net, {multi_var}))
        lines.append(f"Non-binary check: query={multi_var} states={len(bonus_net.values[multi_var])} "
                     f"distribution_rows={len(dist)}")
    else:
        lines.append("Non-binary check: no variable with >2 states found")

    query, evidence = 'Alarm', {'Burglary': 'True'}
    for h in ('min_fill', 'least_degree', 'fewest_factors'):
        net = BayesNet(str(earthquake_bif_path))
        ve = VariableElimination(net)
        t = time.perf_counter()
        order = build_elim_order(net, {query}, heuristic=h)
        result = ve.run(query, evidence, order)
        elapsed = time.perf_counter() - t
        p = result.loc[result[query] == 'True', 'prob']
        lines.append(f"Heuristic={h} elapsed={elapsed:.4f}s "
                     f"P({query}=True|{evidence})={float(p.iloc[0]) if not p.empty else float('nan'):.6f} "
                     f"order={order}")

    return '\n'.join(lines)

