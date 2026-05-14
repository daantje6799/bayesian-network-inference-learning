"""map.py

@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

MAP / MPE inference using variable elimination.

Factors are represented as pandas DataFrames with one column per variable and a
final column named ``prob`` containing the factor values.
"""

from __future__ import annotations

import argparse
import io
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any, TextIO

import pandas as pd

from VE import build_elim_order


class MAP:
    def __init__(self, network: Any):
        """Create a MAP solver for a given BayesNet-like object."""

        self.network = network

    @staticmethod
    def _factor_to_dict(factor: pd.DataFrame) -> tuple[dict[tuple, float], list[str]]:
        """Convert factor DataFrame to dict: (var1_val, var2_val, ...) -> prob.
        Returns (factor_dict, variable_names)."""
        non_prob_cols = [c for c in factor.columns if c != "prob"]
        if not non_prob_cols:
            return {(): float(factor["prob"].iloc[0])}, []
        result = {}
        for _, row in factor.iterrows():
            key = tuple(row[c] for c in non_prob_cols)
            result[key] = float(row["prob"])
        return result, non_prob_cols

    @staticmethod
    def _dict_to_factor(factor_dict: dict[tuple, float], var_names: list[str]) -> pd.DataFrame:
        """Convert factor dict back to DataFrame."""
        if not var_names:
            return pd.DataFrame({"prob": [list(factor_dict.values())[0]]})
        rows = [
            {**{var: val for var, val in zip(var_names, key)}, "prob": prob}
            for key, prob in factor_dict.items()
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _multiply_two_factors_dict(
        f1_dict: dict[tuple, float],
        f2_dict: dict[tuple, float],
        vars1: list[str],
        vars2: list[str],
    ) -> tuple[dict[tuple, float], list[str]]:
        """Multiply two factors in dict form. Returns (result_dict, result_vars)."""
        common = [v for v in vars1 if v in vars2]
        result_vars = vars1 + [v for v in vars2 if v not in vars1]
        
        if common:
            common_idx1 = [vars1.index(c) for c in common]
            common_idx2 = [vars2.index(c) for c in common]
            non_common_idx2 = [i for i in range(len(vars2)) if i not in common_idx2]
            
            result = {}
            for key1, prob1 in f1_dict.items():
                common_key1 = tuple(key1[i] for i in common_idx1)
                for key2, prob2 in f2_dict.items():
                    common_key2 = tuple(key2[i] for i in common_idx2)
                    if common_key1 == common_key2:
                        merged_key = key1 + tuple(key2[i] for i in non_common_idx2)
                        result[merged_key] = prob1 * prob2
            return result, result_vars
        else:
            # Cartesian product
            result = {}
            for key1, prob1 in f1_dict.items():
                for key2, prob2 in f2_dict.items():
                    merged_key = key1 + key2
                    result[merged_key] = prob1 * prob2
            return result, result_vars

    def run_map(
        self,
        query_vars: Sequence[str],
        observed: Mapping[str, Any],
        elim_order: Sequence[str],
        logfile: str | TextIO | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Run MAP (MPE over `query_vars`) given evidence `observed`.

        The elimination order should generally eliminate all non-query, non-evidence
        variables first. Any query variables missing from `elim_order` are appended
        at the end and maximized out.
        """

        if isinstance(logfile, str):
            log_cm = open(logfile, "w", encoding="utf-8")
        elif logfile is None:
            log_cm = nullcontext(None)
        else:
            log_cm = nullcontext(logfile)
        with log_cm as log:
            if log:
                self._log_header(log, query_vars, observed, elim_order)

            # Build initial factors (CPTs) and restrict by evidence.
            factors: list[pd.DataFrame] = []
            for node in self.network.nodes:
                factor = self.network.probabilities[node].copy()
                factor = self._restrict_factor(factor, observed)
                factors.append(factor)

            elim_order_map = list(elim_order) + [q for q in query_vars if q not in elim_order]
            assignments: dict[str, pd.DataFrame] = {}

            for variable in elim_order_map:
                if variable in observed:
                    continue

                bucket = [f for f in factors if variable in f.columns]
                if not bucket:
                    if log:
                        log.write(f"Eliminating variable: {variable}\n")
                        log.write("  - No factors to process for this variable.\n\n")
                    continue

                factors = [f for f in factors if variable not in f.columns]

                if log:
                    log.write(f"Eliminating variable: {variable}\n")
                    log.write(f"  - Factors in bucket for {variable}:\n")
                    for f in bucket:
                        log.write(f"{f.to_string()}\n")

                product = self._multiply_factors(bucket)

                if log:
                    log.write(f"  - Product of factors for {variable}:\n{product.to_string()}\n")

                if variable in query_vars:
                    new_factor, assignment = self.maximize(product, variable)
                    assignments[variable] = assignment
                    if log:
                        log.write(f"  - Maximizing out {variable}:\n")
                        log.write(f"    - New factor:\n{new_factor.to_string()}\n")
                        log.write(f"    - Assignment:\n{assignment.to_string()}\n\n")
                else:
                    new_factor = self._sum_out(product, variable)
                    if log:
                        log.write(f"  - Summing out {variable}:\n")
                        log.write(f"    - New factor:\n{new_factor.to_string()}\n\n")

                factors.append(new_factor)

            final_factor = self._multiply_factors(factors) if factors else None
            final_prob = (
                float(final_factor["prob"].iloc[0])
                if final_factor is not None and not final_factor.empty
                else 1.0
            )

            mpe = self._backtrace_mpe(elim_order_map, query_vars, assignments, log)

            if log:
                log.write(f"\nFinal MPE: {mpe}\n")
                log.write(f"Probability: {final_prob}\n")

        return mpe, final_prob

    @staticmethod
    def _log_header(
        log: TextIO,
        query_vars: Sequence[str],
        observed: Mapping[str, Any],
        elim_order: Sequence[str],
    ) -> None:
        log.write(f"MAP Query: {list(query_vars)}\n")
        log.write(f"Evidence: {dict(observed)}\n")
        log.write(f"Elimination Order: {list(elim_order)}\n\n")

    @staticmethod
    def _restrict_factor(factor: pd.DataFrame, observed: Mapping[str, Any]) -> pd.DataFrame:
        for var, val in observed.items():
            if var in factor.columns:
                factor = factor[factor[var] == val].drop(columns=[var])
        return factor

    @staticmethod
    def _multiply_two_factors(f1: pd.DataFrame, f2: pd.DataFrame) -> pd.DataFrame:
        """Multiply two factors using optimized dict operations."""
        f1_dict, vars1 = MAP._factor_to_dict(f1)
        f2_dict, vars2 = MAP._factor_to_dict(f2)
        result_dict, result_vars = MAP._multiply_two_factors_dict(f1_dict, f2_dict, vars1, vars2)
        return MAP._dict_to_factor(result_dict, result_vars)

    def _multiply_factors(self, factors: Sequence[pd.DataFrame]) -> pd.DataFrame:
        if not factors:
            return pd.DataFrame({"prob": [1.0]})

        product = factors[0]
        for f in factors[1:]:
            product = self._multiply_two_factors(product, f)
        return product

    @staticmethod
    def _sum_out(factor: pd.DataFrame, variable: str) -> pd.DataFrame:
        """Sum out a variable using dict operations."""
        factor_dict, var_names = MAP._factor_to_dict(factor)
        if not var_names:
            return pd.DataFrame({"prob": [list(factor_dict.values())[0]]})
        
        var_idx = var_names.index(variable)
        remaining_vars = [v for i, v in enumerate(var_names) if i != var_idx]
        
        result = {}
        for key, prob in factor_dict.items():
            new_key = tuple(k for i, k in enumerate(key) if i != var_idx)
            result[new_key] = result.get(new_key, 0.0) + prob
        
        return MAP._dict_to_factor(result, remaining_vars)

    @staticmethod
    def maximize(factor: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Maximize out a variable using dict operations."""
        factor_dict, var_names = MAP._factor_to_dict(factor)
        if not var_names:
            return factor, factor[[]]
        
        var_idx = var_names.index(variable)
        remaining_vars = [v for i, v in enumerate(var_names) if i != var_idx]
        
        # Group by remaining vars, find max for each group
        groups: dict[tuple, tuple[float, Any]] = {}  # (remaining_key) -> (max_prob, max_val)
        for key, prob in factor_dict.items():
            remaining_key = tuple(k for i, k in enumerate(key) if i != var_idx)
            max_val = key[var_idx]
            if remaining_key not in groups or prob > groups[remaining_key][0]:
                groups[remaining_key] = (prob, max_val)
        
        # Build result and assignment
        new_factor_dict = {}
        assignment_dict = {}
        for remaining_key, (prob, max_val) in groups.items():
            new_factor_dict[remaining_key] = prob
            assignment_dict[remaining_key] = max_val
        
        new_factor = MAP._dict_to_factor(new_factor_dict, remaining_vars)
        
        # Build assignment table
        assignment_rows = []
        for remaining_key, max_val in assignment_dict.items():
            row = {**{var: val for var, val in zip(remaining_vars, remaining_key)}, variable: max_val}
            assignment_rows.append(row)
        assignment = pd.DataFrame(assignment_rows)
        
        return new_factor, assignment

    @staticmethod
    def _backtrace_mpe(
        elim_order_map: Sequence[str],
        query_vars: Sequence[str],
        assignments: Mapping[str, pd.DataFrame],
        log: TextIO | None,
    ) -> dict[str, Any]:
        """Backtrace to find MPE using dict-based constraint matching."""
        mpe: dict[str, Any] = {}
        if log:
            log.write("Backtracing to find MPE...\n")

        query_set = set(query_vars)
        for variable in reversed(elim_order_map):
            if variable not in query_set:
                continue

            assignment_table = assignments.get(variable)
            if assignment_table is None or assignment_table.empty:
                continue

            # Filter rows based on current MPE assignments
            filtered = assignment_table
            if mpe:
                # Build boolean mask for rows matching current MPE
                mask = pd.Series([True] * len(filtered))
                for mpe_var, mpe_val in mpe.items():
                    if mpe_var in filtered.columns:
                        mask = mask & (filtered[mpe_var] == mpe_val)
                filtered = filtered[mask]

            # If no rows match constraints, use the first row
            if filtered.empty:
                filtered = assignment_table

            mpe[variable] = filtered[variable].iloc[0]
            if log:
                log.write(f"  - For {variable}, found assignment {mpe[variable]}\n")

        return mpe


def run_part2_map(network: Any, map_evidence: Mapping[str, Any]) -> tuple[str, str]:
    """Run assignment Part 2 MAP and return (summary, detailed_log)."""

    map_query_vars = ["Burglary", "Earthquake"]
    elim_order_map = build_elim_order(network, set(map_query_vars) | set(map_evidence.keys()))

    mapper = MAP(network)
    log_buffer = io.StringIO()
    mpe, prob = mapper.run_map(map_query_vars, map_evidence, elim_order_map, logfile=log_buffer)

    summary = (
        f"Part 2 (MAP): query_vars={map_query_vars}, evidence={dict(map_evidence)}, "
        f"mpe={mpe}, probability={prob}"
    )
    return summary, log_buffer.getvalue().strip()


# BONUS FEATURE:
# Tractability benchmark for MAP queries on increasing query sizes.
def run_bonus_tractability(logfile: str | None = "map_bonus_tractability.txt") -> list[str]:
    from read_bayesnet import BayesNet

    benchmark_specs = [
        (
            "earthquake.bif",
            ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"],
        ),
        (
            "endorisk_new.bif",
            [
                "PR",
                "Platelets",
                "Survival1yr",
                "LVSI",
                "LNM",
                "CA125",
                "Histology",
                "p53",
                "PrimaryTumor",
                "Therapy",
            ],
        ),
    ]

    lines: list[str] = []
    lines.append("=== BONUS: MAP tractability check ===")

    for bif_name, query_pool in benchmark_specs:
        net = BayesNet(bif_name)
        mapper = MAP(net)
        lines.append(f"\n--- {bif_name} ---")
        for size in range(1, len(query_pool) + 1):
            query_vars = query_pool[:size]
            elim_order = [n for n in net.nodes if n not in query_vars]
            start = time.time()
            mpe, prob = mapper.run_map(query_vars, {}, elim_order)
            elapsed = time.time() - start
            lines.append(
                f"Q={size}: {elapsed:.4f}s, prob={prob:.6g}, mpe={mpe}"
            )

    if logfile:
        with open(logfile, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    return lines


def _main() -> None:
    parser = argparse.ArgumentParser(description="MAP utilities")
    parser.add_argument(
        "--bonus-tractability",
        action="store_true",
        help="Run bonus MAP tractability benchmark",
    )
    parser.add_argument(
        "--bonus-log",
        default="map_bonus_tractability.txt",
        help="Output log file for bonus benchmark",
    )
    args = parser.parse_args()

    if args.bonus_tractability:
        lines = run_bonus_tractability(args.bonus_log)
        print("\n".join(lines))
        print(f"\nBonus benchmark log written to: {args.bonus_log}")
    else:
        print("No action selected. Use --bonus-tractability to run the bonus benchmark.")


if __name__ == "__main__":
    _main()
