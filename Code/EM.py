"""EM for Bayesian network — learns CPT parameters from data with hidden variables."""
from __future__ import annotations
import argparse, io, itertools, math, random, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import pandas as pd
from read_bayesnet import BayesNet
from VE import VariableElimination, build_elim_order

NAME_ALIASES = {
    "PR": "Progesteron receptor", "Platelets": "Platelets", "Survival1yr": "Survival 1yr",
    "LVSI": "LVSI", "LNM": "Lymph node metastasis", "CA125": "CA125",
    "Histology": "Postoperative grade", "p53": "p53", "PrimaryTumor": "Preoperative grade",
    "Therapy": "Adjuvant therapy", "Survival3yr": "Survival 3yr", "L1CAM": "L1CAM",
    "MyometrialInvasion": "Myometrial invasion", "ER": "Estrogen receptor",
    "Recurrence": "Recurrence", "Cytology": "Endometrium in cervical cytology",
    "Survival5yr": "Survival 5yr", "CTMRI": "Enlarged nodes CT",
}
_canon = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
min_fill_order = lambda net: build_elim_order(net, set(), heuristic="min_fill")
load_dataset = lambda path: pd.read_csv(path, sep="\t", dtype=str)


def align_data_columns(data: pd.DataFrame, net: BayesNet) -> pd.DataFrame:
    data = data.copy()
    inv = {v: k for k, v in NAME_ALIASES.items()}
    canon_map = {_canon(n): n for n in net.nodes}
    rename = {}
    for c in data.columns:
        if c in net.nodes: continue
        rename[c] = (inv.get(c) or NAME_ALIASES.get(c) or canon_map.get(_canon(c)) or c)
    return data.rename(columns={k: v for k, v in rename.items() if v != k})


def detect_hidden_vars(net: BayesNet, data: pd.DataFrame) -> set[str]:
    return (set(net.nodes) - set(data.columns)) | {
        n for n in net.nodes if n in data.columns and data[n].isna().all()
    }


def _dirichlet(rng: random.Random, k: int) -> list[float]:
    d = [rng.random() + 1e-12 for _ in range(k)]
    s = sum(d); return [x / s for x in d]


def randomize_cpt(net: BayesNet, node: str, rng: random.Random) -> None:
    cpt, parents, vals = net.probabilities[node].copy(), net.parents[node], net.values[node]
    if not parents:
        for v, p in zip(vals, _dirichlet(rng, len(vals))):
            cpt.loc[cpt[node] == v, "prob"] = p
    else:
        g = parents[0] if len(parents) == 1 else parents
        for _, sub in cpt.groupby(g, sort=False):
            for v, p in zip(vals, _dirichlet(rng, len(vals))):
                cpt.loc[sub.index[sub[node] == v], "prob"] = p
    net.probabilities[node] = cpt


def mle_cpt(net: BayesNet, node: str, data: pd.DataFrame, alpha: float = 1e-6) -> pd.DataFrame:
    """Compute MLE CPT from data. Optimized with dictionary-based counting."""
    parents, template = net.parents[node], net.probabilities[node].copy()
    key_cols = [c for c in template.columns if c != "prob"]
    
    # Build counts dictionary directly
    counts_dict = {}
    for row_vals in zip(*(data[col] for col in [node] + parents)):
        key = tuple(row_vals)
        counts_dict[key] = counts_dict.get(key, 0) + 1
    
    # Fill in template with counts and compute probabilities
    if not parents:
        total = sum(counts_dict.values()) + alpha * len(net.values[node])
        probs = []
        for v in net.values[node]:
            count = counts_dict.get((v,), 0)
            probs.append((count + alpha) / total)
        template["prob"] = probs
        return template
    
    # Group by parents and compute probabilities
    parent_groups = {}
    for key, count in counts_dict.items():
        parent_key = key[1:]  # Skip node value
        if parent_key not in parent_groups:
            parent_groups[parent_key] = {}
        parent_groups[parent_key][key] = count
    
    probs = []
    for key_tuple in [tuple(r) for r in template[key_cols].itertuples(index=False, name=None)]:
        parent_key = tuple(key_tuple[i] for i in range(1, len(key_tuple)))
        count = parent_groups.get(parent_key, {}).get(key_tuple, 0)
        parent_total = sum(parent_groups.get(parent_key, {}).values()) + alpha * len(net.values[node])
        probs.append((count + alpha) / parent_total)
    
    template["prob"] = probs
    return template


def cpt_mean_abs_diff(orig: pd.DataFrame, learned: pd.DataFrame) -> tuple[float, float]:
    key_cols = [c for c in orig.columns if c != "prob"]
    m = pd.merge(orig[key_cols + ["prob"]].rename(columns={"prob": "o"}),
                 learned[key_cols + ["prob"]].rename(columns={"prob": "l"}), on=key_cols)
    if m.empty: return float("nan"), float("nan")
    d = (m["o"] - m["l"]).abs()
    return float(d.mean()), float(d.max())


def write_bif(net: BayesNet, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("network unknown {\n}\n")
        for v in net.nodes:
            f.write(f"variable {v} {{\n  type discrete [ {len(net.values[v])} ] {{ {', '.join(net.values[v])} }};\n}}\n")
        for v in net.nodes:
            parents, cpt = net.parents[v], net.probabilities[v]
            if parents:
                f.write(f"probability ( {v} | {', '.join(parents)} ) {{\n")
                for pvals in itertools.product(*[net.values[p] for p in parents]):
                    sub = cpt
                    for pn, pv in zip(parents, pvals): sub = sub[sub[pn] == pv]
                    probs = [float(sub[sub[v] == vv]["prob"].iloc[0]) for vv in net.values[v]]
                    f.write(f"  ({', '.join(pvals)}) {', '.join(f'{x:.10g}' for x in probs)};\n")
                f.write("}\n")
            else:
                probs = [float(cpt[cpt[v] == vv]["prob"].iloc[0]) for vv in net.values[v]]
                f.write(f"probability ( {v} ) {{\n  table {', '.join(f'{x:.10g}' for x in probs)};\n}}\n")


@dataclass
class EMResult:
    net: BayesNet; log_likelihood: float; iterations: int; seed: int


def run_em(net, data, elim_order, hidden_vars, rest_seed, max_iter, tol, alpha, ll_sample, log) -> EMResult:
    rng = random.Random(rest_seed)
    obs = set(data.columns)
    involved = {n for n in net.nodes if n in hidden_vars or any(p in hidden_vars for p in net.parents[n])}
    fixed = {n for n in net.nodes if n not in involved and n in obs and set(net.parents[n]) <= obs}

    for n in fixed:    net.probabilities[n] = mle_cpt(net, n, data, alpha)
    for n in involved: randomize_cpt(net, n, rng)

    ve = VariableElimination(net)
    grouped = data.groupby(list(data.columns), dropna=False).size().reset_index(name="weight")
    total_w = float(grouped["weight"].sum())
    hidden_list = sorted(hidden_vars)

    if log:
        log.write(f"Hidden: {hidden_list}\nInvolved: {sorted(involved)}\nFixed: {sorted(fixed)}\n\n")

    # Precompute per-node metadata
    meta: dict[str, dict] = {}
    for node in involved:
        t = net.probabilities[node].copy()
        kc = [c for c in t.columns if c != "prob"]
        meta[node] = dict(
            template=t, key_cols=kc,
            obs_kc=[c for c in kc if c not in hidden_vars],
            hid_kc=[c for c in kc if c in hidden_vars],
            key_tuples=[tuple(r) for r in t[kc].itertuples(index=False, name=None)],
            parents=net.parents[node],
            parent_idx=[kc.index(p) for p in net.parents[node]],
        )
        meta[node]["valid"] = set(meta[node]["key_tuples"])

    prev_ll = None
    for it in range(1, max_iter + 1):
        ec = {n: {k: 0.0 for k in meta[n]["key_tuples"]} for n in involved}
        ll = 0.0
        t0 = time.perf_counter()
        n_ll = len(grouped) if ll_sample is None else min(len(grouped), ll_sample)

        for ri, row in enumerate(grouped.itertuples(index=False, name=None), 1):
            weight = float(row[-1])
            evidence = {c: v for c, v in zip(data.columns, row[:-1]) if v == v and v is not None}
            res = ve.run_joint(hidden_list, evidence, elim_order, return_evidence_prob=True)
            post, p_e = res if isinstance(res, tuple) else (res, 0.0)
            if post.empty: continue
            if ri <= n_ll: ll += weight * math.log(max(float(p_e), 1e-300))

            pcols = list(post.columns)
            prows = list(post.itertuples(index=False, name=None))
            for node in involved:
                m = meta[node]
                if any(c not in evidence for c in m["obs_kc"]): continue
                obs_part = {c: evidence[c] for c in m["obs_kc"]}
                if not m["hid_kc"]:
                    key = tuple(obs_part[c] for c in m["key_cols"])
                    if key in m["valid"]: ec[node][key] += weight
                    continue
                hpos = [pcols.index(c) for c in m["hid_kc"]]
                pp = pcols.index("prob")
                for pr in prows:
                    hp = {c: pr[i] for c, i in zip(m["hid_kc"], hpos)}
                    key = tuple(obs_part.get(c, hp.get(c)) for c in m["key_cols"])
                    if key in m["valid"]: ec[node][key] += float(pr[pp]) * weight

            if log and ri % 25 == 0:
                rps = ri / max(time.perf_counter() - t0, 1e-9)
                log.write(f"  E {ri}/{len(grouped)} ({rps:.1f}/s, ETA {(len(grouped)-ri)/max(rps,1e-9)/60:.1f}m)\n")
                log.flush()

        # M-step
        max_delta = 0.0
        for node in involved:
            m = meta[node]
            t = m["template"].copy()
            old = t["prob"].astype(float).tolist()
            new: list[float] = []
            if not m["parents"]:
                denom = sum(ec[node].values()) + alpha * len(net.values[node])
                new = [(ec[node][k] + alpha) / denom for k in m["key_tuples"]]
            else:
                pd_ = {}
                for k, c in ec[node].items():
                    pk = tuple(k[i] for i in m["parent_idx"])
                    pd_[pk] = pd_.get(pk, 0.0) + c
                for k in m["key_tuples"]:
                    pk = tuple(k[i] for i in m["parent_idx"])
                    new.append((ec[node][k] + alpha) / (pd_[pk] + alpha * len(net.values[node])))
            max_delta = max(max_delta, max(abs(o - n) for o, n in zip(old, new)))
            t["prob"] = new; net.probabilities[node] = t

        delta_ll = None if prev_ll is None else ll - prev_ll
        if log:
            log.write(f"Iter {it:03d}: LL={ll:.4f}" + (f" Δ={delta_ll:.4f}" if delta_ll else "") + f" |Δθ|={max_delta:.2g}\n")
            log.flush()
        if prev_ll is not None and abs(ll - prev_ll) < tol * (1 + abs(prev_ll)):
            if log: log.write("Converged\n")
            return EMResult(net, ll, it, rest_seed)
        prev_ll = ll

    if log: log.write("Stopped: max_iter\n")
    return EMResult(net, prev_ll or float("-inf"), max_iter, rest_seed)


def _best_of_restarts(bif, data, hidden_vars, restarts, max_iter, tol, alpha, ll_sample, log) -> EMResult:
    elim = min_fill_order(BayesNet(bif))
    best: EMResult | None = None
    for r in range(restarts):
        seed = 1337 + r * 17
        if log: log.write(f"=== Restart {r+1}/{restarts} (seed={seed}) ===\n")
        res = run_em(BayesNet(bif), data, elim, hidden_vars, seed, max_iter, tol, alpha, ll_sample, log)
        if log: log.write(f"Done: iters={res.iterations} LL={res.log_likelihood:.4f}\n\n")
        if best is None or res.log_likelihood > best.log_likelihood: best = res
    return best  # type: ignore[return-value]  


def run_part3_em(dataset_path, data_file_path, learned_bif_path,
                 sample_rows=200, restarts=3, max_iter=10) -> tuple[str, str]:
    import os
    sample_rows = int(os.environ.get("EM_SAMPLE_ROWS", sample_rows))
    restarts    = int(os.environ.get("EM_RESTARTS",    restarts))
    max_iter    = int(os.environ.get("EM_MAX_ITER",    max_iter))

    original = BayesNet(str(dataset_path))
    data = align_data_columns(load_dataset(str(data_file_path)), original)
    if sample_rows > 0: data = data.head(sample_rows)
    hidden_vars = detect_hidden_vars(original, data)

    log = io.StringIO()
    log.write(f"BIF: {dataset_path.name}\nData: {data_file_path.name} rows={len(data)}\n"
              f"Restarts={restarts} max_iter={max_iter}\n\n")
    best = _best_of_restarts(str(dataset_path), data, hidden_vars, restarts, max_iter, 1e-5, 1e-6, None, log)

    log.write(f"\n=== Best: seed={best.seed} iters={best.iterations} LL={best.log_likelihood:.4f} ===\n")
    for node in original.nodes:
        ma, mx = cpt_mean_abs_diff(original.probabilities[node], best.net.probabilities[node])
        log.write(f"{node}: mean={ma:.4g} max={mx:.4g}\n")
    write_bif(best.net, str(learned_bif_path))
    log.write(f"Saved: {learned_bif_path.name}\n")

    return (f"Part 3 EM done: restarts={restarts} max_iter={max_iter} data={data_file_path.name}",
            log.getvalue().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bif",         default="endorisk_new.bif")
    ap.add_argument("--data",        default="simulation_data_hid_names.dat")
    ap.add_argument("--hidden-vars", default="")
    ap.add_argument("--restarts",    type=int,   default=5)
    ap.add_argument("--max-iter",    type=int,   default=50)
    ap.add_argument("--tol",         type=float, default=1e-5)
    ap.add_argument("--alpha",       type=float, default=1e-6)
    ap.add_argument("--ll-sample",   type=int,   default=2000)
    ap.add_argument("--log",         default="em_log.txt")
    ap.add_argument("--export-bif",  default="")
    args = ap.parse_args()

    original = BayesNet(args.bif)
    data = align_data_columns(load_dataset(args.data), original)
    hidden_vars = ({v.strip() for v in args.hidden_vars.split(",") if v.strip()}
                   or detect_hidden_vars(original, data))
    if not hidden_vars:
        raise ValueError("No hidden variables found.")

    ll_sample = None if args.ll_sample == 0 else args.ll_sample
    with open(args.log, "w", encoding="utf-8") as log:
        log.write(f"BIF={args.bif} data={args.data} rows={len(data)}\n"
                  f"restarts={args.restarts} max_iter={args.max_iter} tol={args.tol} alpha={args.alpha}\n\n")
        best = _best_of_restarts(args.bif, data, hidden_vars, args.restarts,
                                 args.max_iter, args.tol, args.alpha, ll_sample, log)
        log.write(f"\n=== Best: seed={best.seed} iters={best.iterations} LL={best.log_likelihood:.4f} ===\n")
        for node in original.nodes:
            ma, mx = cpt_mean_abs_diff(original.probabilities[node], best.net.probabilities[node])
            log.write(f"{node}: mean={ma:.4g} max={mx:.4g}\n")

    print(f"Done. Best seed={best.seed} iters={best.iterations} LL={best.log_likelihood:.4f}")
    if args.export_bif: write_bif(best.net, args.export_bif)


if __name__ == "__main__":
    main()