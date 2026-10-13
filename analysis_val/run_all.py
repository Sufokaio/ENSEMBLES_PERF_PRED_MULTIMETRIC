# Entry point: load the saved results, select the best single / ensemble, write CSV files, tables and figures.

import argparse
import os
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

import config as cfg

RAW_TABLES = ["baseline", "variants", "ensembles", "diversity", "ensembles_old"]


class Context:
    pass


def _raw_path(name):
    return os.path.join(cfg.RAW_CACHE_DIR, f"{name}.parquet")


def stage_load(force=False):
    print("\n=== load ===")
    import loaders
    os.makedirs(cfg.RAW_CACHE_DIR, exist_ok=True)
    if not force and all(os.path.exists(_raw_path(n)) for n in RAW_TABLES):
        print("  cache present (use --force to reload)")
        return
    loaders.load_baseline().to_parquet(_raw_path("baseline"), index=False)
    variants, ensembles, diversity, old = loaders.load_all(cfg.N_JOBS)
    variants.to_parquet(_raw_path("variants"), index=False)
    ensembles.to_parquet(_raw_path("ensembles"), index=False)
    diversity.to_parquet(_raw_path("diversity"), index=False)
    old.to_parquet(_raw_path("ensembles_old"), index=False)
    print(f"  variants {len(variants):,} rows, ensembles {len(ensembles):,}, "
          f"diversity {len(diversity):,}, old ensembles {len(old):,}")


# Scott-Knott over the 72 top-k competitors (8 learners x 9 k) of each rule and scenario, on MRE
def k_sk_ranks(ensembles, force=False):
    import core
    path = _raw_path("k_sk_perrule")
    if not force and os.path.exists(path):
        return pd.read_parquet(path)
    top = ensembles[ensembles["selector"] == "top"].copy()
    top["competitor"] = top["model"] + "_k" + top["k"].astype(str)
    parts = []
    for rule in cfg.RULES:
        sk = core.run_sk(top[top["rule"] == rule], core.SCEN, "competitor", "test_MRE")
        sk["rule"] = rule
        parts.append(sk)
    sk = pd.concat(parts, ignore_index=True)
    sk["model"] = sk["competitor"].str.rsplit("_k", n=1).str[0]
    sk["k"] = sk["competitor"].str.rsplit("_k", n=1).str[1].astype(int)
    sk = sk[["dataset", "sample_size", "rule", "model", "k", "sk_rank"]]
    sk.to_parquet(path, index=False)
    return sk


def stage_analyze(selection, out_dir, force=False):
    print(f"\n=== analyze (selection = {selection}) ===")
    import core
    import items
    import paper
    from out import Writer

    ctx = Context()
    ctx.selection = selection
    ctx.memo = {}
    ctx.w = Writer(out_dir)
    ctx.baseline = pd.read_parquet(_raw_path("baseline"))
    ctx.variants = core.add_sa_d(pd.read_parquet(_raw_path("variants")), ctx.baseline)
    ctx.ensembles = core.add_sa_d(pd.read_parquet(_raw_path("ensembles")), ctx.baseline)
    ctx.diversity = pd.read_parquet(_raw_path("diversity"))
    ctx.old = pd.read_parquet(_raw_path("ensembles_old"))

    ctx.singles_best = core.select_best_singles(ctx.variants, selection)
    ctx.ens_best = core.select_best_ensembles(ctx.ensembles, selection)
    ctx.ens_best_rule = core.select_best_k_per_rule(ctx.ensembles, selection)
    print(f"  best singles: {len(ctx.singles_best):,} rows, best ensembles: {len(ctx.ens_best):,} rows")

    t0 = time.time()
    ctx.k_sk = k_sk_ranks(ctx.ensembles, force)
    print(f"  k Scott-Knott ranks ready ({time.time() - t0:.0f} s)")

    paper.run(ctx)
    items.run(ctx)

    index = pd.DataFrame({"file": sorted(ctx.w.written)})
    index.to_csv(os.path.join(out_dir, "output_index.csv"), index=False)
    print(f"  {len(index)} files written under {out_dir}")


def main(default_selection="val", out_dir=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Analysis of the validation-ranked ensembles")
    parser.add_argument("--load", action="store_true", help="Read the result files into the cache")
    parser.add_argument("--analyze", action="store_true", help="Write CSV files, tables and figures")
    parser.add_argument("--all", action="store_true", help="Run both stages")
    parser.add_argument("--selection", default=default_selection, choices=cfg.SELECTIONS,
                        help="How the best single and best ensemble are chosen")
    parser.add_argument("--out", default=out_dir or os.path.join(here, "output_artifacts"),
                        help="Output folder")
    parser.add_argument("--force", action="store_true", help="Ignore the cache")
    args = parser.parse_args()

    if not (args.load or args.analyze or args.all):
        parser.print_help()
        sys.exit(0)
    if args.all or args.load:
        stage_load(args.force)
    if args.all or args.analyze:
        stage_analyze(args.selection, args.out, args.force)
    print("\nDone.")


if __name__ == "__main__":
    main()
