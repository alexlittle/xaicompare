# expailens/dashboard/app.py

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st


# ---------- Helpers ----------
def find_latest_run(base: Path = Path("runs")) -> Path | None:
    """Return newest subfolder under 'runs/' containing meta.json, else None."""
    if not base.exists():
        return None
    candidates = []
    for p in base.iterdir():
        if p.is_dir() and (p / "meta.json").exists():
            candidates.append((p, p.stat().st_mtime))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def parse_cli_run_arg(default: str | None = None) -> str | None:
    """
    Read --run from sys.argv AFTER the '--' separator that Streamlit uses.
    Example launch:
        streamlit run expailens/dashboard/app.py -- --run runs/2026-02-18_chapters_xgb
    """
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1 :]
    else:
        # Streamlit sometimes injects argv without '--', handle that too
        args = sys.argv[1:]
    # naive parse
    for i, a in enumerate(args):
        if a == "--run" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--run="):
            return a.split("=", 1)[1]
    return default


def get_run_from_query_params() -> str | None:
    """Support ?run=... in the URL."""
    try:
        params = st.experimental_get_query_params()
        run = params.get("run", [None])[0]
        return run
    except Exception:
        return None


def load_run(run_dir: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = Path(run_dir)
    meta = json.loads((p / "meta.json").read_text())
    preds = pd.read_parquet(p / "predictions.parquet")
    global_imp = pd.read_parquet(p / "shap_global.parquet")
    local = pd.read_parquet(p / "shap_local.parquet")
    text = pd.read_parquet(p / "text_index.parquet")
    return meta, preds, global_imp, local, text


# ---------- Main ----------
def main():
    st.set_page_config(layout="wide", page_title="XAI Dashboard")

    # 1) Resolve run_dir in this order:
    #    a) CLI arg --run (recommended)
    #    b) Query param ?run=...
    #    c) Latest detected run under runs/
    #    d) fallback "runs/_latest" (if present)
    cli_run = parse_cli_run_arg()
    url_run = get_run_from_query_params()
    latest = find_latest_run()
    fallback = Path("runs/_latest") if (Path("runs/_latest") / "meta.json").exists() else None

    resolved = cli_run or url_run or (str(latest) if latest else None) or (str(fallback) if fallback else None)

    # 2) Sidebar: allow user to override interactively
    st.sidebar.header("Run selection")
    default_input = resolved or ""
    run_dir_input = st.sidebar.text_input("Run folder", value=default_input, help="Path to a run folder containing meta.json")

    # 3) Show available runs to help users pick
    with st.sidebar.expander("Available runs", expanded=False):
        base = Path("runs")
        if base.exists():
            for p in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
                st.write(str(p))
        else:
            st.write("No 'runs/' directory found.")

    # 4) Validation and friendly error (instead of crashing)
    if not run_dir_input:
        st.warning("No run directory provided. Use --run, ?run=, pick the latest, or type a path in the sidebar.")
        st.stop()

    run_path = Path(run_dir_input)
    meta_json = run_path / "meta.json"

    if not meta_json.exists():
        st.error(f"Run folder not found or invalid: `{run_dir_input}`. Expected file: `{meta_json}`")
        st.info("Tip: Launch with:\n\n"
                "`streamlit run expailens/dashboard/app.py -- --run runs/2026-02-18_chapters_xgb`")
        st.stop()

    # 5) Load artifacts
    meta, preds, global_imp, local, text = load_run(run_dir_input)

    # 6) Render
    st.title("XAI Dashboard")
    st.caption(f"Using run directory: `{run_dir_input}`")
    st.json(meta)

    st.header("Global token importance")
    topN = st.sidebar.slider("Top-N", 10, 200, 20, 5)
    st.dataframe(global_imp.sort_values("mean_abs_importance", ascending=False).head(topN))

    st.header("Local explanation")
    sid = st.sidebar.number_input("Sample ID", 0, max(0, len(preds) - 1), 0)
    txt = text.loc[text["sample_id"] == sid, "text"].iloc[0] if "text" in text.columns and (text["sample_id"] == sid).any() else "(no text)"
    st.write(txt)

    row_local = local[local["sample_id"] == sid].sort_values("abs_value", ascending=False)
    st.bar_chart(row_local.set_index("feature")["shap_value"])


if __name__ == "__main__":
    main()