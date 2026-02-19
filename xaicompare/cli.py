# xaicompare/cli.py

import argparse
import sys
from pathlib import Path

def main():
    # Try Streamlit CLI import across versions
    try:
        from streamlit.web import cli as stcli  # modern
    except Exception:  # pragma: no cover
        import streamlit.cli as stcli  # older

    # Resolve path to your app.py inside the installed package
    app_py = Path(__file__).resolve().parent / "dashboard" / "app.py"
    if not app_py.exists():
        print(f"Cannot find Streamlit app at: {app_py}", file=sys.stderr)
        sys.exit(1)

    # Simple CLI: optional positional RUN_DIR and pass-through for other args
    parser = argparse.ArgumentParser(
        prog="xaicompare-dash",
        description="Launch XAICompare Dashboard",
        add_help=True,
    )
    parser.add_argument(
        "run",
        nargs="?",
        help="Optional path to a run folder (containing meta.json). If omitted, the app will let you pick.",
    )
    # Parse known args; forward the rest (e.g., --server.port 8502) to Streamlit
    args, remainder = parser.parse_known_args()

    # Build args for streamlit
    st_args = ["run", str(app_py), "--"]
    if args.run:
        st_args += ["--run", args.run]

    # Forward any extra Streamlit flags the user provides
    st_args += remainder

    # Emulate: streamlit run xaicompare/dashboard/app.py -- [--run ...] [extra]
    sys.argv = ["streamlit"] + st_args
    sys.exit(stcli.main())
