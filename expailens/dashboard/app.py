# xai_kit/dashboard/app.py
import streamlit as st, pandas as pd, json, pathlib

def load_run(run_dir):
    p = pathlib.Path(run_dir)
    meta = json.loads((p / "meta.json").read_text())
    preds = pd.read_parquet(p / "predictions.parquet")
    global_imp = pd.read_parquet(p / "shap_global.parquet")
    local = pd.read_parquet(p / "shap_local.parquet")
    text = pd.read_parquet(p / "text_index.parquet")
    return meta, preds, global_imp, local, text

def main():
    st.set_page_config(layout="wide")
    run_dir = st.sidebar.text_input("Run folder", "runs/_latest")
    meta, preds, global_imp, local, text = load_run(run_dir)

    st.title("XAI Dashboard")
    st.subheader(f"Run: {meta.get('run_id')}  |  Method: {meta.get('method')}")
    # Global
    top_n = st.sidebar.slider("Top-N global tokens", 10, 100, 20)
    st.header("Global token importance (mean |SHAP|)")
    st.dataframe(global_imp.head(top_n))
    # Local
    sid = st.sidebar.number_input("Sample ID", 0, len(preds)-1, 0)
    st.header("Local explanation")
    txt = text[text.sample_id==sid].text.iloc[0] if "text" in text else "(no text)"
    st.write(txt)
    local_row = local[local.sample_id==sid].sort_values("abs_value", ascending=False)
    st.bar_chart(local_row.set_index("feature")["shap_value"])

if __name__ == "__main__":
    main()