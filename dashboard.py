import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import requests
import json
from functools import wraps
import numpy as np
from sklearn.manifold import TSNE
import base64

# ── Page config ─────────────────────────────────────────────────────────────[...]
st.set_page_config(
    page_title="Powerpuff Girls | Cybersecurity Exercise Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Logo: base64-encoded so it renders in HTML regardless of working dir ───────
def get_logo_b64():
    base = os.path.dirname(os.path.abspath(__file__))
    for name in ["logo.png", "logo.jpg", "logo.jpeg"]:
        path = os.path.join(base, name)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                ext = name.split(".")[-1]
                return f"data:image/{ext};base64,{data}"
            except Exception:
                pass
    return None

LOGO_B64 = get_logo_b64()

def logo_img(width=48):
    if LOGO_B64:
        return (f'<img src="{LOGO_B64}" width="{width}" '
                f'style="vertical-align:middle;border-radius:6px;">')
    return '<span style="font-size:1.6rem">🛡️</span>'

# ── CSS ──────────────────────────────────────────────────────────────–[...]
st.markdown("""
<style>
    .main { background-color: #000000; }
    .block-container {
        padding-top: 1.8rem; padding-bottom: 3rem;
        padding-left: 2.5rem; padding-right: 2.5rem;
        max-width: 1400px;
        background-color: #000000;
    }
    .stApp { background-color: #000000; }
    #root > div:first-child { background-color: #000000; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #1e2740;
    }
    [data-testid="stSidebar"] .block-container {
        padding-left: 1.2rem; padding-right: 1.2rem;
    }

    .metric-card {
        background: #0d1117; border: 1px solid #1e2740;
        border-radius: 14px; padding: 1.2rem 1rem;
        text-align: center; margin-bottom: 0.2rem;
    }
    .metric-label {
        color: #6b7fa3; font-size: 0.68rem;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;
    }
    .metric-value { color: #e2e8f0; font-size: 1.75rem; font-weight: 700; line-height: 1.1; }

    .tag-pill {
        display: inline-block; background: #1e3a5f; color: #7ec8e3;
        border: 1px solid #2563a8; border-radius: 20px;
        padding: 3px 11px; font-size: 0.69rem; margin: 2px 3px 2px 0; font-weight: 500;
    }
    .tag-pill-tactic { background: #163324; color: #6ee7b7; border: 1px solid #065f46; }
    .tag-pill-threat { background: #2d1515; color: #fca5a5; border: 1px solid #7f1d1d; }

    .section-header {
        font-size: 1.05rem; font-weight: 700; color: #93c5fd;
        border-bottom: 1px solid #1e2740; padding-bottom: 0.5rem; margin-bottom: 1.2rem;
    }

    .org-card { background: #0d1117; border: 1px solid #1e2740; border-radius: 10px; padding: 0.9rem 1rem; }
    .org-card-row { margin-bottom: 0.55rem; }
    .org-card-label { color: #94a3b8; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .org-card-value { color: #f1f5f9; font-size: 0.88rem; font-weight: 500; }

    .why-box {
        background: #111827; border-left: 3px solid #3b82f6;
        border-radius: 0 10px 10px 0; padding: 1rem 1.2rem;
        color: #cbd5e1; font-size: 0.88rem; line-height: 1.7; margin: 0.6rem 0 1rem 0;
    }
    .why-box-ollama {
        background: #0f1f1a; border-left: 3px solid #10b981;
        border-radius: 0 10px 10px 0; padding: 1rem 1.2rem;
        color: #cbd5e1; font-size: 0.88rem; line-height: 1.7; margin: 0.6rem 0 1rem 0;
    }

    /* ── Legibility fixes ── */
    .streamlit-expanderHeader { background-color: #0d1117 !important; border-radius: 8px !important; }
    div[data-testid="stSelectbox"] label { color: #93c5fd !important; font-weight: 600; font-size: 0.8rem; }
    div[data-testid="stSlider"] label { color: #93c5fd !important; font-weight: 600; font-size: 0.8rem; }
    h1, h2, h3, h4 { color: #f1f5f9 !important; }
    p, span, div { color: inherit; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 6px 16px; color: #cbd5e1; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #93c5fd !important; }

    /* ── Sidebar chat panel ── */
    .chat-panel-header {
        font-size: 0.78rem; font-weight: 700; color: #93c5fd;
        text-transform: uppercase; letter-spacing: 0.07em;
        padding-bottom: 0.5rem; margin-bottom: 0.6rem;
        border-bottom: 1px solid #1e2740;
    }
    .cp-messages {
        display: flex; flex-direction: column; gap: 8px;
        max-height: 300px; overflow-y: auto;
        padding: 4px 0 8px 0;
    }
    .cp-user {
        background: #1e3a5f; color: #bfdbfe; padding: 7px 11px;
        border-radius: 12px 12px 4px 12px; font-size: 0.8rem;
        align-self: flex-end; max-width: 90%; line-height: 1.5; word-break: break-word;
    }
    .cp-bot {
        background: #0f2318; color: #a7f3d0; padding: 7px 11px;
        border-radius: 12px 12px 12px 4px; font-size: 0.8rem;
        align-self: flex-start; max-width: 90%; line-height: 1.5; word-break: break-word;
    }
    .cp-hint { color: #4b5e7a; font-size: 0.76rem; text-align: center; padding: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────[...]
for k, v in {"chat_messages": [], "ollama_available": None,
             "last_query": None, "dev_mode": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Ollama ─────────────────────────────────────────────────────────────–[...]
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "llama3.2:3b"
OLLAMA_TIMEOUT = 30

def check_ollama_available():
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False

def cache_ollama_explanations(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ex_id  = args[0] if args else ""
        org_id = args[1] if len(args) > 1 else ""
        cf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f".cache_ex{ex_id}_org{org_id}.txt")
        if os.path.exists(cf):
            return open(cf, encoding="utf-8").read()
        result = func(*args, **kwargs)
        if result:
            try: open(cf, "w", encoding="utf-8").write(result)
            except Exception: pass
        return result
    return wrapper

@cache_ollama_explanations
def get_ollama_explanation(ex_id, org_id, org_profile, ex_tags, scores):
    try:
        prompt = (f"You are a cybersecurity training advisor. In 2-3 sentences explain "
                  f"why this exercise is recommended.\n\n"
                  f"Org: {org_profile.get('Industry','?')} | "
                  f"{org_profile.get('Region','?')} | "
                  f"Maturity {org_profile.get('Maturity','?')}/5 | "
                  f"Threats: {org_profile.get('Threats','?')}\n"
                  f"Exercise {ex_id}: Tags={', '.join(ex_tags) if ex_tags else 'N/A'} | "
                  f"Hybrid={scores['hybrid']:.3f} | CF={scores['cf']:.2f}/5\n\n"
                  f"Be concise and specific.")
        r = requests.post(OLLAMA_API_URL,
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:
        pass
    return None

def get_ollama_chat_response(user_message, org_profile, context):
    try:
        prompt = (f"You are a cybersecurity assistant. Answer briefly (1-2 sentences).\n"
                  f"Org: {org_profile.get('Industry','?')} | "
                  f"Threats: {org_profile.get('Threats','?')}\n"
                  f"Top exercises:\n{context}\n\nQuestion: {user_message}")
        r = requests.post(OLLAMA_API_URL,
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:
        pass
    return "Ollama connection error. Run `ollama serve` first."

# ── Data loading ───────────────────────────────────────────────────────────–[...]
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    recs   = pd.read_csv(os.path.join(base, "phase2_top10_recommendations.csv"))
    exs    = pd.read_csv(os.path.join(base, "exercises_full.csv"))
    orgs   = pd.read_csv(os.path.join(base, "orgs_full.csv"))
    merged = recs.merge(exs, on="EXID", how="left")
    return recs, exs, orgs, merged

@st.cache_data
def load_phase3():
    base = os.path.dirname(os.path.abspath(__file__))
    gaps = pd.read_csv(os.path.join(base, "phase3_technique_gaps.csv"))
    comp = pd.read_csv(os.path.join(base, "phase3_model_comparison.csv"))
    if "f2" not in comp.columns:
        p, r = comp["precision"], comp["recall"]
        comp["f2"] = (5 * p * r) / (4 * p + r + 1e-8)
    return gaps, comp

@st.cache_data
def load_training_data():
    base = os.path.dirname(os.path.abspath(__file__))
    ae, lv = None, None
    hp = os.path.join(base, "autoencoder_history.json")
    lp = os.path.join(base, "latent_vectors.npy")
    if os.path.exists(hp): ae = json.load(open(hp))
    if os.path.exists(lp): lv = np.load(lp)
    return ae, lv

recs, exs, orgs_df, merged = load_data()

try:
    gaps_df, comparison_df = load_phase3()
    PHASE3_AVAILABLE = True
except FileNotFoundError:
    PHASE3_AVAILABLE = False
    gaps_df = comparison_df = pd.DataFrame()

ae_history, latent_vectors = load_training_data()

# ── Helpers ─────────────────────────────────────────────────────────────[...]
def parse_tags(val):
    if pd.isna(val) or str(val).strip() == "": return []
    return [t.strip() for t in str(val).split(";") if t.strip()]

def tag_pills(tags, css="tag-pill"):
    return " ".join(f'<span class="{css}">{t}</span>' for t in tags)

# ── Sidebar ─────────────────────────────────────────────────────────────[...]
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;padding:0.6rem 0 1.1rem 0">'
        f'{logo_img(88)}'
        f'<div><div style="font-weight:700;color:#f1f5f9;font-size:1.1rem;line-height:1.2">CyberEx</div>'
        f'<div style="color:#94a3b8;font-size:0.73rem;margin-top:2px">Powerpuff Girls · COS70008</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr style="border-color:#1e2740;margin:0 0 0.8rem 0">', unsafe_allow_html=True)

    if st.button("Developer Mode", use_container_width=True,
                 type="primary" if st.session_state.dev_mode else "secondary"):
        st.session_state.dev_mode = not st.session_state.dev_mode

    st.markdown('<hr style="border-color:#1e2740;margin:0.8rem 0">', unsafe_allow_html=True)

    org_ids = sorted(merged["ORGID"].unique().tolist())
    selected_org = st.selectbox("Select Organisation", org_ids,
                                format_func=lambda x: f"Org {x:03d}")
    org_info = (orgs_df[orgs_df["ORGID"] == selected_org].iloc[0]
                if selected_org in orgs_df["ORGID"].values else None)

    if org_info is not None:
        def _r(lbl, val, col="#e2e8f0"):
            return (f'<div class="org-card-row">'
                    f'<div class="org-card-label">{lbl}</div>'
                    f'<div class="org-card-value" style="color:{col}">{val}</div></div>')
        thr = parse_tags(org_info.get("Threats",""))
        st.markdown(
            '<div class="org-card">'
            + _r("Industry",  org_info.get("Industry","—"))
            + _r("Region",    org_info.get("Region","—"))
            + _r("Size",      org_info.get("Size","—"))
            + _r("Maturity",  f"{org_info.get('Maturity','—')} / 5")
            + _r("Threat",    thr[0] if thr else "—", "#fca5a5")
            + "</div>",
            unsafe_allow_html=True
        )

    st.markdown('<hr style="border-color:#1e2740;margin:0.8rem 0">', unsafe_allow_html=True)
    st.markdown('<div style="color:#93c5fd;font-size:0.78rem;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.6rem">'
                'Parameters</div>', unsafe_allow_html=True)
    top_k_value  = st.slider("Top-K Recommendations", 3, 15, 10)

    if st.session_state.dev_mode:
        st.markdown('<hr style="border-color:#1e2740;margin:0.8rem 0">', unsafe_allow_html=True)
        st.markdown('<div style="color:#6b7fa3;font-size:0.72rem;margin-bottom:0.4rem">Dev Status</div>',
                    unsafe_allow_html=True)
        def _badge(text, ok):
            bg, col = ("#0d2818","#6ee7b7") if ok else ("#1f1208","#fbbf24")
            st.markdown(f'<div style="background:{bg};border-radius:6px;padding:4px 10px;'
                        f'color:{col};font-size:0.74rem;margin-bottom:4px">{text}</div>',
                        unsafe_allow_html=True)
        _badge("Phase 3 available" if PHASE3_AVAILABLE else "Phase 3 not found", PHASE3_AVAILABLE)
        _badge("Training history loaded" if ae_history else "No training history", ae_history is not None)
        _badge("Latent vectors loaded" if latent_vectors is not None else "No latent vectors",
               latent_vectors is not None)

# ── Sidebar chat panel ────────────────────────────────────────────────────────
chat_input = None
with st.sidebar:
    st.markdown('<hr style="border-color:#1e2740;margin:0.8rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="chat-panel-header">💬 AI Assistant</div>', unsafe_allow_html=True)

    if st.session_state.ollama_available is None:
        st.session_state.ollama_available = check_ollama_available()

    # Message display
    if not st.session_state.ollama_available:
        msgs_html = '<div class="cp-hint">Ollama is not running.<br>Start with <code>ollama serve</code></div>'
    elif not st.session_state.chat_messages:
        msgs_html = '<div class="cp-hint">Ask anything about the recommendations.</div>'
    else:
        msgs_html = "".join(
            f'<div class="{"cp-user" if m["role"]=="user" else "cp-bot"}">{m["content"]}</div>'
            for m in st.session_state.chat_messages[-14:]
        )
    st.markdown(f'<div class="cp-messages">{msgs_html}</div>', unsafe_allow_html=True)

    # Input row
    chat_input = st.chat_input("Ask about exercises…", key="sidebar_chat_input")
    col_a, col_b = st.columns([2, 1])
    with col_b:
        if st.button("Clear", key="chat_clear_btn", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.last_query = None
            st.rerun()

# Handle new sidebar message
if chat_input and chat_input.strip() and chat_input != st.session_state.last_query:
    st.session_state.last_query = chat_input
    st.session_state.chat_messages.append({"role": "user", "content": chat_input.strip()})
    if st.session_state.ollama_available:
        org_recs_ctx = merged[merged["ORGID"]==selected_org].sort_values("Rank").head(10)
        ctx = "\n".join(
            f"- Ex {int(r['EXID']):02d}: {str(r.get('ExThreat',''))[:45]}"
            for _, r in org_recs_ctx.iterrows()
        )
        od = ({k: str(org_info.get(k,"—")) for k in
               ["Industry","Region","Size","Maturity","Threats"]}
              if org_info is not None else {})
        resp = get_ollama_chat_response(chat_input.strip(), od, ctx)
    else:
        resp = "Ollama is not running. Start it with `ollama serve`."
    st.session_state.chat_messages.append({"role": "bot", "content": resp})
    st.rerun()

# ── Main header ───────────────────────────────────────────────────────────––[...]
st.markdown(
    f'<div style="display:flex;align-items:center;gap:18px;margin-bottom:0.4rem">'
    f'{logo_img(88)}'
    f'<div><div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;line-height:1.2">'
    f'Powerpuff Girls</div>'
    f'<div style="color:#94a3b8;font-size:0.88rem;margin-top:3px">'
    f'Cybersecurity Exercise Recommender · COS70008</div></div></div>',
    unsafe_allow_html=True
)
st.markdown('<hr style="border-color:#1e2740;margin:0.6rem 0 1rem 0">', unsafe_allow_html=True)

if org_info is not None:
    threats = parse_tags(org_info.get("Threats",""))
    st.markdown(
        f'<div style="margin-bottom:0.5rem">'
        f'<span style="font-size:1.1rem;font-weight:600;color:#f1f5f9">'
        f'Organisation {selected_org:03d}</span>&nbsp;&nbsp;'
        f'<span style="color:#94a3b8;font-size:0.88rem">'
        f'{org_info.get("Industry","—")} · {org_info.get("Region","—")} · '
        f'{org_info.get("Size","—")}</span></div>',
        unsafe_allow_html=True
    )
    if threats:
        st.markdown(tag_pills(threats[:4],"tag-pill-threat"), unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=False)

# ── Main content ───────────────────────────────────────────────────────────–[...]
if st.session_state.dev_mode:
    # ══ DEVELOPER VIEW ════════════════════════════════════════════════════════
    st.markdown("## Developer View — Model Analysis")
    st.caption("Technical evaluation metrics and Phase 3 analysis.")
    st.markdown("")

    dt1, dt2, dt3 = st.tabs(["Model Performance","Latent Space","Technique Gaps"])

    with dt1:
        if PHASE3_AVAILABLE:
            st.markdown('<div class="section-header">Phase 3 Model Comparison</div>',
                        unsafe_allow_html=True)
            disp = comparison_df.copy()
            for c in [x for x in disp.columns if x != "model"]:
                disp[c] = pd.to_numeric(disp[c], errors="coerce").round(4)
            st.dataframe(disp, use_container_width=True, hide_index=True)
            mcols = [c for c in ["precision","recall","f1","f2"] if c in comparison_df.columns]
            if mcols:
                pdf = comparison_df.melt(id_vars=["model"], value_vars=mcols,
                                         var_name="metric", value_name="score")
                fig = px.bar(pdf, x="metric", y="score", color="model", barmode="group",
                             text_auto=".3f", height=360,
                             color_discrete_sequence=["#3b82f6","#10b981","#f59e0b"])
                fig.update_layout(yaxis=dict(range=[0,1.1]),
                                  plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                  font=dict(color="#cbd5e1"), margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Run `python train_phase3.py` to generate model comparison data.")

        st.markdown("")
        st.markdown('<div class="section-header">Autoencoder Training Loss</div>',
                    unsafe_allow_html=True)
        if ae_history is not None:
            ldf = pd.DataFrame({
                "Epoch":           range(1, len(ae_history["loss"])+1),
                "Training Loss":   ae_history["loss"],
                "Validation Loss": ae_history["val_loss"],
            })
            fl = px.line(ldf, x="Epoch", y=["Training Loss","Validation Loss"], height=320,
                         color_discrete_map={"Training Loss":"#3b82f6","Validation Loss":"#f59e0b"})
            fl.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                             font=dict(color="#cbd5e1"), margin=dict(t=20))
            st.plotly_chart(fl, use_container_width=True)
        else:
            st.info("No training history found. Run `python train_phase3.py` first.")

    with dt2:
        st.markdown('<div class="section-header">Latent Space Visualisation</div>',
                    unsafe_allow_html=True)
        st.caption("64-dim autoencoder representations reduced to 2D via t-SNE. "
                   "Colour = maturity level (1–5).")
        if latent_vectors is not None:
            with st.spinner("Computing t-SNE…"):
                lv2 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(latent_vectors)
                mats = [float(orgs_df[orgs_df["ORGID"]==o]["Maturity"].values[0])
                        if not orgs_df[orgs_df["ORGID"]==o].empty else 3.0
                        for o in org_ids]
                ldf2 = pd.DataFrame({"Component 1":lv2[:,0], "Component 2":lv2[:,1],
                                     "Maturity Level":mats, "Org ID":org_ids})
                fg = px.scatter(ldf2, x="Component 1", y="Component 2",
                                color="Maturity Level", color_continuous_scale="Viridis",
                                hover_data=["Org ID"], height=460)
                fg.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                 font=dict(color="#cbd5e1"), margin=dict(t=20))
                st.plotly_chart(fg, use_container_width=True)
        else:
            st.info("Latent vectors not found. Run `python train_phase3.py` first.")

    with dt3:
        st.markdown('<div class="section-header">Predicted Technique Gaps</div>',
                    unsafe_allow_html=True)
        st.caption(f"Top missing ATT&CK techniques for Org {selected_org:03d}.")
        if PHASE3_AVAILABLE and not gaps_df.empty:
            og = gaps_df[gaps_df["ORGID"]==selected_org]
            if not og.empty:
                scols = [c for c in ["Rank","TechniqueID","AE_Score","MLP_Score","Ensemble_Score"]
                         if c in og.columns]
                st.dataframe(og[scols].head(15), use_container_width=True, hide_index=True)
                st.markdown("")
                t10 = og.head(10)
                fg2 = px.bar(t10, x="Rank", y="Ensemble_Score", color="Ensemble_Score",
                             color_continuous_scale="Blues", height=320,
                             text="TechniqueID" if "TechniqueID" in t10.columns else None,
                             labels={"Ensemble_Score":"Prediction Score","Rank":"Priority Rank"})
                fg2.update_traces(textposition="outside", textfont_size=9)
                fg2.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                  font=dict(color="#cbd5e1"),
                                  coloraxis_showscale=False, margin=dict(t=20))
                st.plotly_chart(fg2, use_container_width=True)
            else:
                st.info(f"No gap predictions for Org {selected_org:03d}.")
        else:
            st.info("Run `python train_phase3.py` to generate technique gap predictions.")

else:
    # ══ ORGANISATION VIEW ═════════════════════════════════════════════════════
    org_recs = (merged[merged["ORGID"]==selected_org]
                .sort_values("Rank").reset_index(drop=True).head(top_k_value))

    avg_hybrid = org_recs["Hybrid_Score"].mean()
    avg_cf     = org_recs["CF_Predicted_Rating"].mean()
    tacs = set()
    for _, row in org_recs.iterrows():
        tacs.update(parse_tags(row.get("ExTactics","")))

    # Metrics row
    for col, lbl, val in zip(
        st.columns(4),
        ["Recommendations","Avg Hybrid Score","Avg CF Rating","Tactics Covered"],
        [str(len(org_recs)), f"{avg_hybrid:.3f}", f"{avg_cf:.2f} / 5", str(len(tacs))]
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                f'<div class="metric-value">{val}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("", unsafe_allow_html=False)

    # Two-column layout
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<div class="section-header">Recommended Exercises</div>',
                    unsafe_allow_html=True)
        for _, row in org_recs.iterrows():
            tt  = parse_tags(row.get("ExThreat",""))
            tac = parse_tags(row.get("ExTactics",""))
            rank = int(row["Rank"]); ex_id = int(row["EXID"])
            hyb  = row["Hybrid_Score"]; cf = row["CF_Predicted_Rating"]
            cont = row["Content_Score"]
            with st.expander(
                f"#{rank}  ·  Ex {ex_id:02d}  ·  {str(row.get('ExThreat',''))[:55]}  ·  {hyb:.3f}"
            ):
                el, er = st.columns([3,2], gap="medium")
                with el:
                    if tt:
                        st.markdown("**Threat Type**")
                        st.markdown(tag_pills(tt,"tag-pill-threat"), unsafe_allow_html=True)
                        st.markdown("")
                    if tac:
                        st.markdown("**ATT&CK Tactics**")
                        st.markdown(tag_pills(tac[:6],"tag-pill-tactic"), unsafe_allow_html=True)
                        st.markdown("")
                    grp = parse_tags(row.get("ExGroups",""))
                    if grp:
                        st.markdown("**Adversary Groups**")
                        st.markdown(tag_pills(grp[:4]), unsafe_allow_html=True)
                with er:
                    st.markdown("**Scores**")
                    st.progress(min(hyb,1.0),        text=f"Hybrid:  {hyb:.3f}")
                    st.progress(min(cf/5,1.0),        text=f"CF:      {cf:.2f}/5")
                    st.progress(min(cont,1.0),        text=f"Content: {cont:.3f}")
                    st.markdown("")
                    st.markdown(
                        f"**Complexity:** {row.get('ExComplexity','—')}/5 &nbsp;·&nbsp; "
                        f"**Duration:** {row.get('ExLength','—')} min",
                        unsafe_allow_html=True
                    )

    with right_col:
        st.markdown('<div class="section-header">ATT&CK Tactic Coverage</div>',
                    unsafe_allow_html=True)
        tc = {}
        for _, row in org_recs.iterrows():
            for t in parse_tags(row.get("ExTactics","")): tc[t] = tc.get(t,0)+1
        if tc:
            tdf = pd.DataFrame(sorted(tc.items(), key=lambda x:x[1], reverse=True)[:10],
                               columns=["Tactic","Count"])
            ft = px.bar(tdf, x="Count", y="Tactic", orientation="h",
                        color="Count", color_continuous_scale="Blues", height=360)
            ft.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                             font=dict(color="#cbd5e1"), showlegend=False,
                             coloraxis_showscale=False,
                             margin=dict(l=0,r=10,t=10,b=20),
                             xaxis=dict(gridcolor="#1e2740"),
                             yaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(ft, use_container_width=True)

        st.markdown("")
        st.markdown('<div class="section-header">Threat Distribution</div>',
                    unsafe_allow_html=True)
        thrc = {}
        for _, row in org_recs.iterrows():
            for t in parse_tags(row.get("ExThreat","")): thrc[t] = thrc.get(t,0)+1
        if thrc:
            fp = px.pie(pd.DataFrame(thrc.items(), columns=["Threat","Count"]),
                        names="Threat", values="Count", hole=0.5,
                        color_discrete_sequence=px.colors.sequential.Blues_r, height=280)
            fp.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                             font=dict(color="#cbd5e1",size=10),
                             margin=dict(l=0,r=0,t=10,b=0),
                             legend=dict(font=dict(size=9)))
            fp.update_traces(textfont_color="#e2e8f0")
            st.plotly_chart(fp, use_container_width=True)

    # Why this exercise
    st.markdown('<hr style="border-color:#1e2740;margin:1.4rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Why Was This Recommended?</div>',
                unsafe_allow_html=True)

    ex_opts = [
        f"#{int(r['Rank'])}  ·  Ex {int(r['EXID']):02d}  ·  {str(r.get('ExThreat',''))[:45]}"
        for _, r in org_recs.iterrows()
    ]
    if ex_opts:
        sel_lbl  = st.selectbox("Select an exercise", ex_opts)
        sel_rank = int(sel_lbl.split("#")[1].split("·")[0].strip())
        sel_row  = org_recs[org_recs["Rank"]==sel_rank].iloc[0]

        t1, t2 = st.tabs(["Template Explanation","AI Explanation"])

        with t1:
            tt2   = parse_tags(sel_row.get("ExThreat",""))
            othr  = str(org_info.get("Threats","")) if org_info is not None else ""
            shard = [t for t in tt2 if t.lower() in othr.lower()]
            cf_t  = (f"Similar organisations rated this {sel_row['CF_Predicted_Rating']:.2f}/5."
                     if sel_row["CF_Predicted_Rating"] >= 3.0 else
                     f"CF predicted rating: {sel_row['CF_Predicted_Rating']:.2f}/5.")
            co_t  = (f"Content similarity: {sel_row['Content_Score']:.3f} — matches your profile."
                     if sel_row["Content_Score"] >= 0.15 else
                     f"Content similarity: {sel_row['Content_Score']:.3f}.")
            thr_t = (f"Directly targets your threat exposure: {', '.join(shard)}."
                     if shard else
                     (f"Covers {tt2[0]} scenarios to broaden coverage." if tt2 else ""))
            st.markdown(
                f'<div class="why-box">'
                f'<strong>Exercise {int(sel_row["EXID"])} was recommended because:</strong>'
                f'<br><br>• {cf_t}<br>• {co_t}'
                + (f'<br>• {thr_t}' if thr_t else "") + "</div>",
                unsafe_allow_html=True
            )
            tids = parse_tags(sel_row.get("ExTechniqueIDs",""))
            if tids:
                st.markdown("**ATT&CK Techniques Covered**")
                st.markdown(tag_pills(tids[:8]), unsafe_allow_html=True)

        with t2:
            if st.session_state.ollama_available is None:
                st.session_state.ollama_available = check_ollama_available()
            if not st.session_state.ollama_available:
                st.info("AI explanations require Ollama. Run `ollama serve` to enable.")
            else:
                if st.button("Generate AI Explanation", key="ai_explain_btn"):
                    with st.spinner("Asking Ollama…"):
                        od2 = ({k: str(org_info.get(k,"—")) for k in
                                ["Industry","Region","Size","Maturity","Threats"]}
                               if org_info is not None else {})
                        ai_r = get_ollama_explanation(
                            int(sel_row["EXID"]), selected_org, od2,
                            parse_tags(sel_row.get("ExThreat","")),
                            {"hybrid":sel_row["Hybrid_Score"],"cf":sel_row["CF_Predicted_Rating"]}
                        )
                    if ai_r:
                        st.markdown(f'<div class="why-box-ollama">{ai_r}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.error("No response. Verify Ollama is running.")

    # Feedback
    st.markdown('<hr style="border-color:#1e2740;margin:1.4rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feedback</div>', unsafe_allow_html=True)
    st.caption("Rate how useful a recommendation was. Saved for future model improvement.")

    fb1, fb2 = st.columns([1,2], gap="large")
    with fb1:
        fbl    = st.selectbox("Exercise to rate", ex_opts, key="fb_sel")
        fb_rnk = int(fbl.split("#")[1].split("·")[0].strip())
        fb_row = org_recs[org_recs["Rank"]==fb_rnk].iloc[0]
        fb_rat = st.slider("How useful?", 1, 5, 3, format="%d stars")
        st.caption({1:"Not useful",2:"Slightly useful",3:"Neutral",4:"Useful",5:"Very useful"
                    }.get(fb_rat,""))
    with fb2:
        fb_cmt = st.text_area("Comments (optional)",
                              placeholder="Any observations or suggestions?",
                              height=90, key="fb_cmt")
        fb_sub = st.button("Submit Feedback", type="primary", key="fb_sub")

    if fb_sub:
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")
        nr = pd.DataFrame([{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "ORGID": selected_org, "EXID": int(fb_row["EXID"]),
                            "rank": fb_rnk, "rating": fb_rat,
                            "hybrid_score": float(fb_row["Hybrid_Score"]),
                            "comment": fb_cmt.strip()}])
        nr = pd.concat([pd.read_csv(fp), nr], ignore_index=True) if os.path.exists(fp) else nr
        nr.to_csv(fp, index=False)
        st.markdown(
            f'<div style="background:#0d2818;border:1px solid #166534;border-radius:8px;'
            f'padding:0.6rem 1rem;color:#86efac;font-size:0.85rem;margin-top:0.4rem">'
            f'Feedback saved — Exercise {int(fb_row["EXID"])}, {fb_rat} star'
            f'{"s" if fb_rat!=1 else ""}.</div>',
            unsafe_allow_html=True
        )

    fp2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")
    if os.path.exists(fp2):
        fd = pd.read_csv(fp2)
        if not fd.empty:
            with st.expander(f"Collected feedback ({len(fd)} entries)"):
                st.dataframe(fd, use_container_width=True, hide_index=True)

# Footer
st.markdown('<hr style="border-color:#1e2740;margin:2rem 0 0.6rem 0">', unsafe_allow_html=True)
st.markdown(
    '<div style="color:#374151;font-size:0.72rem;text-align:center">'
    'Powerpuff Girls · COS70008 · Phase 3 Dashboard</div>',
    unsafe_allow_html=True
)
