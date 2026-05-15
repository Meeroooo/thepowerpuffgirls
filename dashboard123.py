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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberEx Recommender - Powerpuff Girls",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    .metric-label { color: #8b9ab5; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #e2e8f0; font-size: 1.3rem; font-weight: 700; margin-top: 0.2rem; }
    .tag-pill {
        display: inline-block;
        background: #1e3a5f;
        color: #7ec8e3;
        border: 1px solid #2563a8;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.7rem;
        margin: 2px 3px 2px 0;
    }
    .tag-pill-tactic {
        background: #1e3d2f;
        color: #6ee7b7;
        border: 1px solid #065f46;
    }
    .tag-pill-threat {
        background: #3d1e1e;
        color: #fca5a5;
        border: 1px solid #991b1b;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #93c5fd;
        border-bottom: 1px solid #2d3250;
        padding-bottom: 0.3rem;
        margin-bottom: 0.6rem;
    }
    .org-card {
        background: #1a1f35;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .feedback-saved {
        background: #052e16;
        border: 1px solid #166534;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #86efac;
        font-size: 0.85rem;
    }
    .chat-message {
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .chat-user {
        background: #1e3a5f;
        color: #7ec8e3;
        text-align: right;
    }
    .chat-bot {
        background: #1e3d2f;
        color: #6ee7b7;
    }
    .chat-error {
        background: #3d1e1e;
        color: #fca5a5;
    }
    div[data-testid="stSelectbox"] label { color: #93c5fd !important; font-weight: 600; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .dev-badge {
        background: #7c3aed;
        color: #e9d5ff;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "ollama_available" not in st.session_state:
    st.session_state.ollama_available = None
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False

# ── Ollama Helper Functions ──────────────────────────────────────────────
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 30

def cache_ollama_explanations(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ex_id = args[0] if args else kwargs.get('ex_id', '')
        org_id = args[1] if len(args) > 1 else kwargs.get('org_id', '')
        cache_key = f"ollama_ex{ex_id}_org{org_id}"
        cache_file = os.path.join(os.path.dirname(__file__), f".cache_{cache_key}.txt")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return f.read()
        
        result = func(*args, **kwargs)
        try:
            with open(cache_file, 'w') as f:
                f.write(result)
        except:
            pass
        return result
    return wrapper

@cache_ollama_explanations
def get_ollama_explanation(ex_id, org_id, org_profile, ex_tags, scores):
    try:
        tags_str = ", ".join(ex_tags) if ex_tags else "N/A"
        scores_str = f"Hybrid: {scores['hybrid']:.3f}, CF: {scores['cf']:.2f}/5"
        
        prompt = f"""You are a cybersecurity training advisor. Explain why this exercise is recommended to the organization in 2-3 sentences.

Organization Profile:
- Industry: {org_profile.get('Industry', 'Unknown')}
- Region: {org_profile.get('Region', 'Unknown')}
- Size: {org_profile.get('Size', 'Unknown')}
- Maturity: {org_profile.get('Maturity', 'Unknown')}/5
- Primary Threats: {org_profile.get('Threats', 'Unknown')}

Exercise:
- ID: {ex_id}
- Tags/Threats: {tags_str}
- Scores: {scores_str}

Generate a concise explanation focusing on why this exercise aligns with the organization's profile."""
        
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return None
    except:
        return None

def get_ollama_chat_response(user_message, org_profile, top_recs_context):
    try:
        prompt = f"""You are a helpful cybersecurity training assistant. Answer the user's question based on the context provided.

Organization Profile:
- Industry: {org_profile.get('Industry', 'Unknown')}
- Region: {org_profile.get('Region', 'Unknown')}
- Size: {org_profile.get('Size', 'Unknown')}
- Maturity: {org_profile.get('Maturity', 'Unknown')}/5
- Primary Threats: {org_profile.get('Threats', 'Unknown')}

Top Recommended Exercises Context:
{top_recs_context}

User Question: {user_message}

Provide a helpful, concise answer (1-2 sentences)."""
        
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return "API error"
    except:
        return "Ollama connection error. Ensure Ollama is running."

def check_ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    recs = pd.read_csv(os.path.join(base, "phase2_top10_recommendations.csv"))
    exs = pd.read_csv(os.path.join(base, "exercises_full.csv"))
    orgs_df = pd.read_csv(os.path.join(base, "orgs_full.csv"))
    merged = recs.merge(exs, on="EXID", how="left")
    return recs, exs, orgs_df, merged

recs, exs, orgs_df, merged = load_data()

# ── Load Phase 3 data ──────────────────────────────────────────────────────
@st.cache_data
def load_phase3():
    base = os.path.dirname(__file__)
    gaps = pd.read_csv(os.path.join(base, "phase3_technique_gaps.csv"))
    comparison = pd.read_csv(os.path.join(base, "phase3_model_comparison.csv"))
    comparison['f2'] = (5 * comparison['precision'] * comparison['recall']) / (4 * comparison['precision'] + comparison['recall'] + 1e-8)
    return gaps, comparison

try:
    gaps_df, comparison_df = load_phase3()
    PHASE3_AVAILABLE = True
except FileNotFoundError:
    PHASE3_AVAILABLE = False
    gaps_df = pd.DataFrame()
    comparison_df = pd.DataFrame()

# ── Load training data for dev mode ────────────────────────────────────────
@st.cache_data
def load_training_data():
    base = os.path.dirname(__file__)
    
    ae_history = None
    latent_vectors = None
    
    if os.path.exists(os.path.join(base, "autoencoder_history.json")):
        with open(os.path.join(base, "autoencoder_history.json"), 'r') as f:
            ae_history = json.load(f)
    
    if os.path.exists(os.path.join(base, "latent_vectors.npy")):
        latent_vectors = np.load(os.path.join(base, "latent_vectors.npy"))
    
    return ae_history, latent_vectors

ae_history, latent_vectors = load_training_data()

# ── Helper functions ──────────────────────────────────────────────────────
def parse_tags(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [t.strip() for t in str(val).split(";") if t.strip()]

def tag_pills(tags, css_class="tag-pill"):
    return " ".join([f'<span class="{css_class}">{t}</span>' for t in tags])

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    col_title, col_dev = st.columns([3, 1])
    with col_title:
        st.markdown("## CyberEx Recommender")
        st.markdown("*Powerpuff Girls - COS70008*")
    with col_dev:
        if st.button("🔧 Dev", key="dev_toggle"):
            st.session_state.dev_mode = not st.session_state.dev_mode
    
    st.markdown("---")

    org_ids = sorted(merged["ORGID"].unique().tolist())
    selected_org = st.selectbox("Select Organisation", org_ids, format_func=lambda x: f"Org {x:03d}")

    st.markdown("---")
    org_info = orgs_df[orgs_df["ORGID"] == selected_org].iloc[0] if selected_org in orgs_df["ORGID"].values else None

    if org_info is not None:
        st.markdown("**Organisation Profile**")
        st.markdown(f"""
        <div class="org-card">
            <div class="metric-label">Industry</div>
            <div style="color:#e2e8f0;font-size:0.85rem">{org_info.get('Industry','—')}</div>
            <div class="metric-label">Region</div>
            <div style="color:#e2e8f0;font-size:0.85rem">{org_info.get('Region','—')}</div>
            <div class="metric-label">Size</div>
            <div style="color:#e2e8f0;font-size:0.85rem">{org_info.get('Size','—')}</div>
            <div class="metric-label">Maturity Level</div>
            <div style="color:#e2e8f0;font-size:0.85rem">{org_info.get('Maturity','—')} / 5</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.dev_mode:
        st.markdown("---")
        st.markdown("**Developer Mode Active**")
        if PHASE3_AVAILABLE:
            st.success("Phase 3 models loaded")
        if ae_history:
            st.success("Training history available")
        st.caption("Model analysis available in main panel")
    
    # ── Tunable Parameters ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**⚙️ Parameters**")
    
    top_k_value = st.slider("Top-K Recommendations", min_value=3, max_value=15, value=10, step=1,
                            help="Number of exercises to recommend")
    
    model_choice = st.selectbox("Model", ["Hybrid (Phase 2)", "Content-only (Phase 1)", "Ensemble (Phase 3)"],
                                help="Select which model drives recommendations")
    
    blend_weight = st.slider("Blending Weight (α)", min_value=0.0, max_value=1.0, value=0.9, step=0.05,
                             help="α=1.0 means pure CF, α=0.0 means pure content")

# ── Chat Toggle Button ────────────────────────────────────────────────────
chat_col1, chat_col2, chat_col3 = st.columns([5, 1, 1])
with chat_col2:
    if st.button("💬 Chat", key="chat_toggle"):
        st.session_state.show_chat = not st.session_state.show_chat

# ── Chat Panel ────────────────────────────────────────────────────────────
if st.session_state.show_chat:
    with st.container():
        st.markdown("---")
        st.markdown("### Cybersecurity Assistant")
        
        if st.session_state.ollama_available is None:
            st.session_state.ollama_available = check_ollama_available()
        
        if not st.session_state.ollama_available:
            st.warning("Ollama not available. Run 'ollama serve' first.")
        else:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f"""<div class="chat-message chat-user">👤 {msg["content"]}</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="chat-message chat-bot">🤖 {msg["content"]}</div>""", unsafe_allow_html=True)
        
        user_input = st.text_input("Ask about cybersecurity exercises...", key="chat_input", 
                                   placeholder="Example: Which exercise covers ransomware?")
        
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send_button = st.button("Send", key="send_msg")
        with col_clear:
            clear_button = st.button("Clear Chat", key="clear_chat")
        
        if clear_button:
            st.session_state.chat_messages = []
            st.session_state.last_query = None
            st.rerun()
        
        if send_button and user_input and user_input.strip():
            if st.session_state.last_query != user_input:
                st.session_state.last_query = user_input
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                if st.session_state.ollama_available:
                    org_recs_context = merged[merged["ORGID"] == selected_org].sort_values("Rank").reset_index(drop=True)
                    context_lines = []
                    for idx, row in org_recs_context.head(10).iterrows():
                        threat = str(row.get("ExThreat", ""))[:40]
                        context_lines.append(f"- Ex {int(row['EXID']):02d}: {threat} (Score: {row['Hybrid_Score']:.3f})")
                    context = "\n".join(context_lines)
                    
                    bot_response = get_ollama_chat_response(user_input, org_info, context)
                    st.session_state.chat_messages.append({"role": "bot", "content": bot_response})
                else:
                    st.session_state.chat_messages.append({"role": "bot", "content": "Ollama not connected. Please start Ollama service."})
                
                st.rerun()

# ── Main Header ──────────────────────────────────────────────────────────
st.markdown(f"## Exercise Recommendations - Org {selected_org:03d}")

if org_info is not None:
    threats = parse_tags(org_info.get("Threats", ""))
    st.markdown(
        f"**{org_info.get('Industry','—')}** · {org_info.get('Region','—')} · {org_info.get('Size','—')} · "
        + tag_pills(threats[:3], "tag-pill-threat"),
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Determine which view to show ──────────────────────────────────────────
if st.session_state.dev_mode:
    # ========================================================================
    # DEV VIEW - Model Analysis for Professors
    # ========================================================================
    st.markdown("## 🔧 Developer View: Model Analysis")
    st.caption("Technical analysis of the recommendation models - for evaluation purposes")
    st.markdown("---")
    
    dev_tab1, dev_tab2, dev_tab3 = st.tabs(["Model Performance", "Latent Space Analysis", "Technique Gap Analysis"])
    
    with dev_tab1:
        st.markdown("### Phase 3 Model Metrics")
        if PHASE3_AVAILABLE:
            display_df = comparison_df.copy()
            numeric_display = [c for c in display_df.columns if c != 'model']
            for c in numeric_display:
                display_df[c] = pd.to_numeric(display_df[c], errors='coerce').round(4)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Bar chart for flat metrics
            flat_cols = ['precision', 'recall', 'f1']
            flat_available = [c for c in flat_cols if c in comparison_df.columns]
            if flat_available:
                plot_df = comparison_df.melt(id_vars=['model'], 
                                              value_vars=flat_available,
                                              var_name='metric', value_name='score')
                fig = px.bar(plot_df, x='metric', y='score', color='model', 
                             barmode='group', text_auto='.3f',
                             title="Phase 3 Model Performance Comparison")
                fig.update_layout(yaxis=dict(range=[0, 1.1]), plot_bgcolor='#0f1117',
                                 paper_bgcolor='#0f1117', font=dict(color='#cbd5e1'))
                st.plotly_chart(fig, use_container_width=True)
            
            # Ranking metrics chart
            ranking_cols = [c for c in comparison_df.columns if c.startswith(('P@', 'R@', 'NDCG@', 'TopK'))]
            if ranking_cols:
                st.markdown("### Ranking Metrics (Precision@K, Recall@K, NDCG@K)")
                rank_plot = comparison_df.melt(id_vars=['model'],
                                               value_vars=ranking_cols,
                                               var_name='metric', value_name='score')
                fig_rank = px.bar(rank_plot, x='metric', y='score', color='model',
                                  barmode='group', text_auto='.3f',
                                  title="Ranking Metrics by Model")
                fig_rank.update_layout(plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
                                       font=dict(color='#cbd5e1', size=10))
                st.plotly_chart(fig_rank, use_container_width=True)
        else:
            st.warning("Run train_phase3.py to generate model comparison data.")
        
        st.markdown("### Autoencoder Training Loss")
        if ae_history is not None:
            loss_df = pd.DataFrame({
                'Epoch': list(range(1, len(ae_history['loss']) + 1)),
                'Training Loss': ae_history['loss'],
                'Validation Loss': ae_history['val_loss']
            })
            fig_loss = px.line(loss_df, x='Epoch', y=['Training Loss', 'Validation Loss'],
                              title="Autoencoder Training Progress")
            fig_loss.update_layout(plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
                                  font=dict(color='#cbd5e1'))
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Training history not found. Run train_phase3.py first.")
    
    with dev_tab2:
        st.markdown("### Latent Space Visualization")
        st.caption("Organizations mapped to 64-dim latent space, reduced to 2D via t-SNE")
        
        if latent_vectors is not None:
            with st.spinner("Computing t-SNE..."):
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                latent_2d = tsne.fit_transform(latent_vectors)
                
                # Get maturity levels for coloring
                maturity_levels = []
                for org_id in org_ids:
                    org_row = orgs_df[orgs_df['ORGID'] == org_id]
                    maturity = org_row['Maturity'].values[0] if not org_row.empty else 3
                    maturity_levels.append(maturity)
                
                latent_df = pd.DataFrame({
                    'Component 1': latent_2d[:, 0],
                    'Component 2': latent_2d[:, 1],
                    'Maturity Level': maturity_levels,
                    'Org ID': org_ids
                })
                
                fig_latent = px.scatter(latent_df, x='Component 1', y='Component 2',
                                       color='Maturity Level', 
                                       color_continuous_scale='Viridis',
                                       title="Organizations in Autoencoder Latent Space",
                                       hover_data=['Org ID'])
                fig_latent.update_layout(plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
                                        font=dict(color='#cbd5e1'))
                st.plotly_chart(fig_latent, use_container_width=True)
        else:
            st.info("Latent vectors not found. Run train_phase3.py first.")
    
    with dev_tab3:
        st.markdown("### Predicted Technique Gaps (Phase 3)")
        if PHASE3_AVAILABLE and not gaps_df.empty:
            org_gaps = gaps_df[gaps_df['ORGID'] == selected_org]
            if not org_gaps.empty:
                st.dataframe(org_gaps[['Rank', 'TechniqueID', 'AE_Score', 'MLP_Score', 'Ensemble_Score']].head(15),
                           use_container_width=True, hide_index=True)
                
                # Score distribution
                fig = px.bar(org_gaps.head(10), x='Rank', y='Ensemble_Score',
                            title="Top 10 Gap Scores for Selected Organization",
                            labels={'Ensemble_Score': 'Prediction Score'})
                fig.update_layout(plot_bgcolor='#0f1117', paper_bgcolor='#0f1117')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run train_phase3.py to generate technique gap predictions.")

else:
    # ========================================================================
    # USER VIEW - Clean Exercise Recommender Dashboard
    # ========================================================================
    
    org_recs = merged[merged["ORGID"] == selected_org].sort_values("Rank").reset_index(drop=True)
    org_recs = org_recs.head(top_k_value)  # Apply Top-K slider
    
    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    
    avg_hybrid = org_recs["Hybrid_Score"].mean()
    avg_cf = org_recs["CF_Predicted_Rating"].mean()
    tactics_covered = set()
    for _, row in org_recs.iterrows():
        tactics_covered.update(parse_tags(row.get("ExTactics", "")))
    
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Recommendations</div>
            <div class="metric-value">{len(org_recs)}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Hybrid Score</div>
            <div class="metric-value">{avg_hybrid:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg CF Rating</div>
            <div class="metric-value">{avg_cf:.2f} / 5</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ATT&CK Tactics Covered</div>
            <div class="metric-value">{len(tactics_covered)}</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("")
    
    # Two column layout: Exercises + Tactic Coverage
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown('<div class="section-header">Top 10 Recommended Exercises</div>', unsafe_allow_html=True)
        
        for _, row in org_recs.iterrows():
            threat_tags = parse_tags(row.get("ExThreat", ""))
            tactic_tags = parse_tags(row.get("ExTactics", ""))
            hybrid = row["Hybrid_Score"]
            cf = row["CF_Predicted_Rating"]
            rank = int(row["Rank"])
            
            with st.expander(f"#{rank}  Ex {int(row['EXID']):02d}  ·  {str(row.get('ExThreat',''))[:45]}  ·  Score: {hybrid:.3f}"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("**Threat Type**")
                    st.markdown(tag_pills(threat_tags, "tag-pill-threat"), unsafe_allow_html=True)
                    st.markdown("**ATT&CK Tactics**")
                    st.markdown(tag_pills(tactic_tags[:5], "tag-pill-tactic"), unsafe_allow_html=True)
                    
                    groups = parse_tags(row.get("ExGroups", ""))
                    if groups:
                        st.markdown("**Adversary Groups**")
                        st.markdown(tag_pills(groups[:4]), unsafe_allow_html=True)
                
                with c2:
                    st.markdown("**Scores**")
                    st.progress(min(hybrid, 1.0), text=f"Hybrid: {hybrid:.3f}")
                    st.progress(min(cf / 5, 1.0), text=f"CF Rating: {cf:.2f}/5")
                    content = row["Content_Score"]
                    st.progress(min(content, 1.0), text=f"Content: {content:.3f}")
                    
                    complexity = row.get("ExComplexity", "—")
                    length = row.get("ExLength", "—")
                    st.markdown(f"**Complexity:** {complexity}/5 · **Length:** {length} min", unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="section-header">ATT&CK Tactic Coverage</div>', unsafe_allow_html=True)
        
        tactic_counts = {}
        for _, row in org_recs.iterrows():
            for t in parse_tags(row.get("ExTactics", "")):
                tactic_counts[t] = tactic_counts.get(t, 0) + 1
        
        if tactic_counts:
            tactic_df = pd.DataFrame(
                sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True),
                columns=["Tactic", "Count"]
            )
            fig_tactic = px.bar(
                tactic_df, x="Count", y="Tactic", orientation="h",
                color="Count",
                color_continuous_scale=["#1e3a5f", "#3b82f6", "#93c5fd"],
                height=350
            )
            fig_tactic.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font=dict(color="#cbd5e1", size=11),
                coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=10, b=0),
                xaxis=dict(gridcolor="#1e2130"),
                yaxis=dict(tickfont=dict(color="#cbd5e1"))
            )
            st.plotly_chart(fig_tactic, use_container_width=True)
    
    st.markdown("---")
    
    # Why This Exercise Section
    st.markdown('<div class="section-header">Why Was This Recommended?</div>', unsafe_allow_html=True)
    
    ex_options = [f"#{int(r['Rank'])}  Ex {int(r['EXID']):02d}  — {str(r.get('ExThreat',''))[:50]}" for _, r in org_recs.iterrows()]
    if ex_options:
        selected_ex_label = st.selectbox("Select an exercise to learn more", ex_options)
        selected_rank = int(selected_ex_label.split("#")[1].split(" ")[0])
        selected_row = org_recs[org_recs["Rank"] == selected_rank].iloc[0]
        
        explanation_tab1, explanation_tab2 = st.tabs(["Why this exercise?", "AI Explanation"])
        
        with explanation_tab1:
            threat_tags = parse_tags(selected_row.get("ExThreat", ""))
            org_threat = str(org_info.get("Threats", "")) if org_info is not None else ""
            shared_threats = [t for t in threat_tags if t.lower() in org_threat.lower()]
            
            if avg_cf >= 3.0:
                cf_sentence = f"Organisations similar to yours rated this exercise {selected_row['CF_Predicted_Rating']:.2f}/5, indicating high relevance."
            else:
                cf_sentence = f"This exercise has a collaborative filtering score of {selected_row['CF_Predicted_Rating']:.2f}/5 from similar organisations."
            
            if selected_row['Content_Score'] >= 0.15:
                content_sentence = f"It shares content similarity ({selected_row['Content_Score']:.3f}) with exercises your organisation has already completed."
            else:
                content_sentence = f"The recommendation is driven by collaborative filtering (content similarity: {selected_row['Content_Score']:.3f})."
            
            if shared_threats:
                threat_sentence = f"This exercise directly addresses {', '.join(shared_threats)}, matching your threat profile."
            elif threat_tags:
                threat_sentence = f"It covers {threat_tags[0]} scenarios to broaden your training coverage."
            else:
                threat_sentence = ""
            
            st.markdown(f"""
            <div class="why-box">
                <strong>Why Exercise {int(selected_row['EXID'])} was recommended:</strong><br><br>
                {cf_sentence}<br><br>
                {content_sentence}<br><br>
                {threat_sentence}
            </div>
            """, unsafe_allow_html=True)
            
            # Show technique IDs
            technique_tags = parse_tags(selected_row.get("ExTechniqueIDs", ""))
            if technique_tags:
                st.markdown("**ATT&CK Technique IDs covered:**")
                st.markdown(tag_pills(technique_tags[:8]), unsafe_allow_html=True)
        
        with explanation_tab2:
            if st.session_state.ollama_available is None:
                st.session_state.ollama_available = check_ollama_available()
            
            if not st.session_state.ollama_available:
                st.warning("AI Explanation requires Ollama. Run 'ollama serve' to enable.")
            else:
                if st.button("Generate AI Explanation", key=f"ai_explain_{int(selected_row['EXID'])}"):
                    with st.spinner("Generating explanation..."):
                        ai_exp = get_ollama_explanation(
                            int(selected_row['EXID']),
                            selected_org,
                            {
                                "Industry": org_info.get("Industry", "—") if org_info else "—",
                                "Region": org_info.get("Region", "—") if org_info else "—",
                                "Size": org_info.get("Size", "—") if org_info else "—",
                                "Maturity": org_info.get("Maturity", "—") if org_info else "—",
                                "Threats": org_info.get("Threats", "—") if org_info else "—"
                            },
                            threat_tags,
                            {"hybrid": selected_row['Hybrid_Score'], "cf": selected_row['CF_Predicted_Rating']}
                        )
                        if ai_exp:
                            st.markdown(f"""
                            <div class="why-box-ollama">
                                <strong>AI Analysis:</strong><br><br>
                                {ai_exp}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to generate explanation")
    
    st.markdown("---")
    
    # Feedback Section
    st.markdown('<div class="section-header">Feedback</div>', unsafe_allow_html=True)
    
    fb1, fb2 = st.columns([1, 2])
    
    with fb1:
        fb_ex_label = st.selectbox("Exercise to rate", ex_options, key="fb_ex")
        fb_rank = int(fb_ex_label.split("#")[1].split(" ")[0])
        fb_row = org_recs[org_recs["Rank"] == fb_rank].iloc[0]
        fb_rating = st.slider("How useful was this recommendation?", 1, 5, 3, format="%d stars", key="fb_rating")
    
    with fb2:
        fb_comment = st.text_area("Additional comments (optional)", height=80)
        submit_fb = st.button("Submit Feedback", type="primary")
    
    if submit_fb:
        feedback_path = os.path.join(os.path.dirname(__file__), "feedback.csv")
        new_row = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ORGID": selected_org,
            "EXID": int(fb_row["EXID"]),
            "rank": fb_rank,
            "rating": fb_rating,
            "hybrid_score": float(fb_row["Hybrid_Score"]),
            "comment": fb_comment.strip()
        }])
        if os.path.exists(feedback_path):
            existing = pd.read_csv(feedback_path)
            updated = pd.concat([existing, new_row], ignore_index=True)
        else:
            updated = new_row
        updated.to_csv(feedback_path, index=False)
        st.markdown(f"""
        <div class="feedback-saved">
            Feedback saved — Org {selected_org}, Exercise {int(fb_row['EXID'])}, {fb_rating} stars. Thank you!
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<span style='color:#4b5563;font-size:0.7rem'>Powerpuff Girls - COS70008 - Ameera Shahid Khan - 106197762</span>", unsafe_allow_html=True)