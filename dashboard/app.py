import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon

# streamlit run app.py



# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
CONTENT = ROOT / "data" / "content"

TOPICS = [
    "clima", "sicurezza urbana", "immigrazione",
    "diritti civili", "istruzione", "salute mentale", "affitti-abitazione",
]

TOPIC_KEYWORDS = {
    "clima":               ["clima", "cambiamento climatico", "ambiente", "green", "ecolog"],
    "sicurezza urbana":    ["sicurezza", "criminalit", "delinquenz"],
    "immigrazione":        ["immigra", "stranieri", "integrazione", "multicultur"],
    "diritti civili":      ["diritti", "inclusione", "lgbtq", "discrimina", "parit"],
    "istruzione":          ["istruzione", "scuola", "universit", "formazione", "studio"],
    "salute mentale":      ["salute mentale", "ansia", "depressione", "benessere psicolog"],
    "affitti-abitazione":  ["affitt", "casa", "abitazione", "alloggio"],
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#E2E8F0"),
)

# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_scored_csvs():
    csvs = {}
    for p in PROCESSED.glob("*_AVG_HUMAN.csv"):
        profile = p.stem.replace("_AVG_HUMAN", "")
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(
            df["folder_id"].str.extract(r"^(\d{4}-\d{2}-\d{2})")[0]
        )
        csvs[profile] = df
    return csvs


@st.cache_data
def load_posts(profile: str) -> dict:
    path = CONTENT / profile / f"{profile}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        posts = json.load(f)
    return {p["folder_id"]: p for p in posts}


@st.cache_data
def load_ground_truth() -> list:
    path = PROCESSED / "ground_truth_multiyr.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def youth_topic_weights(ground_truth: list) -> dict[str, float]:
    weights = {t: 0.0 for t in TOPICS}
    total_weight = sum(p["weight_pct"] for p in ground_truth)
    for profile in ground_truth:
        desc = profile["profile_description"].lower()
        w = profile["weight_pct"] / total_weight
        for topic, kws in TOPIC_KEYWORDS.items():
            if any(k in desc for k in kws):
                weights[topic] += w
    total = sum(weights.values()) or 1
    return {t: v / total for t, v in weights.items()}


# ── Chart helpers ──────────────────────────────────────────────────────────────

def radar_chart(dfs: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid
    topics_closed = TOPICS + [TOPICS[0]]
    for i, (name, df) in enumerate(dfs.items()):
        means = [df[t].mean() for t in TOPICS] + [df[TOPICS[0]].mean()]
        fig.add_trace(go.Scatterpolar(
            r=means, theta=topics_closed,
            fill="toself", name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=colors[i % len(colors)].replace("rgb", "rgba").replace(")", ",0.15)"),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(255,255,255,0.03)",
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=10)),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def timeline_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid
    for i, t in enumerate(TOPICS):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[t], mode="lines+markers",
            name=t, line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(title="Data del post", gridcolor="#2A2A3C"),
        yaxis=dict(title="Punteggio Likert (1–5)", range=[0.5, 5.5], gridcolor="#2A2A3C"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.4),
    )
    return fig


def heatmap_chart(df: pd.DataFrame) -> go.Figure:
    matrix = df.set_index("folder_id")[TOPICS].astype(float)
    fig = px.imshow(
        matrix, color_continuous_scale="RdYlGn",
        zmin=1, zmax=5, aspect="auto",
        labels=dict(x="Topic", y="Post", color="Score"),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        coloraxis_colorbar=dict(title="Score"),
        xaxis=dict(tickangle=-30),
    )
    return fig


def comparison_chart(dfs: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid
    for i, (name, df) in enumerate(dfs.items()):
        means = [df[t].mean() for t in TOPICS]
        fig.add_trace(go.Bar(
            name=name, x=TOPICS, y=means,
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        xaxis=dict(tickangle=-20, gridcolor="#2A2A3C"),
        yaxis=dict(title="Punteggio medio (1–5)", range=[0, 5.5], gridcolor="#2A2A3C"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    return fig


def alignment_chart(politician_means: dict, youth_w: dict) -> tuple[go.Figure, float]:
    pol_vals = np.array([politician_means[t] / 5 for t in TOPICS])
    youth_vals = np.array([youth_w[t] for t in TOPICS])
    pol_norm = pol_vals / (pol_vals.sum() or 1)
    youth_norm = youth_vals / (youth_vals.sum() or 1)
    jsd = float(jensenshannon(pol_norm, youth_norm) ** 2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Politico (normalizzato)", x=TOPICS, y=pol_norm.tolist(),
        marker_color="#7C3AED",
    ))
    fig.add_trace(go.Bar(
        name="Giovani (proxy lessicale)", x=TOPICS, y=youth_norm.tolist(),
        marker_color="#10B981",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        xaxis=dict(tickangle=-20, gridcolor="#2A2A3C"),
        yaxis=dict(title="Peso relativo", gridcolor="#2A2A3C"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    return fig, jsd


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HDS Political Dashboard",
    page_icon="🇮🇹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0F0F14; }
[data-testid="stSidebar"] { background: #1A1A24; border-right: 1px solid #2A2A3C; }
.metric-card {
    background: #1A1A24; border: 1px solid #2A2A3C;
    border-radius: 12px; padding: 1.2rem 1.5rem;
}
.jsd-badge {
    display: inline-block; padding: 0.4rem 1rem;
    border-radius: 999px; font-weight: 700; font-size: 1.1rem;
}
h1, h2, h3 { color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇮🇹 HDS Dashboard")
    st.caption("Semantic Gap · Discorso Politico vs Giovani")
    st.divider()

    all_dfs = load_scored_csvs()
    if not all_dfs:
        st.error("Nessun CSV trovato in data/processed/.")
        st.stop()

    profiles = sorted(all_dfs.keys())
    selected = st.selectbox("Politico", profiles, format_func=lambda x: x.replace("_", " ").title())

    df_sel = all_dfs[selected]
    dates = df_sel["date"].dropna().sort_values()
    if len(dates) >= 2:
        date_range = st.date_input(
            "Intervallo date",
            value=(dates.iloc[0].date(), dates.iloc[-1].date()),
            min_value=dates.iloc[0].date(),
            max_value=dates.iloc[-1].date(),
        )
        if len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            df_sel = df_sel[(df_sel["date"] >= start) & (df_sel["date"] <= end)]

    st.divider()
    st.caption(f"Post analizzati: **{len(df_sel)}**")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("Political Discourse Dashboard")
st.caption(f"Profilo selezionato: **{selected.replace('_', ' ').title()}** · {len(df_sel)} post nel range")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Profilo Tematico",
    "📈 Timeline",
    "🔥 Heatmap Post",
    "⚖️ Confronto Politici",
    "🎯 Agenda Alignment",
])

# ── Tab 1: Radar ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Distribuzione media dei topic")
    st.caption("Punteggi Likert medi (1–5) per ciascun tema nel periodo selezionato.")

    col_metrics = st.columns(len(TOPICS))
    for i, t in enumerate(TOPICS):
        with col_metrics[i]:
            val = df_sel[t].mean() if t in df_sel.columns else 0
            st.metric(t.capitalize(), f"{val:.2f}")

    fig_radar = radar_chart({selected: df_sel})
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Tab 2: Timeline ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Andamento temporale dei topic")
    st.caption("Ogni punto corrisponde a un post. Mostra come l'enfasi sui temi cambia nel tempo.")
    df_time = df_sel.dropna(subset=["date"]).sort_values("date")
    if df_time.empty:
        st.info("Nessun dato con data valida nel range selezionato.")
    else:
        st.plotly_chart(timeline_chart(df_time), use_container_width=True)

# ── Tab 3: Heatmap ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Mappa punteggi per post")
    st.caption("Verde = alta rilevanza (5), Rosso = bassa rilevanza (1). Clicca su un post per vedere il testo.")

    fig_heat = heatmap_chart(df_sel)
    st.plotly_chart(fig_heat, use_container_width=True)

    posts_data = load_posts(selected)
    if posts_data:
        post_ids = df_sel["folder_id"].tolist()
        chosen = st.selectbox("Visualizza contenuto del post", post_ids)
        if chosen and chosen in posts_data:
            p = posts_data[chosen]
            with st.expander(f"Post {chosen}", expanded=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.markdown(f"**Tipo:** `{p.get('type','—')}`")
                    st.markdown(f"**Lingua:** `{p.get('language','—')}`")
                with c2:
                    st.markdown("**Caption:**")
                    st.markdown(p.get("caption", "—"))
                    if p.get("text"):
                        st.markdown("**Testo estratto (OCR/Whisper):**")
                        st.text(p["text"][:600] + ("…" if len(p.get("text","")) > 600 else ""))

# ── Tab 4: Confronto ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Confronto tra politici")
    if len(all_dfs) < 2:
        st.info(
            "Al momento è disponibile un solo CSV con punteggi (`ellyesse_AVG_HUMAN.csv`). "
            "Aggiungi `<profilo>_AVG_HUMAN.csv` in `data/processed/` per abilitare il confronto."
        )
    else:
        st.caption("Media punteggi Likert per tema, confronto tra tutti i profili con CSV disponibile.")
        st.plotly_chart(comparison_chart(all_dfs), use_container_width=True)

# ── Tab 5: Agenda Alignment ───────────────────────────────────────────────────
with tab5:
    st.subheader("Agenda Alignment Score")
    st.caption(
        "Confronto tra il profilo tematico del politico e le preoccupazioni dei giovani italiani "
        "(dati ISTAT, proxy lessicale sul `ground_truth_multiyr.json`)."
    )

    ground_truth = load_ground_truth()
    if not ground_truth:
        st.warning("File `ground_truth_multiyr.json` non trovato in `data/processed/`.")
    else:
        pol_means = {t: df_sel[t].mean() for t in TOPICS if t in df_sel.columns}
        youth_w = youth_topic_weights(ground_truth)

        fig_align, jsd = alignment_chart(pol_means, youth_w)

        col_jsd, col_interp = st.columns([1, 3])
        with col_jsd:
            color = "#10B981" if jsd < 0.15 else "#F59E0B" if jsd < 0.35 else "#EF4444"
            label = "Alto allineamento" if jsd < 0.15 else "Allineamento parziale" if jsd < 0.35 else "Gap elevato"
            st.markdown(f"""
            <div class="metric-card" style="text-align:center">
              <p style="margin:0;color:#94A3B8;font-size:0.85rem">JSD (proxy)</p>
              <p class="jsd-badge" style="background:{color}22;color:{color};margin:0.5rem 0">{jsd:.3f}</p>
              <p style="margin:0;color:{color};font-size:0.9rem">{label}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_interp:
            st.info(
                "**Nota metodologica:** Il JSD qui calcolato è un *proxy lessicale* basato su "
                "keyword matching nelle descrizioni dei profili ISTAT. "
                "Il JSD completo del progetto usa un Joint Embedding Space (sentence-transformers)."
            )

        st.plotly_chart(fig_align, use_container_width=True)

        with st.expander("Dettaglio pesi giovani (top profili per peso)"):
            top_profiles = sorted(ground_truth, key=lambda x: x["weight_pct"], reverse=True)[:10]
            for p in top_profiles:
                st.markdown(f"**{p['weight_pct']:.1f}%** ({p['represented_youth']:,} giovani) — {p['profile_description'][:150]}…")
