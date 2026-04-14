"""
Alignment Results — Visual Dashboard
=====================================
Reads alignment_results.json and generates a self-contained HTML dashboard.

Run:
    tools/classification/venv/bin/python tools/alignment/visualize_alignment.py
"""

import json
import os
import numpy as np
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "alignment_results.json")
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "alignment_dashboard.html")

# ─── palette ──────────────────────────────────────────────────────────────────
YOUTH_COLOR  = "#4FC3F7"
POL_COLORS   = ["#EF5350", "#66BB6A", "#FFA726", "#AB47BC", "#26C6DA"]
BG_COLOR     = "#0f1117"
CARD_COLOR   = "#1a1d27"
TEXT_COLOR   = "#e0e0e0"
GRID_COLOR   = "#2a2d3a"

# ─── load data ────────────────────────────────────────────────────────────────
with open(RESULTS_PATH, encoding="utf-8") as f:
    data = json.load(f)

categories   = data.get("meaningful_categories", data["categories"])  # excludes noise sinks
youth_dist   = np.array([data["youth_distribution"][k] for k in categories])
politicians  = data["politicians"]          # sorted by score descending
pol_names    = [p["politician"] for p in politicians]
pol_dists    = {p["politician"]: np.array([p["distribution"][k] for k in categories])
                for p in politicians}
pol_scores   = {p["politician"]: p["alignment_score"] for p in politicians}
pol_noise    = {p["politician"]: p.get("noise_score", 0) for p in politicians}
pol_colors   = {name: POL_COLORS[i % len(POL_COLORS)] for i, name in enumerate(pol_names)}

# Pretty-print politician names (strip underscores, title-case)
def pretty(name): return name.replace("_", " ").replace("official", "").strip().title()

# ─── shared layout defaults ───────────────────────────────────────────────────
def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    base.update(kwargs)
    return base


# ══════════════════════════════════════════════════════════════════════════════
#  1. ALIGNMENT SCORE RANKING  — horizontal bar
# ══════════════════════════════════════════════════════════════════════════════
def fig_ranking():
    names  = [pretty(p["politician"]) for p in politicians]
    scores = [p["alignment_score"] for p in politicians]
    colors = [pol_colors[p["politician"]] for p in politicians]

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(size=14, color=TEXT_COLOR),
        hovertemplate="<b>%{y}</b><br>Alignment Score: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **dark_layout(title=dict(text="Agenda Alignment Score", font=dict(size=18))),
        xaxis=dict(range=[0, 1.05], showgrid=True, gridcolor=GRID_COLOR,
                   title="Score (1 = perfect alignment)"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        showlegend=False,
        height=180 + 60 * len(politicians),
    )
    # Reference line at 0.5
    fig.add_vline(x=0.5, line_dash="dot", line_color="#555", line_width=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  2. RADAR CHART — youth vs each politician
# ══════════════════════════════════════════════════════════════════════════════
def fig_radar():
    cats_closed = categories + [categories[0]]  # close the polygon

    fig = go.Figure()

    # Youth baseline
    youth_closed = list(youth_dist) + [youth_dist[0]]
    fig.add_trace(go.Scatterpolar(
        r=youth_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor=f"rgba(79,195,247,0.15)",
        line=dict(color=YOUTH_COLOR, width=2),
        name="Giovani (ISTAT)",
    ))

    for pol in politicians:
        name   = pol["politician"]
        dist   = list(pol_dists[name]) + [pol_dists[name][0]]
        color  = pol_colors[name]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatterpolar(
            r=dist,
            theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(color=color, width=2, dash="dash"),
            name=pretty(name),
        ))

    fig.update_layout(
        **dark_layout(title=dict(text="Distribuzione Temi — Radar", font=dict(size=18))),
        polar=dict(
            bgcolor=CARD_COLOR,
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=True, tickfont=dict(size=9, color="#888"),
                gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
            ),
        ),
        legend=dict(bgcolor=BG_COLOR, bordercolor=GRID_COLOR, borderwidth=1),
        height=520,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  3. GROUPED BAR — per category, youth vs politicians
# ══════════════════════════════════════════════════════════════════════════════
def fig_grouped_bar():
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Giovani (ISTAT)",
        x=categories,
        y=list(youth_dist),
        marker=dict(color=YOUTH_COLOR),
        hovertemplate="<b>Giovani</b><br>%{x}: %{y:.3f}<extra></extra>",
    ))

    for pol in politicians:
        name  = pol["politician"]
        dist  = pol_dists[name]
        fig.add_trace(go.Bar(
            name=pretty(name),
            x=categories,
            y=list(dist),
            marker=dict(color=pol_colors[name]),
            hovertemplate=f"<b>{pretty(name)}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        **dark_layout(title=dict(text="Distribuzione per Categoria", font=dict(size=18))),
        barmode="group",
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="Probabilità media", showgrid=True,
            gridcolor=GRID_COLOR, range=[0, 1],
        ),
        legend=dict(bgcolor=BG_COLOR, bordercolor=GRID_COLOR, borderwidth=1),
        height=420,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  4. HEATMAP — politicians + youth × categories
# ══════════════════════════════════════════════════════════════════════════════
def fig_heatmap():
    rows       = ["Giovani (ISTAT)"] + [pretty(p["politician"]) for p in politicians]
    raw_rows   = ["__youth__"] + [p["politician"] for p in politicians]
    z          = []
    for r in raw_rows:
        if r == "__youth__":
            z.append(list(youth_dist))
        else:
            z.append(list(pol_dists[r]))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=categories,
        y=rows,
        colorscale="Blues",
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Score", font=dict(color=TEXT_COLOR)),
            tickfont=dict(color=TEXT_COLOR),
        ),
    ))
    fig.update_layout(
        **dark_layout(title=dict(text="Heatmap Temi", font=dict(size=18))),
        xaxis=dict(showgrid=False, side="bottom"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=200 + 70 * len(rows),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  5. DIVERGENCE BREAKDOWN — per-category contribution to JSD
# ══════════════════════════════════════════════════════════════════════════════
def _per_category_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Approximate per-category JSD contribution via M = (P+Q)/2."""
    eps = 1e-10
    m   = (p + q) / 2.0
    kl_pm = np.where(p > eps, p * np.log(p / (m + eps)), 0)
    kl_qm = np.where(q > eps, q * np.log(q / (m + eps)), 0)
    return (kl_pm + kl_qm) / 2.0


def fig_divergence_breakdown():
    fig = go.Figure()

    for pol in politicians:
        name  = pol["politician"]
        dist  = pol_dists[name]
        contrib = _per_category_jsd(youth_dist, dist)
        fig.add_trace(go.Bar(
            name=pretty(name),
            x=categories,
            y=list(contrib),
            marker=dict(color=pol_colors[name]),
            hovertemplate=f"<b>{pretty(name)}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        **dark_layout(title=dict(
            text="Contributo per Categoria alla Divergenza (JSD)",
            font=dict(size=18),
        )),
        barmode="group",
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="Contributo JSD", showgrid=True,
            gridcolor=GRID_COLOR,
        ),
        legend=dict(bgcolor=BG_COLOR, bordercolor=GRID_COLOR, borderwidth=1),
        height=400,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  6. SCATTER — per-category: youth attention vs politician attention
# ══════════════════════════════════════════════════════════════════════════════
def fig_scatter_attention():
    """
    Each dot = one (politician, category) pair.
    X axis = youth attention on that category.
    Y axis = politician attention on that category.
    Diagonal = perfect alignment.
    """
    fig = go.Figure()

    # Diagonal reference line
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#555", dash="dot", width=1))

    for pol in politicians:
        name  = pol["politician"]
        dist  = pol_dists[name]
        fig.add_trace(go.Scatter(
            x=list(youth_dist),
            y=list(dist),
            mode="markers+text",
            marker=dict(color=pol_colors[name], size=14, line=dict(width=1, color="#fff")),
            text=categories,
            textposition="top center",
            textfont=dict(size=9, color=TEXT_COLOR),
            name=pretty(name),
            hovertemplate=(
                f"<b>{pretty(name)}</b><br>"
                "Categoria: %{text}<br>"
                "Giovani: %{x:.3f}<br>"
                "Politico: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **dark_layout(title=dict(
            text="Attenzione Giovani vs Politici per Categoria",
            font=dict(size=18),
        )),
        xaxis=dict(
            title="Attenzione Giovani (ISTAT)",
            showgrid=True, gridcolor=GRID_COLOR, range=[-0.02, 1],
        ),
        yaxis=dict(
            title="Attenzione Politico",
            showgrid=True, gridcolor=GRID_COLOR, range=[-0.02, 1],
        ),
        legend=dict(bgcolor=BG_COLOR, bordercolor=GRID_COLOR, borderwidth=1),
        height=500,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLE HTML
# ══════════════════════════════════════════════════════════════════════════════
def fig_to_div(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def build_score_cards() -> str:
    cards = []
    for pol in politicians:
        name  = pol["politician"]
        score = pol["alignment_score"]
        noise = pol.get("noise_score", 0)
        color = pol_colors[name]
        pct   = int(score * 100)
        noise_pct  = int(noise * 100)
        noise_color = "#ef5350" if noise > 0.25 else "#ffa726" if noise > 0.15 else "#66bb6a"
        cards.append(f"""
        <div class="score-card">
            <div class="score-name">{pretty(name)}</div>
            <div class="score-value" style="color:{color}">{score:.4f}</div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>
            </div>
            <div class="score-label">Alignment Score</div>
            <div class="noise-row">
                <span class="noise-label">Noise (propaganda)</span>
                <span class="noise-value" style="color:{noise_color}">{noise:.3f}</span>
            </div>
        </div>""")
    return "\n".join(cards)


def build_html(figures: dict) -> str:
    divs = {k: fig_to_div(v) for k, v in figures.items()}

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agenda Alignment Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: Inter, sans-serif;
    padding: 24px;
    min-height: 100vh;
  }}
  h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .subtitle {{
    font-size: 0.95rem;
    color: #888;
    margin-bottom: 32px;
  }}
  .section-title {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666;
    margin: 40px 0 12px;
  }}
  .card {{
    background: {CARD_COLOR};
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid {GRID_COLOR};
  }}
  .score-cards {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }}
  .score-card {{
    background: {CARD_COLOR};
    border: 1px solid {GRID_COLOR};
    border-radius: 12px;
    padding: 20px 24px;
    min-width: 200px;
    flex: 1;
  }}
  .score-name {{
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 8px;
    text-transform: capitalize;
  }}
  .score-value {{
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 12px;
  }}
  .score-bar-bg {{
    height: 4px;
    background: {GRID_COLOR};
    border-radius: 2px;
    margin-bottom: 6px;
  }}
  .score-bar-fill {{
    height: 4px;
    border-radius: 2px;
    transition: width 0.6s ease;
  }}
  .score-label {{
    font-size: 0.75rem;
    color: #555;
  }}
  .noise-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid {GRID_COLOR};
  }}
  .noise-label {{ font-size: 0.72rem; color: #555; }}
  .noise-value {{ font-size: 0.85rem; font-weight: 600; }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  @media (max-width: 900px) {{
    .two-col {{ grid-template-columns: 1fr; }}
  }}
  .note {{
    font-size: 0.78rem;
    color: #555;
    margin-top: 10px;
    line-height: 1.5;
  }}
</style>
</head>
<body>

<h1>Agenda Alignment Dashboard</h1>
<p class="subtitle">
  Divergenza semantica tra preoccupazioni giovanili (ISTAT) e contenuti social dei politici.
  Punteggio 1 = allineamento perfetto &nbsp;·&nbsp; 0 = totale divergenza.
</p>

<p class="section-title">Punteggi Finali</p>
<div class="score-cards">
{build_score_cards()}
</div>

<p class="section-title">Ranking</p>
<div class="card">
{divs["ranking"]}
</div>

<p class="section-title">Profilo Tematico</p>
<div class="two-col">
  <div class="card">{divs["radar"]}</div>
  <div class="card">{divs["heatmap"]}</div>
</div>

<p class="section-title">Analisi per Categoria</p>
<div class="card">
{divs["grouped_bar"]}
<p class="note">
  Mostra per ogni categoria quanto spazio occupa nei contenuti dei politici rispetto
  all'attenzione che i giovani ripongono in quel tema.
</p>
</div>

<p class="section-title">Attenzione: Giovani vs Politici</p>
<div class="card">
{divs["scatter"]}
<p class="note">
  Ogni punto = (politico, categoria). La diagonale tratteggiata è la linea di
  allineamento perfetto. Punti sotto = politico parla di quel tema meno dei giovani.
  Punti sopra = parla di più.
</p>
</div>

<p class="section-title">Divergenza per Categoria</p>
<div class="card">
{divs["divergence"]}
<p class="note">
  Contributo di ogni categoria al JSD totale. Picchi alti = categorie dove il gap
  è maggiore. Utile per capire dove si concentra il disallineamento.
</p>
</div>

</body>
</html>"""
    return html


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building figures...")
    figures = {
        "ranking":     fig_ranking(),
        "radar":       fig_radar(),
        "grouped_bar": fig_grouped_bar(),
        "heatmap":     fig_heatmap(),
        "scatter":     fig_scatter_attention(),
        "divergence":  fig_divergence_breakdown(),
    }

    print("Assembling HTML...")
    html = build_html(figures)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved: {OUTPUT_PATH}")
    print("Open in browser: open tools/alignment/alignment_dashboard.html")
