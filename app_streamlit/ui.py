# ui.py
import streamlit as st

GITHUB_URL = "https://github.com/ton-org/tu-repo"     # ← remplace
DOCS_URL   = None
DATA_URL   = "https://data.worldbank.org/indicator/SP.DYN.IMRT.IN"

def _ensure_page_config(title: str):
    if not st.session_state.get("_page_config_set"):
        #  appelle render_header_app() le plus haut possible dans chaque page
        st.set_page_config(page_title=title, layout="wide")
        st.session_state["_page_config_set"] = True

def render_header(title: str, subtitle: str, version: str | None = "0.3.2",
                  logo_left: str | None = None, logo_right: str | None = None,
                  links: list[tuple[str, str]] | None = None):

    _ensure_page_config(title)

    # 
    st.markdown("""
    <style>
      .app-header { position: sticky; top: 0; z-index: 1000;
        width: 100vw; margin-left: calc(50% - 50vw);
        background: linear-gradient(90deg,#0f172a 0%,#1f2937 100%);
        color: #fff; border-bottom: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 2px 12px rgba(0,0,0,.15); }
      .app-header-wrap { max-width: 1200px; margin: 0 auto; padding: .75rem 1rem; }
      .app-header-row { display: flex; align-items: center; gap: 1rem; }
      .grow { flex: 1; }
      .title { font-weight: 800; font-size: 1.15rem; margin: 0; line-height: 1.25; }
      .subtitle { opacity: .85; font-size: .92rem; margin: .15rem 0 0 0; }
      .links a { color: #93c5fd; text-decoration: none; margin-left: 1rem; }
      .badge { background: #111827; border: 1px solid rgba(255,255,255,.18);
               border-radius: 999px; padding: .2rem .55rem; font-size: .78rem; }
      .logo { height: 28px; vertical-align: middle; }
      @media (max-width: 640px){ .subtitle { display:none; } .links a { margin-left: .6rem; } }
    </style>
    """, unsafe_allow_html=True)

    left = []
    if logo_left:
        left.append(f'<img class="logo" src="{logo_left}" alt="logo">')
    left.append(f'<div><div class="title">{title}</div>')
    left.append(f'<div class="subtitle">{subtitle}</div></div>')

    right = []
    if version:
        right.append(f'<span class="badge">v{version}</span>')
    if links:
        right.append('<span class="links">' + "".join(
            f'<a href="{u}" target="_blank" rel="noopener">{t}</a>' for t,u in links if u
        ) + '</span>')
    if logo_right:
        right.append(f'<img class="logo" src="{logo_right}" alt="logo right" style="margin-left:.5rem;">')

    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-header-wrap">
            <div class="app-header-row">
              <div>{"".join(left)}</div>
              <div class="grow"></div>
              <div>{"".join(right)}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)

def render_header_app():
    render_header(
        title="Pronóstico de Mortalidad Infantil (t+1)",
        subtitle="Modelo explicable • Países 2000–2023 • Streamlit + scikit-learn",
        version="0.3.2",
        logo_left=None,
        logo_right=None,
        links=[("GitHub", GITHUB_URL), ("Indicador WDI", DATA_URL), ("Docs", DOCS_URL)],
    )