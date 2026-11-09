from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd

from engine import fetch_seasons, matchup_features, season_codes, train_evaluate

st.set_page_config(page_title="Matchday ML", page_icon="⚽", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,600&display=swap');
:root { --ink:#15251f; --muted:#78847d; --paper:#f5f4ee; --green:#17362b; --line:#dfe2d8; }
.stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#eeeee7; border-right:1px solid var(--line); }
.block-container { padding-top:2rem; max-width:1250px; }
.eyebrow { font:500 11px 'DM Mono',monospace; letter-spacing:.18em; color:#718078; text-transform:uppercase; }
.hero { font-size:clamp(52px,8vw,94px); font-weight:700; letter-spacing:-.075em; line-height:.93; color:var(--ink); margin:.4rem 0 1.2rem; }
.hero em { font-family:'Playfair Display',serif; font-weight:600; letter-spacing:-.06em; }
.deck { max-width:690px; color:#68766e; font-size:17px; line-height:1.7; }
.section-title { font:600 26px 'DM Sans',sans-serif; letter-spacing:-.04em; margin:.5rem 0 .8rem; }
.stButton>button { background:#17362b; color:#fff; border:0; border-radius:4px; font-weight:600; }
.stButton>button:hover { background:#2a5041; color:#fff; border:0; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px 18px; }
div[data-testid="stMetricLabel"] { color:#77847c; font:500 10px 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
div[data-testid="stMetricValue"] { color:var(--ink); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Premier League · Match analytics lab</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Read the <em>form.</em><br>Estimate the match.</div>', unsafe_allow_html=True)
st.markdown('<p class="deck">A transparent, educational model that learns from historical results. Compare each club’s recent form, see a time-based model evaluation, and explore home / draw / away probabilities.</p>', unsafe_allow_html=True)
st.caption("Learning project only · estimates are uncertain and are not betting advice.")

with st.sidebar:
    st.markdown("### Build your dataset")
    st.write("Choose completed Premier League seasons. Source files are cached locally so later launches do not need to download them again.")
    start_year = st.selectbox("First season", range(1993, 2025), index=22, format_func=lambda y: f"{y}/{str(y + 1)[-2:]}")
    end_year = st.selectbox("Last season (inclusive)", range(1994, 2026), index=31, format_func=lambda y: f"{y - 1}/{str(y)[-2:]}")
    st.caption("Data source: Football-Data.co.uk, league code E0.")
    load_button = st.button("Load match data", use_container_width=True)
    if start_year >= end_year:
        st.warning("The last season must be after the first season.")

if load_button:
    if start_year >= end_year:
        st.error("Choose a valid season range.")
    else:
        with st.spinner("Loading and cleaning match files…"):
            try:
                matches, messages = fetch_seasons(season_codes(start_year, end_year))
                st.session_state["matches"] = matches
                st.session_state["load_messages"] = messages
                st.session_state.pop("bundle", None)
            except Exception as error:
                st.error(str(error))

matches: pd.DataFrame | None = st.session_state.get("matches")
if matches is None:
    st.info("Choose a season range in the left panel and press **Load match data** to begin.")
    st.markdown("### What happens to the data")
    st.markdown("1. Season CSVs are combined and checked for missing scores, dates, and duplicate fixtures.  2. Team form is calculated from each club’s **previous five matches only**.  3. A logistic-regression classifier is evaluated on the latest 20% of matches in date order.")
    st.stop()

st.divider()
st.markdown('<div class="eyebrow">Dataset quality check</div>', unsafe_allow_html=True)
first_col, second_col, third_col, fourth_col = st.columns(4)
with first_col:
    st.metric("Completed matches", f"{len(matches):,}")
with second_col:
    st.metric("Clubs", matches["HomeTeam"].nunique())
with third_col:
    st.metric("First fixture", matches["Date"].min().strftime("%b %Y"))
with fourth_col:
    st.metric("Most recent", matches["Date"].max().strftime("%b %Y"))

with st.expander("See data import and cleaning notes"):
    for message in st.session_state.get("load_messages", []):
        st.write(f"• {message}")
    st.write("Missing dates, teams, and final scores are removed. Duplicate fixtures are reduced to one record. Labels are rebuilt from the final score. Odds and post-match stats are not used as predictors.")

st.markdown('<div class="section-title">Train and evaluate</div>', unsafe_allow_html=True)
st.write("The holdout contains the latest 20% of matches by date. This avoids randomly mixing future results into the training set.")
if st.button("Train model and show evaluation", type="primary"):
    with st.spinner("Building pre-match form and fitting the model…"):
        try:
            from engine import engineer_features
            features = engineer_features(matches)
            bundle = train_evaluate(features)
            st.session_state["bundle"] = bundle
        except Exception as error:
            st.error(str(error))

bundle = st.session_state.get("bundle")
if bundle:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Holdout accuracy", f"{bundle.metrics['accuracy']:.1%}")
    m2.metric("Balanced accuracy", f"{bundle.metrics['balanced_accuracy']:.1%}")
    m3.metric("Log loss", f"{bundle.metrics['log_loss']:.3f}", help="Lower is better. Measures the quality of probability estimates.")
    m4.metric("Majority baseline", f"{bundle.metrics['baseline_accuracy']:.1%}")
    st.caption(f"Chronological split: {bundle.train_size:,} training fixtures · {bundle.test_size:,} holdout fixtures · classes: Home win, Draw, Away win.")
    matrix = pd.DataFrame(bundle.matrix, index=bundle.labels, columns=bundle.labels)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0,"#eef1e9"],[1,"#17362b"]], labels={"x":"Predicted result","y":"Actual result","color":"Matches"}, aspect="auto")
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=330, paper_bgcolor="#f5f4ee", plot_bgcolor="#f5f4ee", font=dict(family="DM Sans",color="#15251f"))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download cleaned match table", matches.to_csv(index=False).encode("utf-8"), "premier_league_matches_clean.csv", "text/csv")
    st.download_button("Download pre-match features", bundle.features.to_csv(index=False).encode("utf-8"), "premier_league_form_features.csv", "text/csv")

st.divider()
st.markdown('<div class="eyebrow">Explore a matchup</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Who has the form?</div>', unsafe_allow_html=True)
if not bundle:
    st.info("Train the model first to unlock matchup probabilities.")
else:
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home club", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away_options = [team for team in teams if team != home_team]
    away_team = c2.selectbox("Away club", away_options, index=away_options.index("Liverpool") if "Liverpool" in away_options else 0)
    if st.button("Estimate outcome probabilities"):
        try:
            inputs = matchup_features(bundle, home_team, away_team)
            probabilities = bundle.model.predict_proba(inputs)[0]
            probability_map = dict(zip(bundle.model.classes_, probabilities))
            result_names = {"H":"Home win", "D":"Draw", "A":"Away win"}
            c1, c2, c3 = st.columns(3)
            for col, code in zip((c1,c2,c3), ("H","D","A")):
                col.metric(result_names[code], f"{probability_map.get(code,0):.1%}")
            st.caption("These probabilities use each club’s most recently available form in the loaded dataset. They are illustrative and not live forecasts.")
        except Exception as error:
            st.error(str(error))

st.divider()
st.caption("Data: [Football-Data.co.uk](https://www.football-data.co.uk/downloadm.php) · Python · pandas · scikit-learn · Streamlit · Plotly")
