# Matchday ML

Matchday ML turns historical Premier League results into a small, explainable machine-learning workflow. It cleans source CSVs, creates team form features using matches played **before** each fixture, trains a multiclass model, reports a chronological holdout evaluation, and estimates Home win / Draw / Away win probabilities for a selected matchup.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Then open the local URL printed in the terminal. In the sidebar, choose seasons and click **Load match data**. The first load needs an internet connection; CSV files are cached under `data/` and are excluded from Git.

## Data and leakage controls

The app downloads English Premier League (`E0`) CSVs from [Football-Data.co.uk](https://www.football-data.co.uk/downloadm.php). It uses final scores only to derive the target label and historical form. Form features are calculated from each club's previous five matches at the time of the fixture. Full-time scores, match statistics, odds, and other post-match information are never model inputs.

The final 20% of fixtures, in date order, form the holdout set. This gives a more realistic estimate than a random split, though it is still only a small historical benchmark and not a guarantee about future matches. The project is for learning and analysis, not betting advice.

## What the dashboard shows

- Source row counts, missing/duplicate cleanup, and a match preview
- Recent pre-match form for the two clubs
- Home / draw / away probability estimates
- A date-ordered holdout accuracy, balanced accuracy, log loss, and confusion matrix
- Downloadable cleaned results and engineered features

## Stack

Python · pandas · scikit-learn · Streamlit · Plotly

## Attribution

Match data is provided by [Football-Data.co.uk](https://www.football-data.co.uk/). Consult its notes and current terms before redistributing any downloaded data. This repository does not include the source CSVs.
