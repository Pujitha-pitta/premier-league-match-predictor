"""Data loading, cleaning, feature engineering, and time-aware modelling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
REQUIRED = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
OUTCOMES = ("H", "D", "A")
WINDOW = 5


@dataclass
class ModelBundle:
    model: object
    feature_columns: list[str]
    features: pd.DataFrame
    metrics: dict[str, float]
    matrix: list[list[int]]
    labels: list[str]
    train_size: int
    test_size: int


def season_codes(start: int, end: int) -> list[str]:
    """Return football-data season codes, e.g. 2018..2024 -> 1819..2425."""
    return [f"{year % 100:02d}{(year + 1) % 100:02d}" for year in range(start, end)]


def fetch_seasons(codes: Iterable[str], data_dir: str | Path = "data") -> tuple[pd.DataFrame, list[str]]:
    """Fetch/cached season CSVs. Missing files are reported rather than fatal."""
    target = Path(data_dir)
    target.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    messages: list[str] = []
    headers = {"User-Agent": "MatchdayML/1.0 (educational sports analytics project)"}
    for code in codes:
        path = target / f"E0_{code}.csv"
        try:
            if path.exists() and path.stat().st_size:
                raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
                messages.append(f"{code}: cached")
            else:
                response = requests.get(BASE_URL.format(season=code), headers=headers, timeout=25)
                response.raise_for_status()
                if not response.content.strip():
                    raise ValueError("empty response")
                path.write_bytes(response.content)
                raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
                messages.append(f"{code}: downloaded")
            if all(column in raw.columns for column in REQUIRED):
                raw["Season"] = code
                parts.append(raw)
            else:
                messages.append(f"{code}: file has no recognizable match rows")
        except Exception as exc:  # source can lack a season file or briefly be unavailable
            messages.append(f"{code}: skipped ({type(exc).__name__})")
    if not parts:
        raise ValueError("No match files were loaded. Check the network and try a different season range.")
    return clean_matches(pd.concat(parts, ignore_index=True, sort=False)), messages


def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and keep completed, uniquely identified matches."""
    missing = [column for column in REQUIRED if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")
    data = raw.copy()
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce", format="mixed")
    data["FTHG"] = pd.to_numeric(data["FTHG"], errors="coerce")
    data["FTAG"] = pd.to_numeric(data["FTAG"], errors="coerce")
    data["HomeTeam"] = data["HomeTeam"].astype("string").str.strip()
    data["AwayTeam"] = data["AwayTeam"].astype("string").str.strip()
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data[(data["HomeTeam"] != "") & (data["AwayTeam"] != "")]
    data["FTHG"] = data["FTHG"].astype(int)
    data["FTAG"] = data["FTAG"].astype(int)
    data["FTR"] = data.apply(lambda row: "H" if row.FTHG > row.FTAG else ("A" if row.FTHG < row.FTAG else "D"), axis=1)
    data = data.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
    return data.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def _summary(history: list[dict], venue: str | None = None) -> dict[str, float]:
    selected = [item for item in history if venue is None or item["venue"] == venue][-WINDOW:]
    if not selected:
        selected = history[-WINDOW:]
    if not selected:
        return {"games": 0.0, "points": 0.0, "goal_diff": 0.0, "goals_for": 0.0, "goals_against": 0.0, "win_rate": 0.0}
    size = len(selected)
    return {
        "games": float(size),
        "points": sum(item["points"] for item in selected) / size,
        "goal_diff": sum(item["goal_diff"] for item in selected) / size,
        "goals_for": sum(item["goals_for"] for item in selected) / size,
        "goals_against": sum(item["goals_against"] for item in selected) / size,
        "win_rate": sum(item["points"] == 3 for item in selected) / size,
    }


def engineer_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build rolling form from each club's prior matches only, then update state."""
    data = clean_matches(matches) if "FTR" not in matches.columns or not matches.index.is_monotonic_increasing else matches.copy()
    histories: dict[str, list[dict]] = {}
    rows: list[dict] = []
    for row in data.sort_values("Date").itertuples(index=False):
        home = histories.setdefault(str(row.HomeTeam), [])
        away = histories.setdefault(str(row.AwayTeam), [])
        features: dict[str, float] = {}
        for prefix, history in (("home", home), ("away", away)):
            for name, value in _summary(history).items():
                features[f"{prefix}_{name}_last5"] = value
        for prefix, history, venue in (("home", home, "home"), ("away", away, "away")):
            for name, value in _summary(history, venue).items():
                features[f"{prefix}_{name}_venue_last5"] = value
        features["home_advantage"] = 1.0
        rows.append({"Date": row.Date, "HomeTeam": str(row.HomeTeam), "AwayTeam": str(row.AwayTeam), "FTR": row.FTR, **features})

        home_goals, away_goals = int(row.FTHG), int(row.FTAG)
        home_points = 3 if home_goals > away_goals else (1 if home_goals == away_goals else 0)
        away_points = 3 if away_goals > home_goals else (1 if home_goals == away_goals else 0)
        home.append({"venue": "home", "points": home_points, "goal_diff": home_goals - away_goals, "goals_for": home_goals, "goals_against": away_goals})
        away.append({"venue": "away", "points": away_points, "goal_diff": away_goals - home_goals, "goals_for": away_goals, "goals_against": home_goals})
    return pd.DataFrame(rows)


def train_evaluate(features: pd.DataFrame) -> ModelBundle:
    """Use the final 20% of time-ordered matches as a holdout set."""
    feature_columns = [column for column in features.columns if column.startswith(("home_", "away_")) or column == "home_advantage"]
    data = features.dropna(subset=["FTR"]).sort_values("Date").reset_index(drop=True)
    split = max(1, int(len(data) * 0.8))
    if split < 30 or len(data) - split < 10:
        raise ValueError("Load more seasons to create a useful chronological train/test split.")
    train, test = data.iloc[:split], data.iloc[split:]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
    model.fit(train[feature_columns], train["FTR"])
    prediction = model.predict(test[feature_columns])
    probabilities = model.predict_proba(test[feature_columns])
    labels = list(model.classes_)
    metrics = {
        "accuracy": float(accuracy_score(test["FTR"], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(test["FTR"], prediction)),
        "log_loss": float(log_loss(test["FTR"], probabilities, labels=labels)),
        "baseline_accuracy": float(accuracy_score(test["FTR"], [train["FTR"].mode().iloc[0]] * len(test))),
    }
    matrix = confusion_matrix(test["FTR"], prediction, labels=labels).tolist()
    return ModelBundle(model, feature_columns, features, metrics, matrix, labels, len(train), len(test))


def matchup_features(bundle: ModelBundle, home_team: str, away_team: str) -> pd.DataFrame:
    """Form snapshot as of the latest completed fixture for the chosen teams."""
    rows = bundle.features
    if home_team == away_team:
        raise ValueError("Choose two different clubs.")

    def team_snapshot(team: str, upcoming_venue: str) -> dict[str, float]:
        appearances = rows[(rows["HomeTeam"] == team) | (rows["AwayTeam"] == team)]
        if appearances.empty:
            raise ValueError(f"No earlier matches found for {team}.")
        latest = appearances.iloc[-1]
        side = "home" if latest["HomeTeam"] == team else "away"
        snapshot: dict[str, float] = {}
        for column in bundle.feature_columns:
            if column.startswith(side + "_") and "_venue_" not in column:
                snapshot[column[len(side) + 1:]] = float(latest[column])

        venue_rows = rows[rows[f"{upcoming_venue.title()}Team"] == team]
        if not venue_rows.empty:
            venue_latest = venue_rows.iloc[-1]
            actual_side = upcoming_venue
            for column in bundle.feature_columns:
                marker = actual_side + "_"
                if column.startswith(marker) and "_venue_" in column:
                    key = column[len(marker):]
                    snapshot[key] = float(venue_latest[column])
        return snapshot

    home_snapshot = team_snapshot(home_team, "home")
    away_snapshot = team_snapshot(away_team, "away")
    values = {f"home_{key}": value for key, value in home_snapshot.items()}
    values.update({f"away_{key}": value for key, value in away_snapshot.items()})
    values["home_advantage"] = 1.0
    return pd.DataFrame([values], columns=bundle.feature_columns).fillna(0)
