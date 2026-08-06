# %%
from pathlib import Path
import pandas as pd
import wandb
import json
import numpy as np

# =========================
# Konfiguracja
# =========================
ENTITY = "politechnika-gnn-thesis"
PROJECT = "politechnika-gnn-thesis"

# Run zostanie wybrany, jeśli jego nazwa zawiera przynajmniej jedno z tych słów
KEYWORDS = [
   "HCRExtended3"
]

# Jeśli True: run musi zawierać wszystkie słowa kluczowe
# Jeśli False: wystarczy przynajmniej jedno słowo (domyślnie)
REQUIRE_ALL_KEYWORDS = False

# Eksportować tylko zakończone runy?
ONLY_FINISHED = True

OUTPUT_DIR = Path("/Users/monika/GraphNeuralNetwork_Thesis/outputs/wandb_exports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "all_selected_runs_history_HCRExtended3.csv"


# =========================
# Funkcje pomocnicze
# =========================
def matches_keywords(run_name: str, keywords: list[str], require_all: bool = False) -> bool:
    """Sprawdza, czy nazwa runu pasuje do słów kluczowych."""
    run_name = run_name.lower()
    keywords = [keyword.lower() for keyword in keywords]

    if require_all:
        return all(keyword in run_name for keyword in keywords)

    return any(keyword in run_name for keyword in keywords)


def flatten_dict(data: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Spłaszcza zagnieżdżony config, np.:
    {'model': {'hidden_dim': 64}}
    -> {'model.hidden_dim': 64}
    """
    flattened = {}

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value

    return flattened


# =========================
# Pobranie danych z W&B
# =========================
api = wandb.login(key="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")
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

all_run_histories = []
selected_runs = []

def make_csv_safe(value):
    """Zamienia listy, dicty i tablice NumPy na pojedynczy tekst JSON."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)

    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=False, default=str)

    return value

for run in runs:
    if ONLY_FINISHED and run.state != "finished":
        continue

    if not matches_keywords(
        run_name=run.name,
        keywords=KEYWORDS,
        require_all=REQUIRE_ALL_KEYWORDS,
    ):
        continue

    print(f"Pobieram: {run.name} | id={run.id} | state={run.state}")

    # Bez `keys=`: pobierze wszystkie metryki logowane przez wandb.log()
    history_rows = list(run.scan_history())

    if not history_rows:
        print(f"  Pomijam {run.name}: brak historii metryk.")
        continue

    history_df = pd.DataFrame(history_rows)

    # Metadane runu
    run_metadata = {
        "run_id": run.id,
        "run_name": run.name,
        "run_group": run.group,
        "run_job_type": run.job_type,
        "run_state": run.state,
        "run_url": run.url,
    }

    # Dopisanie całego configu jako kolumn config.*
    config = flatten_dict(dict(run.config))
    config_metadata = {
        f"config.{key}": value
        for key, value in config.items()
    }

    # Dopisanie końcowych metryk/statystyk z W&B Summary jako summary.*
    summary = flatten_dict(dict(run.summary))
    summary_metadata = {
        f"summary.{key}": value
        for key, value in summary.items()
    }

    metadata = {
        **run_metadata,
        **config_metadata,
        **summary_metadata,
    }

    for column_name, value in metadata.items():
        history_df[column_name] = make_csv_safe(value)

    all_run_histories.append(history_df)
    selected_runs.append(
        {
            "run_name": run.name,
            "run_id": run.id,
            "rows": len(history_df),
        }
    )

# =========================
# Zapis do jednego CSV
# =========================
if not all_run_histories:
    raise ValueError(
        "Nie znaleziono runów pasujących do KEYWORDS. "
        "Sprawdź ENTITY, PROJECT, KEYWORDS oraz ONLY_FINISHED."
    )

result_df = pd.concat(all_run_histories, ignore_index=True)

# Przydatne uporządkowanie pierwszych kolumn
priority_columns = [
    "run_name",
    "run_id",
    "run_group",
    "run_state",
    "_step",
    "_timestamp",
    "_runtime",
]

existing_priority_columns = [
    column for column in priority_columns
    if column in result_df.columns
]

other_columns = [
    column for column in result_df.columns
    if column not in existing_priority_columns
]

result_df = result_df[existing_priority_columns + other_columns]
result_df.to_csv(OUTPUT_FILE, index=False)

print("\nEksport zakończony.")
print(f"Liczba runów: {len(selected_runs)}")
print(f"Liczba wszystkich wierszy: {len(result_df)}")
print(f"Plik: {OUTPUT_FILE}")

print("\nWybrane runy:")
for selected_run in selected_runs:
    print(
        f"- {selected_run['run_name']} "
        f"(id={selected_run['run_id']}, rows={selected_run['rows']})"
    )
# %%
