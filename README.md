# BioInsight

BioInsight is a lightweight pipeline for exploring bioactivity data, training simple predictive models, and inspecting results through a Streamlit UI. It's intended for rapid experimentation and explainability work on small bioactivity datasets.

## Features
- Data ingestion and optional local sample DB creation
- Exploratory data analysis (EDA) pipeline
- Model training and evaluation utilities
- Explainability helpers for interpreting model outputs
- Streamlit app for quick results inspection

## Repo Structure
- `bioactivity_data.csv` — raw dataset (expected at repo root)
- `db_setup.py`, `create_sample_db.py` — helpers to create a local sample DB
- `eda_pipeline.py` — exploratory data analysis scripts
- `ml_models.py` — training and evaluation scripts
- `explainability.py` — explainability utilities and analysis
- `streamlit_app.py` — Streamlit app to view results interactively
- `requirements.txt` — Python dependencies
- `models/` — saved model artifacts and metadata (e.g. `feature_names.json`, `results_summary.json`)

## Requirements
- Python 3.8+
- Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

- Run the EDA pipeline:

```bash
python eda_pipeline.py
```

- Train or evaluate models:

```bash
python ml_models.py
```

- Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Usage Notes
- The scripts expect `bioactivity_data.csv` in the repository root unless paths are adjusted.
- Model artifacts and summaries are stored in the `models/` folder.
- Use `db_setup.py` and `create_sample_db.py` to create a small local database for experiments.

## Contributing
- Fork the repo, create a descriptive feature branch, add tests or examples, and open a PR.
- Update `requirements.txt` when adding dependencies.

## License
Add a `LICENSE` file if you wish to open-source the project. No license is included by default.

## Contact
For questions or collaboration, open an issue in this repository or contact the owner.
