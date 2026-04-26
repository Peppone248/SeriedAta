# Serie dAta

This project has the aim to cover all the ETL and Data engineering tasks, handling a dataset based on Serie A teams and
matches statistics between 2020-2025.

## Structure

```text
seriedAta/
├── README.md
├── main.py
├── __init__.py
├── aggregations.py
├── cleaning.py
├── config.py
├── features.py
├── pipeline.py
├── statistics.py
└── validation.py
```

## What each file does

- `config.py`: shared constants such as column lists.
- `cleaning.py`: schema cleanup and dtype conversion.
- `features.py`: engineered football features.
- `statistics.py`: descriptive statistics outputs.
- `aggregations.py`: grouped analysis tables.
- `validation.py`: data quality and consistency checks.
- `pipeline.py`: runs the full workflow and returns all output tables.
- `main.py`: entry point you can run as a script.
- `visualization.py`: include all methods used to plot useful statistics coming from data.

## Run

```bash
python main.py --csv-path matches_seriea.csv
```

If the CSV is elsewhere:

```bash
python main.py --csv-path /full/path/to/matches_seriea.csv
```

## Design notes

- The pipeline loads the CSV once.
- Cleaning and feature functions mutate the working DataFrame and return it.
- Aggregations are split into small reusable functions.
- Validation checks are collected in one place.
- The outputs are returned in a dictionary, so you can reuse them in notebooks.
