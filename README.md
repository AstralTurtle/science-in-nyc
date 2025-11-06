# Science in NYC Final

## Overview

This repository compares GitHub repository metadata (from the
`ibragim-bad/github-repos-metadata-40M` dataset) with developer preferences from the Stack Overflow Developer Survey using pandas analysis.


## Package manager / how to run

This project uses `uv` for running the project. You can run the main script with:

```bash
uv run main.py
```

## Datasets (required)

This repository uses two main datasets.

1) GitHub repository metadata
   - Source: `ibragim-bad/github-repos-metadata-40M` (via the Hugging Face
     `datasets` loader in `main.py`). The dataset is loaded in `main.py` with
     `load_dataset("ibragim-bad/github-repos-metadata-40M", split=...)`.

2) Stack Overflow Developer Survey 
   - Download a single year's CSV from the public Stack Overflow Developer Survey downloads page.
   - Save the CSV file in the repository root (or anywhere you prefer) and point the project to it using the `.env` file 

