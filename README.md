# AF-Analysis

Streamlit UI for browsing AlphaFold, AlphaFold-Multimer, AlphaFold3, and AlphaPulldown prediction folders.

AF-Analysis loads the local dataframe cache when available, merges the interface-analysis CSV (`predictions_with_pae_cutoff_*.csv`) when present, and can run the bundled interaction-report module plus AlphaJudge backfill when the CSV is missing.

## Install

Recommended install from the unpacked folder:

```bash
bash install_af_analysis.sh
```

This installs AF-Analysis and its Python dependencies, including AlphaJudge from the KosinskiLab GitHub repository. It also checks whether the `alphajudge` command is available after installation.

Manual equivalent:

```bash
pip install -e .
```

The dependency list is also available in `requirements.txt`.

## Run

```bash
af-analysis --directory /path/to/predictions --port 8502
```

Or from the unpacked folder:

```bash
bash run_af_analysis.sh -- --directory /path/to/predictions
```

## Notes

- Overview plots are generated only when requested.
- Clicking a Plotly point opens that prediction in the structure viewer.
- The reporting entry point is now the bundled `af_analysis.interaction_report` module.
- Existing dataframe caches are read from the previous location as a backwards-compatible fallback; new caches are written to `.af_analysis_cache`.

## Attribution and contact

AF-Analysis is an MDC-maintained analysis viewer modified and maintained by **Ashwin Karthick Natarajan**.

Queries: ashwin.natarajan@mdc-berlin.de

Built from and extended around these open-source projects:

- APLit: https://github.com/KosinskiLab/aplit
- AlphaJudge: https://github.com/KosinskiLab/AlphaJudge
- af_analysis: https://github.com/samuelmurail/af_analysis

