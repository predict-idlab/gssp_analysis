# Analysis of the GSSP

This repository contains the analysis code for the Ghent Semi-spontaneous Speech Paradigm (GSSP).
The `GSSP` is a picture description task that is used to capture (near) spontaneous speech in a controlled setting.

- The data is collected via a [web application](https://github.com/predict-idlab/gssp_web_app) and can be found on [kaggle](https://www.kaggle.com/datasets/jonvdrdo/gssp-web-app-data)
- The paradigm is described in detail in [this preprint manuscript](TODO).
- The [notebooks](notebooks/README.md) README contains a thorough description of the speech parsing and analysis notebooks.


In a nutshell the [r-scripts](scripts/) folder performs a thorough statistical analysis of the arousal & valence scores for the audio files.
The outcome can be observed in a [shiny html file](scripts/1.2_FactorAnalysis.html).
All speech data transformation and analysis is performed in the [notebooks](notebooks/README.md) folder.


---
## Folder structure

```txt
├── docs
│   └── cgn             <-- CGN related documentation
├── GSSP_utils          <-- Python functions shared across notebooks (and CGN parsing)
├── loc_data            <-- Local data shared across notebooks
├── notebooks           <-- the analysis Jupyter notebooks
├── reports             <-- Generated figures from the notebooks
└── scripts             <-- R scripts for statistical analysis & shiny app
```
## Cite

- A preprint manuscript is available on [psyArxiv](TODO).

```bibtex

```

---

<p align="center">
👤 <i>Jonas Van Der Donckt, Mitchel Kappen</i>
</p>
