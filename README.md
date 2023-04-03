# Analysis of the GSSP

This repository contains the analysis code for the Ghent Semi-spontaneous Speech Paradigm (GSSP).
The `GSSP` is a picture description task that is used to capture (near) spontaneous speech in a controlled setting.

- The data is collected via a [web application](https://github.com/predict-idlab/gssp_web_app) and can be found on [kaggle](https://www.kaggle.com/datasets/jonvdrdo/gssp-web-app-data)
- The paradigm is described in detail in [this preprint manuscript](https://psyarxiv.com/e2qxw).
- The [notebooks](notebooks/README.md) README contains a thorough description of the speech parsing and analysis notebooks.


In a nutshell the [r-scripts](scripts/) folder performs a thorough statistical analysis of the arousal & valence scores for the audio files.
The outcome can be observed in a [shiny html file](scripts/1.2_FactorAnalysis.pdf).
All speech data transformation and analysis is performed in the [notebooks](notebooks/README.md) folder.

The utilized python packages are listed in the [pyproject.toml](pyproject.toml) file and the utilized R packages are listed in the [scripts/r_packages.txt](scripts/r_packages.txt) file.

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

- A preprint manuscript is available on [psyArxiv](https://psyarxiv.com/e2qxw/).

```bibtex
@misc{van_der_donckt_2023,
 title={Ecologically Valid Speech Collection in Behavioral Research: The Ghent Semi-spontaneous Speech Paradigm (GSSP)},
 url={psyarxiv.com/e2qxw},
 DOI={10.31234/osf.io/e2qxw},
 publisher={PsyArXiv},
 author={Van Der Donckt, Jonas and Kappen, Mitchel and Degraeve, Vic and Demuynck, Kris and Vanderhasselt, Marie Anne and Van Hoecke, Sofie},
 year={2023},
 month={Mar}
}
```

---

<p align="center">
👤 <i>Jonas Van Der Donckt, Mitchel Kappen</i>
</p>
