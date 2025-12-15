# Goods vs Services Classification (Multilingual)

This repository contains the MSc Business Data Science (AAU) 1st-semester project on automatically classifying short business-related texts as **goods** or **services**, and predicting **macro-categories**. The project compares lexical baselines (TF-IDF) with sentence embedding approaches (SBERT and SetFit) and emphasizes **external validation** using an EUIPO-like dataset.

## Data

The main dataset is constructed by integrating multiple public sources (e-commerce product texts and service reviews) and harmonizing them into a common schema with:

- `name`
- `goods_or_services`
- `category`
- `about_product`
- `review`

> Note: If full datasets cannot be shared, this repository provides scripts to rebuild the final dataset structure or to reproduce the experiments using an anonymized sample.

## Methods (high level)

1. **Preprocessing**
   - build a unified text field from `name`, `about_product`, and `review`
   - clean text (lowercasing, removing URLs/emails, keeping letters and spaces)

2. **Binary classification (goods vs services)**
   - TF-IDF + Logistic Regression (baseline)
   - TF-IDF + Logistic Regression (keywords removed robustness check)
   - SBERT embeddings + Logistic Regression
   - SBERT embeddings + Logistic Regression (augmented service training set)
   - SetFit (few-shot fine-tuning)

3. **Multi-class classification (macro categories)**
   - category prediction using the same text pipeline

4. **Evaluation**
   - internal validation (train/val/test split with stratification)
   - external validation (EUIPO-like hand-labeled set)
   - metrics: accuracy, macro-F1, weighted-F1, confusion matrices

## Results (key external validation table)

External validation (EUIPO-like):

| model | accuracy | f1_macro | f1_weighted |
|---|---:|---:|---:|
| TF-IDF + LogReg | 0.473684 | 0.321429 | 0.304511 |
| SBERT + LogReg (base) | 0.526316 | 0.424242 | 0.411483 |
| SBERT + LogReg (augmented) | 0.736842 | 0.724638 | 0.721587 |
| SetFit style (SBERT + LogReg) | 0.526316 | 0.424242 | 0.411483 |
| SetFit (real, Trainer, few-shot) | 0.789474 | 0.784091 | 0.782297 |

## Run the experiments

The data processing and model training pipeline is implemented using Jupyter notebooks.

Run the notebooks in the following order:

1. `01_build_dataset.ipynb`
2. `02_goods_services_classification_pipeline.ipynb`

## Run the Streamlit app

```bash
streamlit run 03_main.py```


## Ethics and governance

No personal data is intentionally collected.
All data sources are public datasets.
Only data strictly necessary for reproducibility is stored in the repository.
When redistribution is not possible, the repository provides code to reconstruct the pipeline.

## Authors

Alvaro Buendía
Ioannis Chatzikos

## Supervisor

Milad Abbasiharofteh
Aalborg University Business School











