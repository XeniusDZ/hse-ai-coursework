# AI & Data Science Coursework — HSE

A collection of my completed homework assignments from three courses at the **Higher School of Economics (HSE)**, covering the full path from NumPy fundamentals to generative models, object detection, and transformers.

> **Author:** Yanis Boudiaf
> Notebook text is in Russian (the language of instruction); code, comments and results are language-agnostic.

| Part | Course | Notebooks | Main stack |
|------|--------|-----------|------------|
| [`machine_learning/`](#1-machine-learning) | Intellectual Data Analysis (ML foundations) | 8 | NumPy, pandas, scikit-learn |
| [`deep_learning/`](#2-deep-learning) | Fundamentals of Deep Learning | 4 | PyTorch, Lightning, torchvision, YOLOv8, Transformers |
| [`applied-ds/`](#3-applied-data-science) | Applied Data Science Problems | 3 | PyTorch, torchaudio, ETNA |

---

## 1. Machine Learning

`machine_learning/homeworks/` — the classical ML track, going from library basics to ensembles. Each homework ships with the dataset it uses.

| # | Notebook | Topic | What's inside |
|---|----------|-------|---------------|
| 01 | `hw01-numpy/hw01-numpy.ipynb` | **NumPy** | 10 vectorisation exercises: array manipulation, broadcasting, masks, no-loop implementations. |
| 02 | `hw02-pandas/hw02_pandas.ipynb` | **pandas** | Merging 20 fragmented CSVs (student info + marks) into one table, descriptive stats, missing values, group-bys, score → letter-grade conversion. |
| 03 | `hw03-plots/hw03-EDA.ipynb` | **EDA & visualisation** | Exploratory analysis of the *German Credit* dataset with matplotlib/seaborn: distributions, correlations, categorical breakdowns, conclusions. |
| 04 | `hw04-knn-linreg/hw04_knn_linreg.ipynb` | **kNN & linear models** | Decision-surface visualisation, a **from-scratch kNN classifier**, linear regression with feature engineering. |
| 05 | `hw05-grad/hw05_gd.ipynb` | **Gradient descent** | Hand-written GD / SGD / momentum / Adam optimisers behind a common abstract interface, then linear regression trained with them on the *UCI Automobile* dataset (+ bonus). |
| 06 | `hw06-texts/hw06_texts.ipynb` | **Text classification** | *Disaster Tweets*: cleaning, custom NLTK tokeniser/stemmer plugged into `CountVectorizer`, TF-IDF, hashing trick, logistic regression — target F1 ≥ 0.75. |
| 07 | `hw07-trees/hw07-trees-rf.ipynb` | **Decision trees & random forest** | Impurity criteria by hand, **decision tree implemented from scratch**, bagging and random forest on the *Mushroom*, *Diabetes* and *User Knowledge* datasets. |
| 08 | `hw08-boosting/homework.ipynb` | **Gradient boosting** | Boosting models on the *Data Science Salaries* dataset: preprocessing, tuning, comparison with earlier models. |

---

## 2. Deep Learning

`deep_learning/` — the "Fundamentals of Deep Learning" minor. Four larger projects, each with an experiment report at the end of the notebook.

| # | Notebook | Topic | What's inside |
|---|----------|-------|---------------|
| 1 | `homework-1.ipynb` | **Intro to PyTorch, MLPs** | Linear regression with a manual training loop (no built-in optimisers/losses, only `autograd`), then a fully-connected network on a regression dataset with a target RMSE ≤ 8.75. |
| 2 | `homework-2.ipynb` | **Image classification** | Two CNN experiments in PyTorch Lightning: (a) a network trained **from scratch** without pretrained weights or resizing (target acc ≥ 0.44), (b) a **fine-tuned pretrained** model (target acc ≥ 0.84). Augmentations, optimiser/scheduler ablations, written report. |
| 3 | `homework-3.ipynb` | **Object detection** | Parsing Pascal-VOC XML annotations, building a detection architecture and its training loop **from scratch**, then converting the data to YOLO format and training **YOLOv8 (ultralytics)**; albumentations for augmentation. |
| 4 | `homework-4.ipynb` | **NLP with transformers** | News pipeline in 4 parts: (1) train a news **category classifier** (baseline → improved architectures), (2) predict categories for unlabelled news, (3) run a pretrained **Hugging Face sentiment model** on comments, (4) analytics — most positive/negative categories and articles. |

---

## 3. Applied Data Science

`applied-ds/` — "Applied Data Science Problems": advanced, domain-specific deep learning.

| # | Notebook | Topic | What's inside |
|---|----------|-------|---------------|
| 1 | `homework-1/hw-1.ipynb` | **Generative models** | Simulating gamma-telescope (MAGIC) particle images: a **Conditional WGAN** built and trained in PyTorch, generation quality measured with a classifier ROC-AUC, then a **diffusion model** with a noise scheduler. |
| 2 | `homework-2/hw_2.ipynb` | **Audio deep learning** | Audio classification from raw waveform with an **RNN**, log-mel spectrograms with torchaudio, RNN and **CNN** on spectrograms, **SpecAugment** augmentation. |
| 3 | `homework-3/hw_3.ipynb` | **Time-series forecasting** | Using the **ETNA** library: EDA, missing values & outliers, **Prophet**, hierarchical time series, feature engineering, forecasting strategies, ensembles, and a bonus transformer model. |

---

## Repository layout

```
.
├── machine_learning/
│   └── homeworks/
│       ├── hw01-numpy/          hw01-numpy.ipynb
│       ├── hw02-pandas/         hw02_pandas.ipynb, Data/*.csv
│       ├── hw03-plots/          hw03-EDA.ipynb, german_credit.csv
│       ├── hw04-knn-linreg/     hw04_knn_linreg.ipynb
│       ├── hw05-grad/           hw05_gd.ipynb, data.csv
│       ├── hw06-texts/          hw06_texts.ipynb, train.csv
│       ├── hw07-trees/          hw07-trees-rf.ipynb, agaricus-lepiota.data, diabetes.csv, students.csv
│       └── hw08-boosting/       homework.ipynb, ds_salaries.csv
├── deep_learning/
│   ├── homework-1.ipynb         PyTorch basics, MLP
│   ├── homework-2.ipynb         CNN image classification
│   ├── homework-3.ipynb         Object detection, YOLOv8
│   └── homework-4.ipynb         Transformers, sentiment analysis
└── applied-ds/
    ├── homework-1/              hw-1.ipynb + img/ (Conditional WGAN, diffusion)
    ├── homework-2/              hw_2.ipynb (audio)
    └── homework-3/              hw_3.ipynb (time series, ETNA)
```

## Running the notebooks

Small datasets are included next to their notebooks. Larger ones (images, audio, news) are downloaded inside the notebooks themselves (via `gdown` / Google Drive links provided by the course), so the deep-learning and applied-DS notebooks are best run in **Google Colab** with a GPU.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk mlxtend \
            torch torchvision torchaudio torchmetrics pytorch-lightning \
            transformers ultralytics albumentations opencv-python etna gdown
```

## Skills demonstrated

- Implementing core algorithms from scratch: kNN, gradient-descent optimisers, decision trees, a detection model, a conditional WGAN
- Classical ML pipeline: EDA → feature engineering → model selection → evaluation (scikit-learn)
- Deep learning with PyTorch & Lightning: CNNs, RNNs, transfer learning, augmentation, experiment reporting
- Computer vision (classification, detection with YOLOv8), audio (mel spectrograms, SpecAugment), NLP (TF-IDF, transformers, Hugging Face), generative models (GAN, diffusion), time series (ETNA, Prophet)
