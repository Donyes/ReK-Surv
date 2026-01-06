# Data Directory

This directory contains the datasets used for training and evaluation.

## Datasets Overview

| Dataset | Domain | Samples | Features | Event Rate | Format |
|---------|--------|---------|----------|------------|--------|
| METABRIC | Breast Cancer | 1,904 | 9 | 57.9% | HDF5 |
| SUPPORT | ICU Mortality | 8,873 | 14 | 68.0% | HDF5 |
| WHAS | Heart Attack | 1,638 | 6 | 42.1% | HDF5 |
| RGBSG | Breast Cancer | 2,232 | 7 | 43.2% | HDF5 |
| HLB | Citrus Disease | 771 | 18 | 2.6% | Excel |

## METABRIC Dataset

The METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset.

- **File**: `metabric.h5`
- **Endpoint**: Overall survival
- **Source**: [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)

## SUPPORT Dataset

The SUPPORT (Study to Understand Prognoses Preferences Outcomes and Risks of Treatment) dataset.

- **File**: `support.h5`
- **Endpoint**: Time to death
- **Source**: [Vanderbilt Biostatistics](http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc)

## WHAS Dataset

The WHAS (Worcester Heart Attack Study) dataset.

- **File**: `whas.h5`
- **Endpoint**: Time to death after heart attack
- **Source**: [Applied Survival Analysis (Hosmer & Lemeshow)](https://www.umass.edu/statdata/statdata/data/whas500.txt)

## RGBSG Dataset

The RGBSG (Rotterdam & German Breast Cancer Study Group) dataset.

- **File**: `rgbsg.h5`
- **Endpoint**: Recurrence-free survival
- **Source**: [R survival package](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/rotterdam.html)

## HLB Dataset

The HLB (Huanglongbing/Citrus Greening Disease) dataset for agricultural survival analysis.

- **File**: `hlb.xlsx`
- **Endpoint**: Time to infection
- **Features**: Environmental and tree characteristics

## HDF5 File Structure

All HDF5 files follow the same structure:
```
dataset.h5
├── train/
│   ├── x    # Features (n_train, n_features)
│   ├── t    # Survival times (n_train,)
│   └── e    # Event indicators (n_train,)
└── test/
    ├── x    # Features (n_test, n_features)
    ├── t    # Survival times (n_test,)
    └── e    # Event indicators (n_test,)
```

## Usage

```python
from utils import load_metabric, load_hlb, load_h5_dataset

# Load METABRIC
X_train, X_test, T_train, T_test, E_train, E_test = load_metabric('data/metabric.h5')

# Load other HDF5 datasets (SUPPORT, WHAS, RGBSG)
X_train, X_test, T_train, T_test, E_train, E_test = load_h5_dataset('data/support.h5')

# Load HLB (Excel format)
X_train, X_test, T_train, T_test, E_train, E_test = load_hlb('data/hlb.xlsx')
```
