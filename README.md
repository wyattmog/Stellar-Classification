# Stellar Classification Project

A comprehensive machine learning project for classifying celestial objects (stars, galaxies, and quasars) using photometric data from astronomical surveys.

## Overview

This project implements and compares multiple machine learning algorithms to classify stellar objects into three categories:
- **GALAXY** (0): Extended objects composed of many stars
- **QSO** (1): Quasi-stellar objects (quasars)
- **STAR** (2): Point-like stellar objects

The classification is performed using photometric features including magnitude measurements in different filters (u, g, r, i, z) and redshift data.

## Dataset

The project uses a local stellar classification dataset (`stellar_classification.csv`) containing photometric measurements with the following key features:
- **u, g, r, i, z**: Photometric magnitudes in different filters
- **redshift**: Cosmological redshift measurement
- **class**: Target variable (GALAXY, QSO, STAR)

The dataset is included in the repository for easy access and reproducibility.

### Data Preprocessing
- Removal of unnecessary attributes (object IDs, coordinates, etc.)
- Handling of extreme outliers in photometric measurements
- Class encoding (GALAXY: 0, QSO: 1, STAR: 2)
- SMOTE resampling to address class imbalance

## Machine Learning Models

The project implements and compares five different algorithms:

1. **Logistic Regression**
   - Hyperparameters: C=10, penalty='l2'
   - Linear classification approach

2. **K-Nearest Neighbors (KNN)**
   - Optimal neighbors: n_neighbors=1
   - Instance-based learning

3. **XGBoost**
   - Hyperparameters: learning_rate=0.1, max_depth=5, n_estimators=200
   - Gradient boosting ensemble method

4. **Naive Bayes**
   - Gaussian Naive Bayes implementation
   - Probabilistic classification

5. **Random Forest**
   - Hyperparameters: max_depth=None, min_samples_split=2, n_estimators=100
   - Ensemble decision tree method

## Key Features

### Data Analysis
- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Statistical measures (mean, median, mode, variance, etc.)
- ✅ Visualization of data distributions and relationships
- ✅ ECDF vs CDF analysis for distribution comparison

### Model Optimization
- ✅ GridSearchCV for hyperparameter tuning
- ✅ 5-fold cross-validation
- ✅ SMOTE for handling class imbalance
- ✅ Learning curve analysis

### Evaluation Metrics
- ✅ Accuracy and Recall scores
- ✅ ROC-AUC analysis
- ✅ Confusion matrices with heatmap visualization
- ✅ Detailed classification reports
- ✅ Model performance comparison

## Project Structure

```
Stellar-Classification/
├── README.md
├── Stellar_Classification_Report.ipynb    # Main analysis notebook
├── requirements.txt                       # Python dependencies
└── stellar_classification.csv             # Dataset file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wyattmog/Stellar-Classification.git
cd Stellar-Classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Open and run the Jupyter notebook:
```bash
jupyter notebook Stellar_Classification_Report.ipynb
```

The notebook contains:
- Local data loading from CSV file
- Data preprocessing steps
- Exploratory data analysis with visualizations
- Model training and hyperparameter optimization
- Performance evaluation and comparison
- Learning curve analysis

## Results

The project provides comprehensive model comparison including:
- Performance metrics for all five algorithms
- Confusion matrices for each model
- ROC curves and AUC scores
- Learning curves showing training vs. validation performance
- Visual comparisons of accuracy and recall scores

## Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and metrics
- **XGBoost** - Gradient boosting framework
- **Seaborn & Matplotlib** - Data visualization
- **Imbalanced-learn** - SMOTE for handling class imbalance
- **Statsmodels** - Statistical analysis

## Future Improvements

- [ ] Feature engineering with astronomical domain knowledge
- [ ] Deep learning approaches (neural networks)
- [ ] Ensemble methods combining multiple models
- [ ] Cross-validation with astronomical survey-specific splits
- [ ] Integration with additional astronomical catalogs

## License

This project is available under the MIT License. See LICENSE file for details.

## Acknowledgments

- Astronomical survey data providers
- Scikit-learn and XGBoost communities
- Open source astronomy tools and libraries