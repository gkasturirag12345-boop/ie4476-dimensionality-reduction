# Dimensionality Reduction and Its Effect on Classification Performance

**Course Project: Machine Learning Fundamentals**
**Date:** November 2026

## Overview

This project investigates how Principal Component Analysis (PCA) dimensionality reduction affects classification performance using the MNIST handwritten digit dataset and K-Nearest Neighbors (KNN) classifier (K=5).

## Project Structure

```
Dimensionality_Reduction_Project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── main.py                             # Main experiment implementation
├── generate_report.py                  # Word report generation script
├── data/
│   └── mnist.npz                       # MNIST dataset (downloaded)
├── results/
│   └── experimental_results.csv        # Complete experimental data
├── figures/
│   ├── accuracy_vs_dimensions.png      # Accuracy vs PCA dimensions
│   ├── error_rate_vs_dimensions.png    # Error rate vs PCA dimensions
│   ├── explained_variance.png          # Cumulative explained variance
│   └── computational_time.png          # Training/testing time comparison
└── report/
    └── Dimensionality_Reduction_Report.docx  # Academic report (4-5 pages)
```

## Experimental Design

### Dataset
- **MNIST**: 60,000 training images, 10,000 test images
- **Original dimensionality**: 784 (28×28 pixels)
- **Classes**: 10 digits (0-9)

### Methodology
1. **Data Preprocessing**: Normalized pixel values to [0, 1] range
2. **PCA Application**: Fitted on training data only (prevents data leakage)
3. **Dimensions Tested**: 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, and baseline (784)
4. **Classification**: K-Nearest Neighbors (K=5, Euclidean distance)
5. **Evaluation**: Accuracy, error rate, explained variance, computational time

### Key Findings

**Best Performance (30 components):**
- Accuracy: **97.55%** (better than baseline!)
- Explained Variance: 73.05%
- Testing speedup: **12.6× faster** than baseline
- Dimensionality reduction: 96.2%

**Optimal Balance (10 components):**
- Accuracy: 92.76% (95.7% of baseline performance)
- Explained Variance: 48.8%
- Dimensionality reduction: 98.7%
- Testing speedup: **8.5× faster**

**Baseline (784 dimensions):**
- Accuracy: 96.88%
- Testing time: 3.49s

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - numpy>=1.21.0
   - pandas>=1.3.0
   - scikit-learn>=1.0.0
   - matplotlib>=3.4.0
   - seaborn>=0.11.0
   - python-docx>=0.8.11
   - tensorflow>=2.10.0

## Running the Experiment

### Full Experiment (from scratch)

```bash
python3 main.py
```

This will:
1. Load MNIST dataset from `data/mnist.npz`
2. Run 14 experiments (13 PCA dimensions + baseline)
3. Generate results CSV and 4 visualization figures
4. Take approximately 5-10 minutes to complete

**Expected output:**
- `results/experimental_results.csv`
- `figures/accuracy_vs_dimensions.png`
- `figures/error_rate_vs_dimensions.png`
- `figures/explained_variance.png`
- `figures/computational_time.png`

### Generate Report

```bash
python3 generate_report.py
```

This creates:
- `report/Dimensionality_Reduction_Report.docx` (4-5 page academic report)

## Results Summary

| Dimensions | Explained Variance | Accuracy | Error Rate | Test Time |
|------------|-------------------|----------|------------|-----------|
| 5          | 33.2%             | 74.73%   | 25.27%     | 0.07s     |
| 10         | 48.8%             | 92.76%   | 7.24%      | 0.41s     |
| 20         | 64.4%             | 96.94%   | 3.06%      | 0.38s     |
| **30**     | **73.1%**         | **97.55%** | **2.45%** | **0.28s** |
| 50         | 82.5%             | 97.48%   | 2.52%      | 0.36s     |
| 100        | 91.5%             | 97.27%   | 2.73%      | 0.54s     |
| 784 (baseline) | 100.0%        | 96.88%   | 3.12%      | 3.49s     |

## Key Observations

1. **Performance Improvement**: PCA with 30 components achieves **higher accuracy** (97.55%) than the baseline (96.88%), demonstrating that dimensionality reduction can improve generalization by removing noise.

2. **Computational Efficiency**: The 30-component model is **12.6× faster** during inference, making it ideal for real-time applications.

3. **Diminishing Returns**: Beyond 30 dimensions, accuracy plateaus while computational cost continues to increase.

4. **Sweet Spot**: 10-30 components provide the optimal balance between performance, speed, and complexity.

## Reproducing Results

To reproduce the exact results in the report:

1. Ensure MNIST dataset is present in `data/mnist.npz`
2. Run: `python3 main.py`
3. Results will match reported values (±0.1% due to randomness in KNN distance calculations)

**Note**: The random_state is set to 42 for PCA to ensure reproducibility.

## File Descriptions

### Source Code
- **main.py**: Complete experimental pipeline
  - Data loading and preprocessing
  - PCA transformation (fitted on training data only)
  - KNN classifier training and evaluation
  - Results collection and visualization
  - ~500 lines, fully documented

- **generate_report.py**: Automated report generation
  - Loads experimental results from CSV
  - Creates formatted Word document with tables and figures
  - Academic writing style with 5 sections
  - ~350 lines

### Data Files
- **data/mnist.npz**: MNIST dataset in NumPy compressed format
- **results/experimental_results.csv**: Complete experimental data (14 rows × 6 columns)

### Visualizations
All figures are 300 DPI PNG format suitable for publication:
1. **accuracy_vs_dimensions.png**: Shows classification accuracy vs. number of PCA components
2. **error_rate_vs_dimensions.png**: Shows error rate (1 - accuracy) vs. components
3. **explained_variance.png**: Cumulative explained variance by number of components
4. **computational_time.png**: Training and testing time comparison across dimensions

### Report
- **Dimensionality_Reduction_Report.docx**: 4-5 page academic report including:
  - Abstract
  - Introduction (motivation, curse of dimensionality)
  - Methodology (PCA, KNN, experimental setup)
  - Results (tables, figures, analysis)
  - Discussion (optimal dimensionality, trade-offs)
  - Conclusion
  - References

## Technical Details

### Data Leakage Prevention
PCA is **fitted only on training data**:
```python
pca = PCA(n_components=n)
X_train_pca = pca.fit_transform(X_train)  # Fit on training
X_test_pca = pca.transform(X_test)        # Transform test only
```

### Performance Metrics
- **Accuracy**: Percentage of correctly classified samples
- **Error Rate**: 1 - Accuracy
- **Explained Variance**: Proportion of dataset variance retained
- **Training Time**: Time to fit KNN model
- **Testing Time**: Time to classify all 10,000 test samples

## Limitations and Future Work

1. **Single Classifier**: Only KNN tested; future work could compare with SVM, Random Forest, Neural Networks
2. **Single Dataset**: Only MNIST; could extend to CIFAR-10, Fashion-MNIST
3. **PCA Only**: Could compare with other methods (LDA, t-SNE, UMAP, Autoencoders)
4. **Hyperparameters**: K=5 fixed; could optimize with cross-validation

## References

1. van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
2. LeCun, Y., Cortes, C., & Burges, C. J. (2010). MNIST handwritten digit database. *ATT Labs*.
3. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.

## Author

Final Year Project
November 2026

## License

This project is for educational purposes only.


---
## About this repository
Coursework: PCA, LDA and PCA→LDA on MNIST with kNN (k chosen by cross-validation) and a minimum-Mahalanobis classifier; v1 (original) and v2 (training-only selection, McNemar tests).

**Left out of this repository:** MNIST data (data/mnist.npz) and the written reports.

*Private draft repository; under review before any public release.*
