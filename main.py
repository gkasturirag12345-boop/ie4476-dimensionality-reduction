"""
Dimensionality Reduction and Its Effect on Classification Performance
Using Principal Component Analysis (PCA) and K-Nearest Neighbors (KNN) on MNIST Dataset

This script investigates how PCA dimensionality reduction affects KNN classification performance.
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Use TensorFlow/Keras for MNIST (more reliable than OpenML)

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DIMENSIONS_TO_TEST = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]
K_NEIGHBORS = 5
TRAIN_SIZE = 60000
TEST_SIZE = 10000

print("="*80)
print("DIMENSIONALITY REDUCTION AND CLASSIFICATION PERFORMANCE EXPERIMENT")
print("="*80)
print(f"\nConfiguration:")
print(f"  - Dataset: MNIST Handwritten Digits")
print(f"  - Training samples: {TRAIN_SIZE}")
print(f"  - Testing samples: {TEST_SIZE}")
print(f"  - Classifier: K-Nearest Neighbors (K={K_NEIGHBORS})")
print(f"  - Dimensions to test: {DIMENSIONS_TO_TEST}")
print(f"  - Original dimensionality: 784")
print("="*80)


def load_mnist_dataset():
    """
    Load MNIST dataset and split into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test: Training and testing data
    """
    print("\n[1/8] Loading MNIST dataset...")

    # Load MNIST from local file (downloaded manually to avoid SSL issues)
    data = np.load('data/mnist.npz')
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']

    # Flatten images from 28x28 to 784-dimensional vectors
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    # Convert labels to integers
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    print(f"  ✓ Training set: {X_train.shape}")
    print(f"  ✓ Testing set: {X_test.shape}")
    print(f"  ✓ Number of classes: {len(np.unique(y_train))}")

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """
    Preprocess data: normalize to [0, 1] range.

    Args:
        X_train: Training features
        X_test: Testing features

    Returns:
        Normalized training and testing features
    """
    print("\n[2/8] Preprocessing data...")

    # Normalize pixel values to [0, 1]
    X_train_normalized = X_train / 255.0
    X_test_normalized = X_test / 255.0

    print(f"  ✓ Normalized to range [0, 1]")
    print(f"  ✓ Training data range: [{X_train_normalized.min():.2f}, {X_train_normalized.max():.2f}]")
    print(f"  ✓ Testing data range: [{X_test_normalized.min():.2f}, {X_test_normalized.max():.2f}]")

    return X_train_normalized, X_test_normalized


def apply_pca(X_train, X_test, n_components):
    """
    Apply PCA dimensionality reduction.
    IMPORTANT: PCA is fitted ONLY on training data to prevent data leakage.

    Args:
        X_train: Training features
        X_test: Testing features
        n_components: Number of principal components to retain

    Returns:
        Reduced training and testing features, fitted PCA model
    """
    pca = PCA(n_components=n_components, random_state=42)

    # Fit PCA on training data ONLY
    X_train_pca = pca.fit_transform(X_train)

    # Transform testing data using the same fitted PCA
    X_test_pca = pca.transform(X_test)

    # Calculate cumulative explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100

    return X_train_pca, X_test_pca, pca, explained_variance


def train_knn_classifier(X_train, y_train, k=K_NEIGHBORS):
    """
    Train K-Nearest Neighbors classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        k: Number of neighbors

    Returns:
        Trained KNN model, training time
    """
    start_time = time.time()

    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', algorithm='auto')
    knn.fit(X_train, y_train)

    training_time = time.time() - start_time

    return knn, training_time


def evaluate_classifier(knn, X_test, y_test):
    """
    Evaluate classifier performance.

    Args:
        knn: Trained KNN model
        X_test: Testing features
        y_test: True testing labels

    Returns:
        Accuracy, error rate, prediction time
    """
    start_time = time.time()

    y_pred = knn.predict(X_test)

    prediction_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred) * 100
    error_rate = 100 - accuracy

    return accuracy, error_rate, prediction_time


def run_experiment(X_train, X_test, y_train, y_test, n_components, experiment_num, total_experiments):
    """
    Run a single PCA + KNN experiment for a given dimensionality.

    Args:
        X_train, X_test, y_train, y_test: Dataset
        n_components: Number of PCA components (or None for baseline)
        experiment_num: Current experiment number
        total_experiments: Total number of experiments

    Returns:
        Dictionary containing experimental results
    """
    if n_components is None:
        # Baseline experiment (no PCA)
        print(f"\n[Experiment {experiment_num}/{total_experiments}] Original Dimensionality (784)")
        print("-" * 60)

        X_train_reduced = X_train
        X_test_reduced = X_test
        explained_variance = 100.0
        dim = 784

    else:
        # PCA experiment
        print(f"\n[Experiment {experiment_num}/{total_experiments}] PCA with {n_components} components")
        print("-" * 60)

        X_train_reduced, X_test_reduced, pca, explained_variance = apply_pca(
            X_train, X_test, n_components
        )
        dim = n_components

    # Train KNN classifier
    print(f"  Training KNN (K={K_NEIGHBORS})...")
    knn, train_time = train_knn_classifier(X_train_reduced, y_train)

    # Evaluate classifier
    print(f"  Evaluating on test set...")
    accuracy, error_rate, test_time = evaluate_classifier(knn, X_test_reduced, y_test)

    # Print results
    print(f"  ✓ Dimensionality: {dim}")
    print(f"  ✓ Explained Variance: {explained_variance:.2f}%")
    print(f"  ✓ Accuracy: {accuracy:.2f}%")
    print(f"  ✓ Error Rate: {error_rate:.2f}%")
    print(f"  ✓ Training Time: {train_time:.2f}s")
    print(f"  ✓ Testing Time: {test_time:.2f}s")

    return {
        'n_components': dim,
        'explained_variance': explained_variance,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'train_time': train_time,
        'test_time': test_time
    }


def generate_results_table(results_df, output_path):
    """
    Save experimental results to CSV file.

    Args:
        results_df: DataFrame containing results
        output_path: Path to save CSV file
    """
    results_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Results saved to: {output_path}")


def plot_accuracy_vs_dimensions(results_df, output_path):
    """
    Plot classification accuracy vs number of dimensions.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(results_df['n_components'], results_df['accuracy'],
             marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Classification Accuracy')

    # Highlight baseline (784 dimensions)
    baseline = results_df[results_df['n_components'] == 784]
    plt.scatter(baseline['n_components'], baseline['accuracy'],
                color='red', s=150, zorder=5, label='Baseline (784D)', marker='*')

    plt.xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
    plt.ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Classification Accuracy vs Dimensionality\n(K-Nearest Neighbors on MNIST Dataset)',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_error_rate_vs_dimensions(results_df, output_path):
    """
    Plot classification error rate vs number of dimensions.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(results_df['n_components'], results_df['error_rate'],
             marker='s', linewidth=2, markersize=8, color='#A23B72', label='Classification Error Rate')

    # Highlight baseline
    baseline = results_df[results_df['n_components'] == 784]
    plt.scatter(baseline['n_components'], baseline['error_rate'],
                color='red', s=150, zorder=5, label='Baseline (784D)', marker='*')

    plt.xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
    plt.ylabel('Classification Error Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Classification Error Rate vs Dimensionality\n(K-Nearest Neighbors on MNIST Dataset)',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_explained_variance(results_df, output_path):
    """
    Plot cumulative explained variance vs number of components.
    """
    # Exclude baseline (784) from this plot
    df_pca = results_df[results_df['n_components'] != 784].copy()

    plt.figure(figsize=(10, 6))

    plt.plot(df_pca['n_components'], df_pca['explained_variance'],
             marker='D', linewidth=2, markersize=8, color='#F18F01', label='Cumulative Explained Variance')

    # Add horizontal line at 90% and 95%
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Variance')
    plt.axhline(y=95, color='blue', linestyle='--', alpha=0.5, label='95% Variance')

    plt.xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
    plt.title('Information Retention vs Dimensionality\n(Principal Component Analysis on MNIST)',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='lower right')
    plt.ylim(0, 105)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_computational_time(results_df, output_path):
    """
    Plot computational time vs number of dimensions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training time
    ax1.plot(results_df['n_components'], results_df['train_time'],
             marker='o', linewidth=2, markersize=6, color='#06A77D', label='Training Time')
    ax1.set_xlabel('Number of Principal Components', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('KNN Training Time vs Dimensionality', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)

    # Testing time
    ax2.plot(results_df['n_components'], results_df['test_time'],
             marker='s', linewidth=2, markersize=6, color='#D62246', label='Testing Time')
    ax2.set_xlabel('Number of Principal Components', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Testing Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('KNN Prediction Time vs Dimensionality', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def main():
    """
    Main execution function.
    """
    # Load dataset
    X_train, X_test, y_train, y_test = load_mnist_dataset()

    # Preprocess data
    X_train_norm, X_test_norm = preprocess_data(X_train, X_test)

    print("\n[3/8] Running PCA + KNN experiments...")
    print("="*80)

    # Store results
    results = []

    # Total experiments: PCA dimensions + baseline
    total_experiments = len(DIMENSIONS_TO_TEST) + 1
    experiment_num = 1

    # Run PCA experiments
    for n_comp in DIMENSIONS_TO_TEST:
        result = run_experiment(
            X_train_norm, X_test_norm, y_train, y_test,
            n_comp, experiment_num, total_experiments
        )
        results.append(result)
        experiment_num += 1

    # Run baseline experiment (no PCA)
    result = run_experiment(
        X_train_norm, X_test_norm, y_train, y_test,
        None, experiment_num, total_experiments
    )
    results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('n_components').reset_index(drop=True)

    print("\n" + "="*80)
    print("[4/8] Experimental Results Summary")
    print("="*80)
    print(results_df.to_string(index=False))

    # Save results
    print("\n[5/8] Saving experimental results...")
    results_path = os.path.join('results', 'experimental_results.csv')
    generate_results_table(results_df, results_path)

    # Generate graphs
    print("\n[6/8] Generating visualizations...")

    plot_accuracy_vs_dimensions(
        results_df,
        os.path.join('figures', 'accuracy_vs_dimensions.png')
    )

    plot_error_rate_vs_dimensions(
        results_df,
        os.path.join('figures', 'error_rate_vs_dimensions.png')
    )

    plot_explained_variance(
        results_df,
        os.path.join('figures', 'explained_variance.png')
    )

    plot_computational_time(
        results_df,
        os.path.join('figures', 'computational_time.png')
    )

    # Analysis summary
    print("\n[7/8] Experimental Analysis Summary")
    print("="*80)

    best_accuracy_idx = results_df['accuracy'].idxmax()
    best_result = results_df.loc[best_accuracy_idx]

    baseline_result = results_df[results_df['n_components'] == 784].iloc[0]

    print(f"\nBaseline Performance (784 dimensions):")
    print(f"  - Accuracy: {baseline_result['accuracy']:.2f}%")
    print(f"  - Training Time: {baseline_result['train_time']:.2f}s")
    print(f"  - Testing Time: {baseline_result['test_time']:.2f}s")

    print(f"\nBest PCA Performance ({int(best_result['n_components'])} components):")
    print(f"  - Accuracy: {best_result['accuracy']:.2f}%")
    print(f"  - Explained Variance: {best_result['explained_variance']:.2f}%")
    print(f"  - Training Time: {best_result['train_time']:.2f}s")
    print(f"  - Testing Time: {best_result['test_time']:.2f}s")

    accuracy_diff = best_result['accuracy'] - baseline_result['accuracy']
    time_speedup = baseline_result['train_time'] / best_result['train_time']

    print(f"\nComparison:")
    print(f"  - Accuracy difference: {accuracy_diff:+.2f}%")
    print(f"  - Training speedup: {time_speedup:.2f}x")
    print(f"  - Dimensionality reduction: {(1 - best_result['n_components']/784)*100:.1f}%")

    # Find optimal balance (good accuracy with low dimensions)
    df_pca = results_df[results_df['n_components'] != 784]
    # Components that achieve >95% of baseline accuracy
    threshold = baseline_result['accuracy'] * 0.95
    good_performers = df_pca[df_pca['accuracy'] >= threshold]

    if not good_performers.empty:
        optimal = good_performers.loc[good_performers['n_components'].idxmin()]
        print(f"\nOptimal Balance ({int(optimal['n_components'])} components):")
        print(f"  - Retains {optimal['explained_variance']:.1f}% of variance")
        print(f"  - Achieves {optimal['accuracy']:.2f}% accuracy ({optimal['accuracy']/baseline_result['accuracy']*100:.1f}% of baseline)")
        print(f"  - Reduces dimensionality by {(1-optimal['n_components']/784)*100:.0f}%")
        print(f"  - {baseline_result['train_time']/optimal['train_time']:.1f}x faster training")

    print("\n[8/8] Experiment completed successfully!")
    print("="*80)
    print("\nAll deliverables generated:")
    print("  ✓ results/experimental_results.csv")
    print("  ✓ figures/accuracy_vs_dimensions.png")
    print("  ✓ figures/error_rate_vs_dimensions.png")
    print("  ✓ figures/explained_variance.png")
    print("  ✓ figures/computational_time.png")
    print("\nReady for report generation!")
    print("="*80)


if __name__ == "__main__":
    main()
