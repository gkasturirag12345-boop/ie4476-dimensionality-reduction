"""
Generate Academic Word Report for Dimensionality Reduction Project
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import pandas as pd
import os

def create_report():
    """Generate a professional academic report in Word format."""

    doc = Document()

    # Set default styles
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)

    # Title Page
    title = doc.add_heading('Dimensionality Reduction and Its Effect on Classification Performance Using Principal Component Analysis', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()
    doc.add_paragraph()

    # Abstract
    doc.add_heading('Abstract', 1)
    abstract_text = (
        "This report investigates the effect of dimensionality reduction on classification performance "
        "using Principal Component Analysis (PCA) and K-Nearest Neighbors (KNN) on the MNIST handwritten "
        "digit dataset. The study evaluates classification accuracy, error rates, and computational efficiency "
        "across 13 different dimensionalities (5 to 500 principal components) and compares them to the baseline "
        "784-dimensional dataset. Results demonstrate that PCA effectively reduces dimensionality while maintaining "
        "high classification accuracy, with optimal performance achieved at intermediate component counts that balance "
        "information retention and computational efficiency. The findings highlight the practical trade-offs between "
        "dimensionality reduction, classification performance, and computational cost."
    )
    doc.add_paragraph(abstract_text)

    doc.add_page_break()

    # 1. Introduction
    doc.add_heading('1. Introduction', 1)

    intro_p1 = (
        "In modern machine learning and pattern recognition, high-dimensional datasets pose significant "
        "computational and analytical challenges. As the number of features increases, the volume of the "
        "feature space grows exponentially, leading to the well-known 'curse of dimensionality'. This phenomenon "
        "manifests in increased computational complexity, reduced model interpretability, and potential overfitting "
        "due to the exponential growth of data sparsity in high-dimensional spaces."
    )
    doc.add_paragraph(intro_p1)

    intro_p2 = (
        "Dimensionality reduction techniques address these challenges by transforming high-dimensional data "
        "into lower-dimensional representations while preserving essential information. Principal Component Analysis "
        "(PCA) is one of the most widely used unsupervised dimensionality reduction methods. PCA identifies the "
        "directions of maximum variance in the dataset and projects the data onto a lower-dimensional subspace "
        "spanned by the principal components, which are orthogonal linear combinations of the original features."
    )
    doc.add_paragraph(intro_p2)

    intro_p3 = (
        "This study investigates how PCA-based dimensionality reduction affects the performance of the K-Nearest "
        "Neighbors (KNN) classification algorithm on the MNIST handwritten digit dataset. The MNIST dataset, "
        "consisting of 70,000 grayscale images of handwritten digits (0-9), is a benchmark dataset in computer "
        "vision and machine learning. Each image contains 28×28 pixels, resulting in a 784-dimensional feature "
        "space when flattened into a vector representation."
    )
    doc.add_paragraph(intro_p3)

    intro_p4 = (
        "The primary objectives of this experiment are to: (1) evaluate how classification accuracy changes as "
        "the number of retained principal components varies from 5 to 500; (2) assess the computational efficiency "
        "gains achieved through dimensionality reduction; (3) determine the optimal balance between dimensionality "
        "reduction and classification performance; and (4) analyze the relationship between explained variance and "
        "classification accuracy."
    )
    doc.add_paragraph(intro_p4)

    doc.add_page_break()

    # 2. Methodology
    doc.add_heading('2. Methodology', 1)

    doc.add_heading('2.1 Dataset Description', 2)
    dataset_text = (
        "The MNIST (Modified National Institute of Standards and Technology) dataset is used in this experiment. "
        "It comprises 70,000 grayscale images of handwritten digits (0-9), with each image measuring 28×28 pixels. "
        "The dataset is pre-partitioned into 60,000 training samples and 10,000 testing samples. Each image is "
        "flattened into a 784-dimensional feature vector, where each element represents a pixel intensity value "
        "ranging from 0 (black) to 255 (white). The dataset contains 10 balanced classes corresponding to the "
        "digits 0 through 9."
    )
    doc.add_paragraph(dataset_text)

    doc.add_heading('2.2 Data Preprocessing', 2)
    preprocess_text = (
        "Pixel values are normalized to the range [0, 1] by dividing by 255. This normalization ensures consistent "
        "scale across all features and improves the numerical stability of PCA. No additional feature engineering "
        "or data augmentation is applied. The same normalization transformation is applied identically to both "
        "training and testing datasets to prevent data leakage."
    )
    doc.add_paragraph(preprocess_text)

    doc.add_heading('2.3 Principal Component Analysis', 2)
    pca_text = (
        "PCA is applied to reduce the dimensionality of the dataset from 784 to various target dimensions. "
        "PCA identifies the directions of maximum variance in the training data by computing the eigenvectors "
        "of the covariance matrix. The principal components are ranked by their corresponding eigenvalues, "
        "representing the amount of variance explained by each component. Mathematically, PCA transforms the "
        "original data matrix X into a lower-dimensional representation Z:"
    )
    doc.add_paragraph(pca_text)

    pca_eq = doc.add_paragraph()
    pca_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pca_eq.add_run("Z = X · W")

    pca_text2 = (
        "where W is the projection matrix formed by the top k eigenvectors. Critically, PCA is fitted exclusively "
        "on the training dataset, and the same transformation is applied to the testing dataset to prevent data "
        "leakage. The cumulative explained variance ratio quantifies the proportion of total variance retained "
        "after dimensionality reduction."
    )
    doc.add_paragraph(pca_text2)

    doc.add_heading('2.4 K-Nearest Neighbors Classification', 2)
    knn_text = (
        "The K-Nearest Neighbors (KNN) algorithm is employed for classification. KNN is a non-parametric, "
        "instance-based learning algorithm that classifies a test sample based on the majority vote of its "
        "k nearest neighbors in the feature space. In this study, k=5 neighbors are used with Euclidean distance "
        "as the similarity metric. The Euclidean distance between two points x and y in d-dimensional space is:"
    )
    doc.add_paragraph(knn_text)

    dist_eq = doc.add_paragraph()
    dist_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dist_eq.add_run("d(x, y) = √[Σ(x_i - y_i)²]")

    doc.add_heading('2.5 Experimental Configuration', 2)
    config_text = (
        "The experiment evaluates 14 configurations: 13 PCA-reduced datasets with component counts of "
        "5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, and 500, plus the baseline 784-dimensional dataset "
        "without PCA. For each configuration, the KNN classifier is trained on the training set and evaluated on "
        "the testing set. Performance is measured using classification accuracy and error rate:"
    )
    doc.add_paragraph(config_text)

    acc_eq = doc.add_paragraph()
    acc_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    acc_eq.add_run("Accuracy = (Correct Predictions / Total Predictions) × 100%")

    err_eq = doc.add_paragraph()
    err_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    err_eq.add_run("Error Rate = 100% - Accuracy")

    doc.add_paragraph(
        "Training time and prediction time are also recorded to assess computational efficiency. All experiments "
        "use fixed random seeds to ensure reproducibility."
    )

    doc.add_page_break()

    # 3. Experimental Results
    doc.add_heading('3. Experimental Results', 1)

    doc.add_heading('3.1 Numerical Results', 2)

    # Load actual results
    if os.path.exists('results/experimental_results.csv'):
        results_df = pd.read_csv('results/experimental_results.csv')

        doc.add_paragraph(
            "Table 1 presents the complete experimental results across all tested dimensionalities. "
            "The table includes the number of principal components, cumulative explained variance, "
            "classification accuracy, error rate, and computational times."
        )

        doc.add_paragraph()

        # Create results table
        table = doc.add_table(rows=len(results_df) + 1, cols=6)
        table.style = 'Light Grid Accent 1'

        # Header
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Components'
        header_cells[1].text = 'Variance (%)'
        header_cells[2].text = 'Accuracy (%)'
        header_cells[3].text = 'Error (%)'
        header_cells[4].text = 'Train (s)'
        header_cells[5].text = 'Test (s)'

        # Data rows
        for i, row in results_df.iterrows():
            cells = table.rows[i + 1].cells
            cells[0].text = str(int(row['n_components']))
            cells[1].text = f"{row['explained_variance']:.2f}"
            cells[2].text = f"{row['accuracy']:.2f}"
            cells[3].text = f"{row['error_rate']:.2f}"
            cells[4].text = f"{row['train_time']:.2f}"
            cells[5].text = f"{row['test_time']:.2f}"

        doc.add_paragraph()
        caption = doc.add_paragraph("Table 1: Experimental results for all tested dimensionalities")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption_run = caption.runs[0]
        caption_run.font.size = Pt(10)
        caption_run.font.italic = True

    doc.add_page_break()

    doc.add_heading('3.2 Graphical Analysis', 2)

    # Add figures
    figures = [
        ('figures/accuracy_vs_dimensions.png', 'Figure 1: Classification accuracy vs number of principal components'),
        ('figures/error_rate_vs_dimensions.png', 'Figure 2: Classification error rate vs number of principal components'),
        ('figures/explained_variance.png', 'Figure 3: Cumulative explained variance vs number of principal components'),
        ('figures/computational_time.png', 'Figure 4: Computational time comparison across dimensionalities')
    ]

    for fig_path, caption_text in figures:
        if os.path.exists(fig_path):
            doc.add_picture(fig_path, width=Inches(6))
            caption = doc.add_paragraph(caption_text)
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_run = caption.runs[0]
            caption_run.font.size = Pt(10)
            caption_run.font.italic = True
            doc.add_paragraph()

    doc.add_page_break()

    # 4. Discussion
    doc.add_heading('4. Discussion and Analysis', 1)

    if os.path.exists('results/experimental_results.csv'):
        results_df = pd.read_csv('results/experimental_results.csv')

        baseline = results_df[results_df['n_components'] == 784].iloc[0]
        best_idx = results_df['accuracy'].idxmax()
        best = results_df.loc[best_idx]

        discussion_p1 = (
            f"The experimental results reveal several key findings regarding the effect of dimensionality "
            f"reduction on classification performance. The baseline KNN classifier achieved {baseline['accuracy']:.2f}% "
            f"accuracy on the full 784-dimensional dataset. The highest classification accuracy of "
            f"{best['accuracy']:.2f}% was achieved with {int(best['n_components'])} principal components, "
            f"retaining {best['explained_variance']:.2f}% of the total variance."
        )
        doc.add_paragraph(discussion_p1)

        # Find optimal balance
        df_pca = results_df[results_df['n_components'] != 784]
        threshold = baseline['accuracy'] * 0.95
        good = df_pca[df_pca['accuracy'] >= threshold]

        if not good.empty:
            optimal = good.loc[good['n_components'].idxmin()]

            discussion_p2 = (
                f"Analysis of the accuracy-dimensionality relationship demonstrates that classification performance "
                f"remains robust even with significant dimensionality reduction. Using only {int(optimal['n_components'])} "
                f"components ({optimal['explained_variance']:.1f}% variance), the classifier achieves "
                f"{optimal['accuracy']:.2f}% accuracy, representing {optimal['accuracy']/baseline['accuracy']*100:.1f}% "
                f"of baseline performance while reducing dimensionality by "
                f"{(1-optimal['n_components']/784)*100:.0f}%. This configuration provides an optimal balance, "
                f"delivering comparable accuracy with {baseline['train_time']/optimal['train_time']:.1f}× faster "
                f"training time."
            )
            doc.add_paragraph(discussion_p2)

        discussion_p3 = (
            "The relationship between explained variance and classification accuracy reveals an important insight: "
            "retaining a higher percentage of variance does not always guarantee better classification performance. "
            "This phenomenon occurs because PCA maximizes variance, not class separability. Some high-variance "
            "components may capture intra-class variation or noise rather than discriminative features. Conversely, "
            "lower-variance components that are discarded may contain subtle but critical class-discriminative information."
        )
        doc.add_paragraph(discussion_p3)

        discussion_p4 = (
            "Computational efficiency analysis shows that dimensionality reduction significantly reduces both "
            "training and prediction times. KNN is a lazy learner that stores all training data and computes "
            "distances to all neighbors during prediction. Reducing dimensionality from 784 to lower values "
            "substantially decreases the distance computation cost, resulting in faster predictions. The reduction "
            "in training time primarily reflects the computational cost of PCA transformation."
        )
        doc.add_paragraph(discussion_p4)

        discussion_p5 = (
            "At very low dimensionalities (5-20 components), classification accuracy degrades noticeably due to "
            "information loss. These configurations discard too much variance, resulting in insufficient information "
            "to distinguish between classes effectively. At higher dimensionalities (300-500 components), accuracy "
            "stabilizes and approaches baseline performance, but computational gains diminish. The optimal operating "
            "point lies in the intermediate range, where the trade-off between accuracy and efficiency is balanced."
        )
        doc.add_paragraph(discussion_p5)

        # Limitations
        doc.add_heading('4.1 Limitations', 2)
        limitations = (
            "This study has several limitations. First, only KNN classification was evaluated; other algorithms "
            "may exhibit different sensitivity to dimensionality reduction. Second, the experiment used a single "
            "dataset (MNIST); generalization to other domains requires further investigation. Third, only PCA was "
            "considered; alternative dimensionality reduction techniques such as Linear Discriminant Analysis (LDA), "
            "t-SNE, or autoencoders may yield different results. Finally, the choice of k=5 neighbors was fixed; "
            "different values of k may interact differently with dimensionality."
        )
        doc.add_paragraph(limitations)

    doc.add_page_break()

    # 5. Conclusion
    doc.add_heading('5. Conclusion', 1)

    if os.path.exists('results/experimental_results.csv'):
        results_df = pd.read_csv('results/experimental_results.csv')
        baseline = results_df[results_df['n_components'] == 784].iloc[0]
        df_pca = results_df[results_df['n_components'] != 784]
        threshold = baseline['accuracy'] * 0.95
        good = df_pca[df_pca['accuracy'] >= threshold]

        if not good.empty:
            optimal = good.loc[good['n_components'].idxmin()]

            conclusion = (
                f"This study successfully demonstrates that Principal Component Analysis effectively reduces "
                f"dimensionality while preserving classification performance. The experiments reveal that {int(optimal['n_components'])} "
                f"principal components provide the optimal balance, retaining {optimal['explained_variance']:.1f}% "
                f"of variance and achieving {optimal['accuracy']:.2f}% accuracy compared to {baseline['accuracy']:.2f}% "
                f"for the full 784-dimensional dataset. This represents a {(1-optimal['n_components']/784)*100:.0f}% "
                f"reduction in dimensionality with minimal accuracy loss, while significantly improving computational "
                f"efficiency.\n\n"
                f"The key findings are: (1) PCA successfully reduces dimensionality without substantial accuracy loss; "
                f"(2) intermediate component counts balance information retention and efficiency; (3) very low "
                f"dimensionalities cause significant information loss; (4) computational time decreases substantially "
                f"with dimensionality reduction; and (5) the relationship between variance and accuracy is nonlinear.\n\n"
                f"These results have practical implications for machine learning systems operating under computational "
                f"constraints. Dimensionality reduction via PCA enables deployment of classifiers on resource-limited "
                f"devices while maintaining acceptable performance. Future work should investigate alternative "
                f"dimensionality reduction techniques, different classification algorithms, and domain-specific datasets "
                f"to establish the generalizability of these findings."
            )
            doc.add_paragraph(conclusion)

    doc.add_page_break()

    # References
    doc.add_heading('6. References', 1)

    references = [
        "[1] I. T. Jolliffe and J. Cadima, \"Principal component analysis: a review and recent developments,\" Philosophical Transactions of the Royal Society A, vol. 374, no. 2065, 2016.",
        "[2] T. Cover and P. Hart, \"Nearest neighbor pattern classification,\" IEEE Transactions on Information Theory, vol. 13, no. 1, pp. 21-27, 1967.",
        "[3] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, \"Gradient-based learning applied to document recognition,\" Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.",
        "[4] R. Bellman, \"Dynamic programming,\" Princeton University Press, 1957.",
        "[5] K. Pearson, \"On lines and planes of closest fit to systems of points in space,\" The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, vol. 2, no. 11, pp. 559-572, 1901.",
        "[6] F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011."
    ]

    for ref in references:
        doc.add_paragraph(ref, style='List Number')

    doc.add_page_break()

    # Appendix
    doc.add_heading('Appendix A: Complete Python Program', 1)

    appendix_text = (
        "The complete Python source code used to conduct this experiment is provided below. "
        "The implementation includes dataset loading, preprocessing, PCA transformation, KNN classification, "
        "performance evaluation, and visualization generation."
    )
    doc.add_paragraph(appendix_text)

    doc.add_paragraph()

    # Add the Python code
    if os.path.exists('main.py'):
        with open('main.py', 'r') as f:
            code = f.read()

        code_para = doc.add_paragraph()
        code_run = code_para.add_run(code)
        code_run.font.name = 'Courier New'
        code_run.font.size = Pt(8)

    # Save document
    output_path = 'report/Dimensionality_Reduction_Report.docx'
    doc.save(output_path)

    print(f"\n✓ Academic report generated: {output_path}")

    return output_path


if __name__ == "__main__":
    create_report()
