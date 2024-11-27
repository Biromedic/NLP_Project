package org.Nlp.validator;

import org.Nlp.knn.KNNClassifier;
import org.Nlp.metrics.MetricsCalculator;

import java.util.*;
import java.util.stream.Collectors;

public class CrossValidator {

    private final MetricsCalculator metricsCalculator = new MetricsCalculator();
    private final List<Map<String, Double>> foldMetrics = new ArrayList<>();

    /**
     * Performs stratified 10-fold cross-validation.
     *
     * @param tfidfMatrix TF-IDF matrix (2D array) representing document-term features.
     * @param labels      List of true class labels for the documents.
     * @param k           Number of neighbors for k-NN.
     * @param metric      Similarity metric (e.g., "cosine", "euclidean").
     */
    public void crossValidate(double[][] tfidfMatrix, List<String> labels, int k, String metric) {
        if (tfidfMatrix == null || labels == null || tfidfMatrix.length != labels.size()) {
            throw new IllegalArgumentException("Invalid input: TF-IDF matrix and labels must have matching lengths.");
        }

        // Prepare data for stratified folds
        Map<String, List<Integer>> labelIndices = groupByLabel(labels);
        List<List<Integer>> folds = createStratifiedFolds(labelIndices, 10);

        for (int foldIndex = 0; foldIndex < folds.size(); foldIndex++) {
            System.out.println("Processing fold " + (foldIndex + 1) + "...");

            // Test indices for the current fold
            Set<Integer> testIndices = new HashSet<>(folds.get(foldIndex));
            int finalFoldIndex = foldIndex;

            // Train indices: all other folds
            Set<Integer> trainIndices = folds.stream()
                    .filter(fold -> !fold.equals(folds.get(finalFoldIndex)))
                    .flatMap(Collection::stream)
                    .collect(Collectors.toSet());

            // Split into train/test sets
            double[][] trainMatrix = extractRows(tfidfMatrix, trainIndices);
            double[][] testMatrix = extractRows(tfidfMatrix, testIndices);
            List<String> trainLabels = extractLabels(labels, trainIndices);
            List<String> testLabels = extractLabels(labels, testIndices);

            // Train and test k-NN classifier
            KNNClassifier knn = new KNNClassifier(Arrays.asList(trainMatrix), trainLabels);
            List<String> predictedLabels = new ArrayList<>();
            for (double[] testVector : testMatrix) {
                predictedLabels.add(knn.predict(testVector, k, metric));
            }

            // Compute confusion matrix and metrics
            int[][] confusionMatrix = metricsCalculator.computeConfusionMatrix(
                    testLabels, predictedLabels, new ArrayList<>(new HashSet<>(labels))
            );
            foldMetrics.add(metricsCalculator.calculateMetrics(confusionMatrix));
        }
    }

    /**
     * Groups document indices by their corresponding labels.
     *
     * @param labels List of labels.
     * @return Map of labels to document indices.
     */
    private Map<String, List<Integer>> groupByLabel(List<String> labels) {
        Map<String, List<Integer>> labelIndices = new HashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            labelIndices.computeIfAbsent(labels.get(i), k -> new ArrayList<>()).add(i);
        }
        return labelIndices;
    }

    /**
     * Creates stratified folds for cross-validation.
     *
     * @param labelIndices Map of labels to document indices.
     * @param numFolds     Number of folds.
     * @return List of folds (each fold is a list of indices).
     */
    private List<List<Integer>> createStratifiedFolds(Map<String, List<Integer>> labelIndices, int numFolds) {
        List<List<Integer>> folds = new ArrayList<>();
        for (int i = 0; i < numFolds; i++) {
            folds.add(new ArrayList<>());
        }

        for (Map.Entry<String, List<Integer>> entry : labelIndices.entrySet()) {
            List<Integer> indices = entry.getValue();
            Collections.shuffle(indices, new Random(42)); // Fixed seed for reproducibility

            int foldIndex = 0;
            for (int index : indices) {
                folds.get(foldIndex % numFolds).add(index);
                foldIndex++;
            }
        }

        return folds;
    }

    /**
     * Extracts rows from a matrix based on the specified indices.
     */
    private double[][] extractRows(double[][] matrix, Set<Integer> indices) {
        return indices.stream()
                .map(index -> Arrays.copyOf(matrix[index], matrix[index].length))
                .toArray(double[][]::new);
    }


    /**
     * Extracts labels for the specified indices.
     */
    private List<String> extractLabels(List<String> labels, Set<Integer> indices) {
        return indices.stream().map(labels::get).collect(Collectors.toList());
    }

    /**
     * Aggregates and returns the average performance metrics across all folds.
     *
     * @return A map containing macro-averaged precision, recall, and F1-score.
     */
    public Map<String, Double> getPerformanceMetrics() {
        double macroPrecision = 0, macroRecall = 0, macroF1 = 0;

        for (Map<String, Double> fold : foldMetrics) {
            macroPrecision += fold.get("Macro Precision");
            macroRecall += fold.get("Macro Recall");
            macroF1 += fold.get("Macro F1-Score");
        }

        int numFolds = foldMetrics.size();
        macroPrecision /= numFolds;
        macroRecall /= numFolds;
        macroF1 /= numFolds;

        Map<String, Double> summaryMetrics = new HashMap<>();
        summaryMetrics.put("Macro Precision", macroPrecision);
        summaryMetrics.put("Macro Recall", macroRecall);
        summaryMetrics.put("Macro F1-Score", macroF1);
        return summaryMetrics;
    }
}
