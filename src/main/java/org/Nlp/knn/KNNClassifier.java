package org.Nlp.knn;

import java.util.*;

public class KNNClassifier {
    private final List<double[]> tfidfMatrix;
    private final List<String> labels;

    /**
     * Constructor for KNNClassifier.
     *
     * @param tfidfMatrix The TF-IDF matrix as a list of double arrays.
     * @param labels      The labels corresponding to each document in the matrix.
     */
    public KNNClassifier(List<double[]> tfidfMatrix, List<String> labels) {
        if (tfidfMatrix == null || labels == null || tfidfMatrix.size() != labels.size()) {
            throw new IllegalArgumentException("TF-IDF matrix and labels must be non-null and of equal size.");
        }
        this.tfidfMatrix = tfidfMatrix;
        this.labels = labels;
    }

    /**
     * Predicts the label for a given test document using k-NN.
     *
     * @param testVector The TF-IDF vector for the test document.
     * @param k          The number of nearest neighbors to consider.
     * @param metric     The similarity metric to use (e.g., "cosine", "euclidean").
     * @return The predicted label.
     */
    public String predict(double[] testVector, int k, String metric) {
        if (testVector == null || k <= 0) {
            throw new IllegalArgumentException("Test vector cannot be null, and k must be greater than 0.");
        }

        // Calculate distances/similarities
        List<Map.Entry<String, Double>> distances = new ArrayList<>();
        for (int i = 0; i < tfidfMatrix.size(); i++) {
            double score = calculateMetric(testVector, tfidfMatrix.get(i), metric);
            distances.add(Map.entry(labels.get(i), score));
        }

        // Sort by similarity in descending order (higher is better for similarity metrics like cosine)
        distances.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        // Get the top k labels
        Map<String, Integer> labelCounts = new HashMap<>();
        for (int i = 0; i < k && i < distances.size(); i++) {
            String label = distances.get(i).getKey();
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }

        // Return the label with the highest count
        return labelCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
    }

    /**
     * Calculates the similarity or distance metric between two vectors.
     *
     * @param vec1   The first vector.
     * @param vec2   The second vector.
     * @param metric The metric to use ("cosine", "euclidean").
     * @return The calculated metric value.
     */
    private double calculateMetric(double[] vec1, double[] vec2, String metric) {
        return switch (metric.toLowerCase()) {
            case "cosine" -> cosineSimilarity(vec1, vec2);
            case "euclidean" -> euclideanDistance(vec1, vec2);
            default -> throw new IllegalArgumentException("Unsupported metric: " + metric);
        };
    }

    /**
     * Calculates cosine similarity between two vectors.
     *
     * @param vec1 The first vector.
     * @param vec2 The second vector.
     * @return The cosine similarity score.
     */
    private double cosineSimilarity(double[] vec1, double[] vec2) {
        double dotProduct = 0.0;
        double magnitudeVec1 = 0.0;
        double magnitudeVec2 = 0.0;

        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            magnitudeVec1 += Math.pow(vec1[i], 2);
            magnitudeVec2 += Math.pow(vec2[i], 2);
        }

        if (magnitudeVec1 == 0.0 || magnitudeVec2 == 0.0) {
            return 0.0; // To avoid division by zero
        }
        return dotProduct / (Math.sqrt(magnitudeVec1) * Math.sqrt(magnitudeVec2));
    }

    /**
     * Calculates Euclidean distance between two vectors.
     *
     * @param vec1 The first vector.
     * @param vec2 The second vector.
     * @return The Euclidean distance.
     */
    private double euclideanDistance(double[] vec1, double[] vec2) {
        double sumSquaredDifferences = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            sumSquaredDifferences += Math.pow(vec1[i] - vec2[i], 2);
        }
        return Math.sqrt(sumSquaredDifferences);
    }
}
