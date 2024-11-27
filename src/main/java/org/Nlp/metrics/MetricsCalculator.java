package org.Nlp.metrics;

import java.util.*;

public class MetricsCalculator {

    /**
     * Computes the confusion matrix for multi-class classification.
     *
     * @param trueLabels     Ground truth labels.
     * @param predictedLabels Predicted labels.
     * @param uniqueLabels   List of unique class labels in the dataset.
     * @return Confusion matrix as a 2D array.
     */
    public int[][] computeConfusionMatrix(List<String> trueLabels, List<String> predictedLabels, List<String> uniqueLabels) {
        if (trueLabels == null || predictedLabels == null || uniqueLabels == null) {
            throw new IllegalArgumentException("Input labels and uniqueLabels must not be null.");
        }
        if (trueLabels.size() != predictedLabels.size()) {
            throw new IllegalArgumentException("True labels and predicted labels must have the same size.");
        }

        int numClasses = uniqueLabels.size();
        int[][] confusionMatrix = new int[numClasses][numClasses];

        // Map each unique label to an index
        Map<String, Integer> labelToIndex = new HashMap<>();
        for (int i = 0; i < uniqueLabels.size(); i++) {
            labelToIndex.put(uniqueLabels.get(i), i);
        }

        // Populate confusion matrix
        for (int i = 0; i < trueLabels.size(); i++) {
            String trueLabel = trueLabels.get(i);
            String predictedLabel = predictedLabels.get(i);

            Integer trueIndex = labelToIndex.get(trueLabel);
            Integer predictedIndex = labelToIndex.get(predictedLabel);

            if (trueIndex == null || predictedIndex == null) {
                throw new IllegalArgumentException("Label not found in uniqueLabels list: " + trueLabel + ", " + predictedLabel);
            }

            confusionMatrix[trueIndex][predictedIndex]++;
        }

        return confusionMatrix;
    }

    /**
     * Computes precision for a specific class.
     *
     * @param confusionMatrix Confusion matrix.
     * @param classIndex      Class index for which precision is calculated.
     * @return Precision for the given class.
     */
    public double computePrecision(int[][] confusionMatrix, int classIndex) {
        int truePositive = confusionMatrix[classIndex][classIndex];
        int falsePositive = 0;
        for (int i = 0; i < confusionMatrix.length; i++) {
            if (i != classIndex) {
                falsePositive += confusionMatrix[i][classIndex];
            }
        }
        if (truePositive + falsePositive == 0) {
            return 0.0; // Avoid division by zero
        }
        return (double) truePositive / (truePositive + falsePositive);
    }

    /**
     * Computes recall for a specific class.
     *
     * @param confusionMatrix Confusion matrix.
     * @param classIndex      Class index for which recall is calculated.
     * @return Recall for the given class.
     */
    public double computeRecall(int[][] confusionMatrix, int classIndex) {
        int truePositive = confusionMatrix[classIndex][classIndex];
        int falseNegative = 0;
        for (int i = 0; i < confusionMatrix[classIndex].length; i++) {
            if (i != classIndex) {
                falseNegative += confusionMatrix[classIndex][i];
            }
        }
        if (truePositive + falseNegative == 0) {
            return 0.0; // Avoid division by zero
        }
        return (double) truePositive / (truePositive + falseNegative);
    }

    /**
     * Computes F1-score given precision and recall.
     *
     * @param precision Precision value.
     * @param recall    Recall value.
     * @return F1-score.
     */
    public double computeF1Score(double precision, double recall) {
        if (precision + recall == 0) {
            return 0.0; // Avoid division by zero
        }
        return 2 * (precision * recall) / (precision + recall);
    }

    /**
     * Calculates precision, recall, and F1-score metrics for all classes and macro averages.
     *
     * @param confusionMatrix Confusion matrix.
     * @return Map containing macro averages for precision, recall, and F1-score.
     */
    public Map<String, Double> calculateMetrics(int[][] confusionMatrix) {
        int numClasses = confusionMatrix.length;

        double macroPrecision = 0.0, macroRecall = 0.0, macroF1 = 0.0;
        for (int i = 0; i < numClasses; i++) {
            double precision = computePrecision(confusionMatrix, i);
            double recall = computeRecall(confusionMatrix, i);
            double f1Score = computeF1Score(precision, recall);

            macroPrecision += precision;
            macroRecall += recall;
            macroF1 += f1Score;
        }

        // Calculate macro averages
        macroPrecision /= numClasses;
        macroRecall /= numClasses;
        macroF1 /= numClasses;

        Map<String, Double> metrics = new HashMap<>();
        metrics.put("Macro Precision", macroPrecision);
        metrics.put("Macro Recall", macroRecall);
        metrics.put("Macro F1-Score", macroF1);

        return metrics;
    }

    /**
     * Logs the confusion matrix for debugging purposes.
     *
     * @param confusionMatrix Confusion matrix to log.
     * @param uniqueLabels    List of unique class labels.
     */
    public void logConfusionMatrix(int[][] confusionMatrix, List<String> uniqueLabels) {
        System.out.println("Confusion Matrix:");
        System.out.printf("%10s", " ");
        for (String label : uniqueLabels) {
            System.out.printf("%10s", label);
        }
        System.out.println();

        for (int i = 0; i < confusionMatrix.length; i++) {
            System.out.printf("%10s", uniqueLabels.get(i));
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                System.out.printf("%10d", confusionMatrix[i][j]);
            }
            System.out.println();
        }
    }
}
