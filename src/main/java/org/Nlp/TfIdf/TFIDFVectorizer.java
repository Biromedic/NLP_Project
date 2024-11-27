package org.Nlp.TfIdf;

import java.util.*;
import java.util.stream.Collectors;

public class TFIDFVectorizer {
    private final Map<String, Double> idfValues = new HashMap<>();
    private final List<Map<String, Double>> tfidfMatrix = new ArrayList<>();
    private List<String> vocabulary = new ArrayList<>();

    /**
     * Computes the TF-IDF values for the given documents.
     *
     * @param documents List of preprocessed documents (each as a list of terms).
     */
    public void fit(List<List<String>> documents) {
        if (documents == null || documents.isEmpty()) {
            throw new IllegalArgumentException("Document list cannot be null or empty.");
        }

        computeIDFValues(documents);
        computeTFIDFMatrix(documents);
    }

    /**
     * Computes the IDF values for all terms in the documents.
     *
     * @param documents List of preprocessed documents (each as a list of terms).
     */
    private void computeIDFValues(List<List<String>> documents) {
        Map<String, Integer> docFrequency = new HashMap<>();
        int totalDocs = documents.size();

        for (List<String> doc : documents) {
            Set<String> uniqueTerms = new HashSet<>(doc); // Unique terms in the document
            for (String term : uniqueTerms) {
                docFrequency.put(term, docFrequency.getOrDefault(term, 0) + 1);
            }
        }

        for (String term : docFrequency.keySet()) {
            idfValues.put(term, Math.log((double) totalDocs / docFrequency.get(term)));
        }
        vocabulary = new ArrayList<>(idfValues.keySet());
    }

    /**
     * Computes the TF-IDF matrix for the given documents.
     *
     * @param documents List of preprocessed documents (each as a list of terms).
     */
    private void computeTFIDFMatrix(List<List<String>> documents) {
        for (List<String> doc : documents) {
            Map<String, Double> tfidfRow = new HashMap<>();
            Map<String, Long> termCounts = doc.stream()
                    .collect(Collectors.groupingBy(term -> term, Collectors.counting()));

            for (String term : termCounts.keySet()) {
                double tf = termCounts.get(term) / (double) doc.size();
                double idf = idfValues.getOrDefault(term, 0.0);
                tfidfRow.put(term, tf * idf);
            }
            tfidfMatrix.add(tfidfRow);
        }
    }

    /**
     * Returns the TF-IDF matrix as a 2D array.
     *
     * @return 2D array of TF-IDF values.
     */
    public double[][] getTFIDFMatrix() {
        double[][] matrix = new double[tfidfMatrix.size()][vocabulary.size()];

        for (int i = 0; i < tfidfMatrix.size(); i++) {
            Map<String, Double> row = tfidfMatrix.get(i);
            for (int j = 0; j < vocabulary.size(); j++) {
                matrix[i][j] = row.getOrDefault(vocabulary.get(j), 0.0);
            }
        }
        return matrix;
    }

        /**
     * Returns the vocabulary used in TF-IDF computation.
     *
     * @return List of unique terms in the vocabulary.
     */
    public List<String> getVocabulary() {
        return vocabulary;
    }
}
