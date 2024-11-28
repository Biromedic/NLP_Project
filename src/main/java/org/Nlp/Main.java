package org.Nlp;

import org.Nlp.TfIdf.TFIDFVectorizer;
import org.Nlp.knn.KNNClassifier;
import org.Nlp.metrics.MetricsCalculator;
import org.Nlp.preprocessing.Preprocessor;
import org.Nlp.validator.CrossValidator;
import org.Nlp.dataLoader.DataLoader;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        try {
            // 1. Veri Y�kleme
            DataLoader dataLoader = new DataLoader();
            Map<String, String> classFolders = Map.of(
                    "Positive", "src/main/java/org/Nlp/tweets/1",
                    "Negative", "src/main/java/org/Nlp/tweets/2",
                    "Neutral", "src/main/java/org/Nlp/tweets/3"
            );
            dataLoader.loadData(classFolders);
            dataLoader.cleanData();
            Map<String, List<String>> classData = dataLoader.getAllData();
            Map<String, Integer> classDistributions = dataLoader.calculateClassDistributions();
            System.out.println("S�n�f da��l�mlar�: " + classDistributions);

            // Veriyi birle�tir ve s�n�f etiketlerini olu�tur
            List<String> rawDocuments = new ArrayList<>();
            List<String> labels = new ArrayList<>();
            List<String> documentNames = new ArrayList<>();
            for (Map.Entry<String, List<String>> entry : classData.entrySet()) {
                String className = entry.getKey();
                List<String> documents = entry.getValue();
                rawDocuments.addAll(documents);
                labels.addAll(Collections.nCopies(documents.size(), className));
                for (int i = 0; i < documents.size(); i++) {
                    documentNames.add("Doc_" + (rawDocuments.size() - documents.size() + i + 1));
                }
            }
            System.out.println("Toplam y�klenen belge: " + rawDocuments.size());

            // 2. �n ��leme
            Preprocessor preprocessor = new Preprocessor(null); // Varsay�lan stopword listesiyle
            List<List<String>> preprocessedDocuments = new ArrayList<>();
            List<String> filteredLabels = new ArrayList<>();
            List<String> filteredDocumentNames = new ArrayList<>();

            for (int i = 0; i < rawDocuments.size(); i++) {
                String doc = rawDocuments.get(i);
                List<String> processed = preprocessor.preprocess(doc);
                if (!processed.isEmpty()) {
                    preprocessedDocuments.add(processed);
                    filteredLabels.add(labels.get(i));
                    filteredDocumentNames.add(documentNames.get(i));
                }
            }
            System.out.println("�n i�leme tamamland�. ��lenmi� belgeler: " + preprocessedDocuments.size());

            // 3. TF-IDF Vekt�rizasyonu
            TFIDFVectorizer vectorizer = new TFIDFVectorizer();
            vectorizer.fit(preprocessedDocuments);
            double[][] tfidfMatrix = vectorizer.getTFIDFMatrix();
            System.out.println("TF-IDF vekt�rizasyonu tamamland�.");

            // TF-IDF De�erlerini CSV Format�nda Kaydet
            exportTFIDFToCSV(tfidfMatrix, filteredLabels, filteredDocumentNames, vectorizer.getVocabulary(), "tfidf_values.csv");

            // 4. Farkl� k De�erleri i�in �apraz Do�rulama
            List<Integer> kValues = List.of(3);
            Map<Integer, Map<String, Double>> allMetrics = new HashMap<>();
            CrossValidator crossValidator = new CrossValidator();

            for (int k : kValues) {
                System.out.println("Processing cross-validation for k=" + k);
                crossValidator.crossValidate(tfidfMatrix, filteredLabels, k, "cosine");
                allMetrics.put(k, crossValidator.getPerformanceMetrics());
            }

            // En �yi Performansl� k De�erini Bulma
            int bestK = findBestK(allMetrics);

            // 5. Model Performans Analizi
            MetricsCalculator metricsCalculator = new MetricsCalculator();
            KNNClassifier knnClassifier = new KNNClassifier(Arrays.asList(tfidfMatrix), filteredLabels);
            List<String> predictedLabels = new ArrayList<>();

            for (double[] documentVector : tfidfMatrix) {
                predictedLabels.add(knnClassifier.predict(documentVector, bestK, "cosine"));
            }

            int[][] confusionMatrix = metricsCalculator.computeConfusionMatrix(filteredLabels, predictedLabels, new ArrayList<>(new HashSet<>(filteredLabels)));
            metricsCalculator.logConfusionMatrix(confusionMatrix, new ArrayList<>(new HashSet<>(filteredLabels)));
            Map<String, Double> finalMetrics = metricsCalculator.calculateMetrics(confusionMatrix);
            System.out.println("En iyi k (" + bestK + ") ile model metrikleri: " + finalMetrics);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void exportTFIDFToCSV(double[][] tfidfMatrix, List<String> labels, List<String> documentNames, List<String> vocabulary, String fileName) throws IOException {
        try (FileWriter writer = new FileWriter(fileName)) {
            // Ba�l�klar� Yaz
            writer.write("Document," + String.join(",", vocabulary) + ",Class\n");

            // TF-IDF De�erlerini ve S�n�flar� Yaz
            for (int i = 0; i < tfidfMatrix.length; i++) {
                writer.write(documentNames.get(i) + ",");
                for (int j = 0; j < tfidfMatrix[i].length; j++) {
                    writer.write(tfidfMatrix[i][j] + ",");
                }
                writer.write(labels.get(i) + "\n");
            }
        }
        System.out.println("TF-IDF de�erleri CSV dosyas�na kaydedildi: " + fileName);
    }

    private static int findBestK(Map<Integer, Map<String, Double>> allMetrics) {
        return allMetrics.entrySet().stream()
                .max(Comparator.comparingDouble(e -> e.getValue().get("Macro F1-Score")))
                .map(Map.Entry::getKey)
                .orElseThrow(() -> new IllegalStateException("En iyi k de�eri bulunamad�!"));
    }
}
