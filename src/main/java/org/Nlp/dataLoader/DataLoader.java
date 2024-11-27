package org.Nlp.dataLoader;

import java.io.IOException;
import java.util.*;

public class DataLoader {

    private final FileReader fileReader;
    private final Map<String, List<String>> classData = new HashMap<>();

    /**
     * Constructor for DataLoader.
     */
    public DataLoader() {
        this.fileReader = new FileReader();
    }

    /**
     * Loads data from the given folder paths and associates them with their class labels.
     *
     * @param classFolders A map of class labels to their respective folder paths.
     * @throws IOException if an error occurs while reading files.
     */
    public void loadData(Map<String, String> classFolders) throws IOException {
        for (Map.Entry<String, String> entry : classFolders.entrySet()) {
            String className = entry.getKey();
            String folderPath = entry.getValue();

            // Read files for the class and add to the map
            List<String> data = fileReader.readFiles(folderPath);
            if (data.isEmpty()) {
                System.err.println("Warning: No data found for class: " + className);
            }
            classData.put(className, data);
        }
    }

    /**
     * Calculates the class distributions based on the loaded data.
     *
     * @return A map of class labels to the count of their data entries.
     */
    public Map<String, Integer> calculateClassDistributions() {
        Map<String, Integer> distributions = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : classData.entrySet()) {
            distributions.put(entry.getKey(), entry.getValue().size());
        }
        return distributions;
    }

    /**
     * Cleans up data entries with null or empty content.
     */
    public void cleanData() {
        for (Map.Entry<String, List<String>> entry : classData.entrySet()) {
            List<String> cleanedData = new ArrayList<>();
            for (String text : entry.getValue()) {
                if (text != null && !text.trim().isEmpty()) {
                    cleanedData.add(text);
                }
            }
            classData.put(entry.getKey(), cleanedData);
        }
    }
}
