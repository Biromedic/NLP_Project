package org.Nlp.dataLoader;

import java.nio.file.*;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class FileReader {

    private static final String CHARSET = "ISO-8859-9";

    /**
     * Reads all files in a folder and combines their content into a list of strings.
     * Each file's content is treated as a single string entry in the returned list.
     *
     * @param folderPath Path to the folder containing files.
     * @return List of file contents as strings.
     * @throws IOException if an I/O error occurs.
     */
    public List<String> readFiles(String folderPath) throws IOException {
        Path folder = Paths.get(folderPath);

        // Validate that the folder path exists and is a directory.
        if (!Files.exists(folder) || !Files.isDirectory(folder)) {
            throw new IllegalArgumentException("Invalid folder path: " + folderPath);
        }

        // Read all files and process their contents.
        try (Stream<Path> paths = Files.walk(folder)) {
            return paths.filter(Files::isRegularFile)
                    .map(this::readFileContent)
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());
        }
    }

    /**
     * Reads the content of a single file.
     *
     * @param filePath Path to the file.
     * @return Content of the file as a single string, or null if an error occurs.
     */
    private String readFileContent(Path filePath) {
        try {
            List<String> lines = Files.readAllLines(filePath, java.nio.charset.Charset.forName(CHARSET));
            return String.join(" ", lines);
        } catch (IOException e) {
            System.err.println("Failed to read file: " + filePath + ". Error: " + e.getMessage());
            return null;
        }
    }
}
