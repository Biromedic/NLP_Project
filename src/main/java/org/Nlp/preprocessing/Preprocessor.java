package org.Nlp.preprocessing;

import zemberek.tokenization.*;
import zemberek.morphology.*;
import zemberek.morphology.analysis.*;

import java.util.*;

public class Preprocessor {
    private final Set<String> stopWords;
    private final TurkishMorphology morphology;
    private final TurkishTokenizer tokenizer;

    /**
     * Default Turkish stopword list
     */
    private static final Set<String> DEFAULT_STOP_WORDS = Set.of(
            "acaba", "altı", "ama", "ancak", "arada", "artık", "aslında",
            "ayrıca", "az", "bana", "bazen", "bazı", "belki", "ben", "benden",
            "beni", "benim", "beri", "bile", "bin", "bir", "biraz", "birçok",
            "biz", "bizden", "bize", "bizi", "bizim", "bu", "bunu", "bunun",
            "burada", "bütün", "çok", "çünkü", "da", "daha", "de", "defa",
            "diğer", "diye", "elbette", "en", "fakat", "gibi", "göre", "hem",
            "her", "herhangi", "herkes", "hiç", "hiçbir", "ile", "ise", "işte",
            "kadar", "karşı", "mı", "mi", "mu", "mü", "nasıl", "ne", "neden",
            "nerede", "niçin", "için", "icin",  "o", "olan", "olarak", "oldu", "olduğu", "olmak",
            "olmayan", "olmaz", "olsun", "on", "önce", "sadece", "sana", "sen",
            "seni", "şey", "şöyle", "şu", "tarafından", "ve", "veya", "ya", "yani", "illa"
    );

    /**
     * Constructor for Preprocessor.
     * @param customStopWords Additional stop words to merge with the default list.
     */
    public Preprocessor(Set<String> customStopWords) {
        this.stopWords = new HashSet<>(DEFAULT_STOP_WORDS);
        if (customStopWords != null) {
            this.stopWords.addAll(customStopWords);
        }
        this.morphology = TurkishMorphology.createWithDefaults();
        this.tokenizer = TurkishTokenizer.DEFAULT;
    }

    /**
     * Tokenizes the input text.
     *
     * @param text Input text.
     * @return List of tokens.
     */
    public List<String> tokenize(String text) {
        if (text == null || text.isBlank()) {
            return Collections.emptyList();
        }
        return tokenizer.tokenizeToStrings(text.toLowerCase(Locale.forLanguageTag("tr")));
    }

    /**
     * Removes punctuation and stopwords from the tokenized text.
     *
     * @param tokens List of tokens.
     * @return List of tokens without stopwords and punctuation.
     */
    public List<String> removeStopWordsAndPunctuation(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> filteredTokens = new ArrayList<>();
        for (String token : tokens) {
            if (!stopWords.contains(token) && token.matches("\\p{L}+")) { // Retain only words (no punctuation or numbers)
                filteredTokens.add(token);
            }
        }
        return filteredTokens;
    }

    /**
     * Applies stemming to a list of tokens.
     *
     * @param tokens List of tokens.
     * @return List of stemmed tokens.
     */
    public List<String> stemTokens(List<String> tokens) {
        List<String> stemmedTokens = new ArrayList<>();
        for (String token : tokens) {
            try {
                List<SingleAnalysis> analyses = morphology.analyzeAndDisambiguate(token).bestAnalysis();
                if (!analyses.isEmpty()) {
                    stemmedTokens.add(analyses.get(0).getStem());
                } else {
                    stemmedTokens.add(token); // If no stem found, keep the original
                }
            } catch (Exception e) {
                System.err.println("Stemming error for token: " + token + " - " + e.getMessage());
                stemmedTokens.add(token); // Add the original token in case of an error
            }
        }
        return stemmedTokens;
    }

    /**
     * Preprocesses the input text.
     *
     * @param text Input text.
     * @return Preprocessed tokens.
     */
    public List<String> preprocess(String text) {
        if (text == null || text.isBlank()) {
            return Collections.emptyList();
        }

        List<String> tokens = tokenize(text);
        List<String> filteredTokens = removeStopWordsAndPunctuation(tokens);
        return stemTokens(filteredTokens);
    }
}
