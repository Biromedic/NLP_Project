package org.Nlp;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.*;

public class MetricsChart {

    public static void main(String[] args) {
        // Define class names and metrics
        String[] classes = {"Neutral", "Negative", "Positive"};
        DefaultCategoryDataset dataset = getDefaultCategoryDataset(classes);

        // Generate the chart
        JFreeChart barChart = ChartFactory.createBarChart(
                "Metrics Visualization",
                "Class",
                "Value",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        // Display the chart
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        ChartPanel chartPanel = new ChartPanel(barChart);
        frame.add(chartPanel);
        frame.setVisible(true);
    }

    private static DefaultCategoryDataset getDefaultCategoryDataset(String[] classes) {
        double[] precision = {0.588055, 0.860887, 0.712034};
        double[] recall = {0.805031, 0.664075, 0.657407};
        double[] f1Score = {0.679646, 0.749781, 0.683631};
        double macroPrecision = 0.720326;
        double macroRecall = 0.708838;
        double macroF1 = 0.704353;

        // Create a dataset for precision, recall, and F1-score
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < classes.length; i++) {
            dataset.addValue(precision[i], "Precision", classes[i]);
            dataset.addValue(recall[i], "Recall", classes[i]);
            dataset.addValue(f1Score[i], "F1-Score", classes[i]);
        }

        // Add Macro Averages
        dataset.addValue(macroPrecision, "Precision", "Macro Average");
        dataset.addValue(macroRecall, "Recall", "Macro Average");
        dataset.addValue(macroF1, "F1-Score", "Macro Average");
        return dataset;
    }
}
