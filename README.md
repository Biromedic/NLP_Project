# Metrics Visualization for Classification

This project visualizes the performance metrics of a classification model using a bar chart. It leverages the **JFreeChart** library to create visualizations for **Precision**, **Recall**, and **F1-Score** for each class, including Macro Averages.

## Features
- **Visual Metrics**: Displays a bar chart with the precision, recall, and F1-score for each class.
- **Macro Averages**: Includes macro average values for precision, recall, and F1-score.
- **Interactive Visualization**: The chart is displayed in a separate window for easier interaction.

## Prerequisites
Before running the project, ensure you have the following installed:
1. **Java Development Kit (JDK)**: Version 8 or later.
2. **JFreeChart Library**
3. **Zemberek Library**: Add zemberek via [link](https://github.com/ahmetaa/zemberek-nlp) to your project

## Input Data
The metrics used in the chart are hardcoded for this project:
- Classes: **Neutral**, **Negative**, **Positive**
- Metrics:
  - Precision: `Neutral: 0.588, Negative: 0.861, Positive: 0.712`
  - Recall: `Neutral: 0.805, Negative: 0.664, Positive: 0.657`
  - F1-Score: `Neutral: 0.680, Negative: 0.750, Positive: 0.684`
  - Macro Averages:
    - Precision: `0.720`
    - Recall: `0.709`
    - F1-Score: `0.704`

The chart displays **Precision**, **Recall**, and **F1-Score** for each class along with the Macro Averages.

![Metrics Visualization](https://github.com/Biromedic/NLP_Project/blob/main/chart.png)