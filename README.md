# Search Algorithms

[HTML Version Live Example](https://showhub.uosweb.co.uk/examples/search-algorithms.html)

This repository contains an assignment that explores the implementation and comparison of various search algorithms, Bayesian networks, and machine learning models using Python. The programs demonstrate how these techniques can be applied to solve complex problems and make predictions based on data.

## Contents

1. **Search Algorithms and Implementation**
   - **Uniform Cost Search:** This algorithm finds the least expensive path from a starting node to a goal node by using a priority queue. It explores nodes based on path costs, reconstructing the path if the goal is reached.
   - **Greedy Search:** Prioritises reaching the goal quickly by selecting paths with the lowest heuristic values. It is not always optimal, as it may overlook longer paths that lead to better solutions.
   - **A* Search:** Combines Uniform Cost and Greedy Search by evaluating both path cost and heuristic estimates. It efficiently finds the optimal path by expanding nodes with the lowest combined cost.

2. **Bayesian Networks**
   - **Probabilistic Graphical Models:** Represents a set of variables and their conditional dependencies using a directed acyclic graph. The model includes demographic and socio-economic variables to predict travel behavior.
   - **Model Structure Learning:** Utilizes Hill-Climbing and K2Score algorithms to learn the network structure and evaluate model performance using the F1-score. Compares learned models with expert-based models using BIC scores.

3. **Machine Learning**
   - **Naive Bayes Classification:** Processes large datasets for sentiment analysis, using text cleaning and transformation techniques. Implements grid search to optimize hyperparameters and evaluate model performance.
   - **Logistic Regression:** Provides a comparison to Naive Bayes by training on the same data and evaluating accuracy. Demonstrates slightly better performance in sentiment analysis.
   - **Model Evaluation:** Uses confusion matrices to visualize predictions and assess model performance, highlighting the importance of comparing multiple models to identify the best approach.

## Project Highlights

- **Search Algorithms:** Demonstrates different search strategies and their applications in finding optimal paths in a network of cities.
- **Bayesian Networks:** Uses probabilistic models to understand relationships between variables and make informed predictions based on data.
- **Machine Learning Models:** Compares Naive Bayes and Logistic Regression for sentiment analysis, emphasizing the significance of model evaluation and optimization.

## Conclusion

The programs showcases the implementation of search algorithms, Bayesian networks, and machine learning models to address complex problems in AI and data science. By evaluating and comparing different approaches, it highlights the importance of selecting the appropriate method for a given dataset and problem domain.

## How to Use

1. Clone this repository to your local machine.
2. Ensure you have Python and the necessary libraries installed.
3. Run the Python scripts provided in each section to see the implementation and results.

## Requirements

- Python 3.x
- Libraries: NumPy, Pandas, Scikit-learn, Matplotlib

Feel free to explore the code and contribute to improve the models or add new features!

##### Copyright: [Robertas Kirkickas](https://github.com/RobertasKirkickas)
