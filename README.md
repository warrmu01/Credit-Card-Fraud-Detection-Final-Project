Credit Card Fraud Detection
Overview
This project aims to detect fraudulent credit card transactions using various machine learning algorithms and techniques. The goal is to achieve the highest possible accuracy, precision, recall, and F1 score by comparing traditional machine learning algorithms, ensemble learning methods, and deep neural networks (DNNs).

Performance Expectations
Based on previous works using the same dataset, the following accuracy results were observed:

99.950% (DNN) - Deep Learning for Credit Card Fraud Detection
99.970% (XGB Classifier) - Credit Card Fraud Detection - Evaluation ROC-AUC
99.979% (XGB Classifier) - Unmasking Deception: Innovations in Credit Card Fraud Detection
100% (RandomForestClassifier) - Credit Card Fraud Detection
Given these high performance benchmarks, our goal was to achieve similar levels of accuracy with our models.

Final Analysis and Conclusion
Traditional Algorithms Comparison (Project 1)
Random Forest

Accuracy: 0.99984
Precision: 0.99968
Recall: 1.0
F1 Score: 0.99984
K-Nearest Neighbors (KNN)

Accuracy: 0.99971
Precision: 1.0
Recall: 0.82051
F1 Score: 0.90141
Decision Tree

Accuracy: 0.99833
Precision: 0.99751
Recall: 0.99916
F1 Score: 0.99833
Ensemble Learning Methods
AdaBoostClassifier

Accuracy: 0.999859
GradientBoostingClassifier

Accuracy: 0.999825
Deep Neural Networks (DNNs)
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
Summary
Best Accuracy: Deep Neural Networks (1.00)
Best Precision: Deep Neural Networks (1.00)
Best Recall: Deep Neural Networks (1.00)
Best F1 Score: Deep Neural Networks (1.00)
Conclusion
Best Overall Performance: DNNs with perfect scores across all metrics.
Best Traditional Algorithm: Random Forest, consistently performing the best across all metrics.
Best Ensemble Method: AdaBoostClassifier, with the highest accuracy among ensemble methods.
Our results confirm that high accuracy can be achieved with various machine learning techniques, aligning with findings from other research on this dataset.

How to Use This Project
Prerequisites
Python 3.11
Jupyter Notebook
Required libraries: pandas, numpy, scikit-learn, tensorflow, xgboost
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/creditcard_fraud_detection.git
cd creditcard_fraud_detection
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Running the Notebook
Open the Jupyter Notebook:

bash
Copy code
jupyter notebook creditcard_final_comparison.ipynb
Execute the cells to reproduce the results and visualizations.

Data
The dataset used for this project can be found on Kaggle: Credit Card Fraud Detection

Contributions
Feel free to submit pull requests or report issues.

License
This project is licensed under the MIT License.
