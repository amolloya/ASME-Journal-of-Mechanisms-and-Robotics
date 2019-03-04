# Machine Learning Driven Individualized Gait Rehabilitation: Gait Classification, Prediction, and Mechanism Design

End-to-End computational approach for individualized gait rehabilitation using machine learning techniques for gait classification, prediction, and specialized device design.

The initial component of our End-to-End approach is the Gait Classifier Module, which analyzes the current gait of the individual on which various further actions are dependent. Gait classification is carried out using K-nearest neighbors, Support Vector Machines, Artificial Neural Network, and Random Forests to classify the gaits into healthy, Cerebellar Ataxia (CA), Hereditary Spastic Paraparesis (HSP), and Parkinson’s disease (PD). Leave one out and k-fold cross-validation methods are used to find out the performance of the models.
The following table gives the classification accuracies of the models with different cross-validations.

| Method  | LOO CV | 3-fold CV  | 5-fold CV | 7-fold CV | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| KNN  | 78.57%  | 71.04%  | 78.55%  | 78.99%  |
| SVM  | 88.23%  | 83.18%  | 86.55%  | 87.81%  |
| ANN  | 89.49%  | 83.58%  | 90.16%  | 88.65%  |
| RF   | 98.47%  | 93.60%  | 92.42%  | 95.79%  |
