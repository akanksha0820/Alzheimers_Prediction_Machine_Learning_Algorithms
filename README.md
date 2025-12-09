Project Overview:
The project contains multiple machine learning models to develop a system for the early detection of Alzheimer's disease from MRI scan images and handwriting data. It includes deep learning models such as CNN, ensemble methods, KNN, and SVM solving issues such as class imbalance through data augmentation and providing model interpretability using explainable AI methods like Grad CAM and SHAP, which highlight region of the MRI scan contributing to the model's output, making the diagnosis process more accurate. The relationship between patient's handwriting data with the presence of Alzheimer's is also explored using XGBoost model with PCA for dimensionality reduction, creating a strong foundation to build a system that learns the interaction between multiple modalities and their influence on Alzheimer's disease.
This project implements and compares several machine learning and deep learning algorithms to classify MRI scans into four stages of Alzheimer’s disease:  
1. Non-Demented 
2. Mild Demented 
3. Moderate Demented 
4. Very Mild Demented
   
Goal:
This project developed a robust, multi-class system for diagnosing Alzheimer’s disease severity from MRI scans by rigorously comparing ML/DL algorithms (CNN, Ensembles) and enhancing model performance with data augmentation. 
The framework achieves clinical transparency through Explainable AI (Grad CAM/SHAP) and demonstrates the potential of a multimodal diagnostic pipeline integrating patient handwriting data.

Features & Highlights -
CNN (Convolutional Neural Network) for high-accuracy image classification (~91% test accuracy).  
Class balancing and data augmentation to handle imbalanced datasets and improve model robustness. 
Traditional ML models: SVM, KNN, and ensemble learning for comparative performance analysis. 
End-to-end ML workflow: data preprocessing, training, evaluation, and visualization. 
  

Tools & Libraries: Python, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn. 

Project Structure-
CNN.ipynb: Implements the foundational CNN model.
Data augmentation.ipynb: Implements the CNN model, augmenting new data samples by transforming the existing ones.
Ensemble-based learning.ipynb: Implements Ensemble-based learning methods such as RandomForest, Bagging, and XGBoost classifiers.
SHAP and SmoothGrad (CNN Model).ipynb: Performs SHAP analysis on the outputs of the foundational CNN model.
SHAP and SmoothGrad (Data augmentation).ipynb: Performs SHAP analysis on the outputs of the data-augmented CNN model
KNN-Classifier.ipynb: Implements the KNN-Classifier model.
SVM.ipynb: Implements the foundational SVM model.
Handwriting images.ipynb: Implements the XGBoost model along with PCA to predict the presence of Alzheimer’s based on patients’ handwriting data.


