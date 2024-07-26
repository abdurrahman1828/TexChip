import numpy as np
from mahotas.features import haralick
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    VotingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)


def extract(X, y, fold, data_name, split='train'):
    ML_data_x = X
    ML_data_y = y

    featuredata = pd.DataFrame(index=[],
                               columns=['ASM', 'Contrast', 'Correlation', 'SSV', 'IDM', 'SA', 'SV', 'SE', 'Entropy',
                                        'DV',
                                        'DE', 'IMC1', 'IMC2', 'MCC',
                                        'DfASM', 'DfContrast', 'DfCorrelation', 'DfSSV', 'DfIDM', 'DfSA', 'DfSV',
                                        'DfSE',
                                        'DfEntropy', 'DfDV', 'DfDE', 'DfIMC1', 'DfIMC2', 'DfMCC', 'label'])

    for i in range(len(ML_data_x)):
        temp = ML_data_x[i, :, :, :]
        temp = temp.astype('int16')
        fdata = haralick(temp, return_mean_ptp=True, compute_14th_feature=True)
        tempdf = pd.DataFrame(fdata.reshape((1, len(fdata))), index=[0],
                              columns=['ASM', 'Contrast', 'Correlation', 'SSV', 'IDM', 'SA', 'SV', 'SE', 'Entropy',
                                       'DV',
                                       'DE', 'IMC1', 'IMC2', 'MCC',
                                       'DfASM', 'DfContrast', 'DfCorrelation', 'DfSSV', 'DfIDM', 'DfSA', 'DfSV', 'DfSE',
                                       'DfEntropy', 'DfDV', 'DfDE', 'DfIMC1', 'DfIMC2', 'DfMCC'])
        tempdf['label'] = ML_data_y[i]
        featuredata = featuredata.append(tempdf)

    featuredata.to_csv(f'data/{data_name}_{fold}_{split}.csv')


def evaluate_ml(train, test, data_name, fold):
    print("Fold: ", fold)
    # Assume the last column is the label and the rest are features
    X_train = train.iloc[:, :-1]
    X_test = test.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    y_test = test.iloc[:, -1]

    # List of classifiers
    classifiers = [
        ('Random Forest', RandomForestClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('SVM', SVC(kernel='rbf', probability=True)),
        ('Logistic Regression', LogisticRegression()),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Neural Network', MLPClassifier()),
        ('AdaBoost', AdaBoostClassifier()),
        ('Bagging', BaggingClassifier()),
        ('Extra Trees', ExtraTreesClassifier()),
        ('Voting Classifier', VotingClassifier(estimators=[
            ('Random Forest', RandomForestClassifier()),
            ('Extra Trees', ExtraTreesClassifier()),
            ('Gradient Boosting', GradientBoostingClassifier()),
        ], voting='soft'))
    ]

    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

    # Iterate over classifiers
    for model_name, model in classifiers:
        # Fit the model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Append results to DataFrame
        results_df = results_df.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, ignore_index=True)

    # Save results to a CSV file
    results_file_path = f'results/{data_name}_Fold_{fold}.csv'  # Replace with your desired file path
    results_df.to_csv(results_file_path, index=False)

    # Display results
    print(results_df)
