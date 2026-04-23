import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mrmr import mrmr_classif
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier
# from itertools import combinations

def classify(X,y):

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred)
    rf_conf_mat = confusion_matrix(y_test, rf_pred)
    conf_matrix_percentage = normalize(rf_conf_mat, axis=1, norm='l1') * 100

    # Generate LaTeX table
    latex_confusion_matrix = tabulate(conf_matrix_percentage, tablefmt="latex", stralign='center', numalign='center')

    # Print or save the LaTeX confusion matrix
    #print('LaTex confusion matrix:\n',latex_confusion_matrix)

    print("Random Forest Accuracy:", rf_accuracy)
    #print('Classification Report:\n', rf_report)

    ax= plt.subplot()
    #plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
    # Add labels 
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')

    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Room 2 classification confusion matrix'); 
    ax.xaxis.set_ticklabels(['empty', 'sitting', 'standing', 'walking']); ax.yaxis.set_ticklabels(['empty', 'sitting', 'standing', 'walking']);



def feat_selection(X,y,k):

    # Create a pandas DataFrame from the NumPy matrix
    #dataset
    X = pd.DataFrame(X)
    #class vector
    y = pd.Series(y)

    #specify column names, feature labelling
    feature_names = ['theta_int', 'max_int', 'theta_dist', 'max_dist', 'theta_flat', 'max_flat']
    X.columns = feature_names

    selected_features = mrmr_classif(X=X, y=y, K=k)
    print(selected_features)

    new_dataset = X[selected_features]

    new_X = new_dataset.values

    classify(new_X,y)



#load the dataset
loaded_dataset = np.load('saved_dataset/dataset_r2_w2_40.npy')
#realizations and associated classes vector extraction
X, y = loaded_dataset[:,[0,1,2,3,4,5]], loaded_dataset[:,6]

#feat_selection(X,y,6)
classify(X,y)

# Show the plot
plt.show()

feature_names = ['theta_int', 'max_int', 'theta_dist', 'max_dist', 'theta_flat', 'max_flat', 'label']

df = pd.DataFrame(loaded_dataset, columns=feature_names)

print(df.head())     # first 5 rows
print(df.shape)      # (rows, columns)
















############################################################################################################################################


    ########## feature selection - not good
    # Generate all combinations
    # all_combinations = []
    # for r in range(2, 6):
    #     combinations_r = list(combinations([0,1,2,3,4,5], r))
    #     all_combinations.extend(combinations_r)

    # for combo in all_combinations:
    #     X_selected = X[:,combo]

    #     # Scale features
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X_selected)

    #     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

    #     print(X_train.shape)


    #     # Random Forest
    #     rf_model = RandomForestClassifier()
    #     rf_model.fit(X_train, y_train)
    #     rf_pred = rf_model.predict(X_test)
    #     rf_accuracy = accuracy_score(y_test, rf_pred)
    #     rf_report = classification_report(y_test, rf_pred)
    #     rf_conf_mat = confusion_matrix(y_test, rf_pred)
    #     conf_matrix_percentage = normalize(rf_conf_mat, axis=1, norm='l1') * 100
    #     print('Selected features: ', combo)
    #     print("Random Forest Accuracy:", rf_accuracy)
    #     #print('Classification Report:\n', rf_report)


    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=False)

    #     # Add labels and a title
    #     plt.xlabel('Predicted Labels')
    #     plt.ylabel('True Labels')
    #     plt.title(f"Confusion Matrix(Percentage), features: {combo}")

    # # feature extraction
    # model = ExtraTreesClassifier(n_estimators=10)
    # model.fit(X, y)
    # print(model.feature_importances_)

    # #Show the plot
    # plt.show()



#################### Other classifiers ####################

    # #Naive Bayes
    # # Initialize classifier:
    # gnb = GaussianNB()
    # # Train the classifier:
    # model = gnb.fit(X_train, y_train)
    # # Make predictions with the classifier:
    # y_pred = gnb.predict(X_test)
    # # Evaluate label (subsets) accuracy:
    # accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    # print('Naive Bayes Accuracy:', accuracy)
    # #print('Classification Report:\n', report)
    

    # # Logistic Regression
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # reg_report = classification_report(y_test, y_pred)
    # print('Logistic Regression Accuracy:', accuracy)
    # #print('Classification Report:\n', reg_report)


    # # Neural Network (MLP)
    # mlp_model = MLPClassifier()
    # mlp_model.fit(X_train, y_train)
    # mlp_pred = mlp_model.predict(X_test)
    # mlp_accuracy = accuracy_score(y_test, mlp_pred)
    # mlp_report = classification_report(y_test, mlp_pred)
    # print("Neural Network Accuracy:", mlp_accuracy)
    # #print('Classification Report:\n', reg_report)


    # # Support Vector Machine
    # svm_model = SVC()
    # svm_model.fit(X_train, y_train)
    # svm_pred = svm_model.predict(X_test)
    # svm_accuracy = accuracy_score(y_test, svm_pred)
    # svm_report = classification_report(y_test, svm_pred)
    # print("SVM Accuracy:", svm_accuracy)
    # #print('Classification Report:\n', svm_report)


    # # K-Nearest Neighbors
    # knn_model = KNeighborsClassifier()
    # knn_model.fit(X_train, y_train)
    # knn_pred = knn_model.predict(X_test)
    # knn_accuracy = accuracy_score(y_test, knn_pred)
    # knn_report = classification_report(y_test, knn_pred)
    # print("KNN Accuracy:", knn_accuracy)
    # #print('Classification Report:\n', knn_report)