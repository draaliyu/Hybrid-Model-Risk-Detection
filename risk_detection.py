import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy.stats import mode
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import time
def print_and_plot_model_performance(model, model_name, X_test, y_test, color_dict,predictions=None):
    # Define the range for each class
    class_ranges = {
        0: "NR",    #No Risk
        1: "MinR",  #Minimal Risk
        2: "LR",    #Low Risk
        3: "ModR",  #Moderate Risk
        4: "HR",    #High Risk
        5: "SR",    #Severe Risk
        6: "ER"     #Extreme Risk
    }

    # Get the predictions from the model if they're not provided
    if predictions is None:
        y_pred = model.predict(X_test)
    else:
        y_pred = predictions

    # Print classification report
    print(f"\n{model_name} Performance")
    print(classification_report(y_test, y_pred))

    # Plotting confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.xticks(np.arange(len(class_ranges)) + 0.5, [f'{i} ({class_ranges[i]})' for i in class_ranges.keys()])
    plt.yticks(np.arange(len(class_ranges)) + 0.5, [f'{i} ({class_ranges[i]})' for i in class_ranges.keys()])
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'./results/{model_name}_confusion_matrix.png')

    # ROC Curve
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)
    y_pred_bin = lb.transform(y_pred)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    lw = 2
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','Red','Blue','Purple','Black'])
    # ROC Curve plotting section

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} ({2}) (area = {1:0.2f})'
                       ''.format(i, roc_auc[i], class_ranges[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic to Multi-Class')
    plt.legend(loc="lower right")
    plt.savefig(f'./results/{model_name}_roc_curve.png')

    #Plotting the pie chart for correctly and incorrectly classified instances
    correct_counts = y_test[y_pred == y_test].value_counts()
    incorrect_counts = y_test[y_pred != y_test].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(correct_counts, labels=None, colors=[color_dict[i] for i in correct_counts.index],
            autopct=lambda p: '{:.0f}'.format(p * sum(correct_counts) / 100), pctdistance=0.75)
    plt.legend([f'{i} ({class_ranges[i]})' for i in correct_counts.index], title="Classes", loc="center left",
               bbox_to_anchor=(0, 0, 0.5, 0))
    plt.title(f'{model_name} Number of correctly classified instances')
    plt.savefig(f'./results/{model_name}_correct_classification_pie_chart.png')

    plt.figure(figsize=(8, 8))
    plt.pie(incorrect_counts, labels=None, colors=[color_dict[i] for i in incorrect_counts.index],
            autopct=lambda p: '{:.0f}'.format(p * sum(incorrect_counts) / 100), pctdistance=0.75)
    plt.legend([f'{i} ({class_ranges[i]})' for i in incorrect_counts.index], title="Classes", loc="center left",
               bbox_to_anchor=(0, 0, 0.5, 0))
    plt.title(f'{model_name} Number of incorrectly classified instances')
    plt.savefig(f'./results/{model_name}_incorrect_classification_pie_chart.png')

    return y_pred


def hybrid_model(train_file_path, test_file_path, feature_columns, label_column):
    import time
    # Load the CSV dataset
    train_dataset = pd.read_csv(train_file_path)
    test_dataset = pd.read_csv(test_file_path, encoding='latin1')

    # Select features and target variable for training
    X_train = train_dataset[feature_columns]
    y_train = train_dataset[label_column]

    # Select features and target variable for testing
    X_test = test_dataset[feature_columns]
    y_test = test_dataset[label_column]

    # Initialize models
    dtc = DecisionTreeClassifier(random_state=42)
    rfc = RandomForestClassifier(random_state=42)
    lr = LogisticRegression(random_state=42)
    svc = SVC(probability=True, random_state=42)

    model_names = ["DecisionTreeClassifier", "RandomForestClassifier", "LogisticRegression", "SVC"]
    models = [dtc, rfc, lr, svc]
    training_times = []
    testing_times = []

    # Defining the colors for each class
    colors = ['lime', 'darkorange', 'cornflowerblue', 'olive', 'magenta', 'violet', 'Yellow']
    color_dict = {class_label: color for class_label, color in zip(y_test.unique(), colors)}

    # Train models and record training and testing time
    for model, name in zip(models, model_names):
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        training_times.append(training_time)

        start_time = time.time()
        print_and_plot_model_performance(model, name, X_test, y_test, color_dict)
        end_time = time.time()
        testing_time = end_time - start_time
        testing_times.append(testing_time)

    # Plot the training and testing times
    plt.figure(figsize=(8, 8))
    plt.pie(training_times,
            labels=[f"{model_name}\n{time:.2f} seconds" for model_name, time in zip(model_names, training_times)],
            labeldistance=0.5, textprops={'fontsize': 10, 'ha': 'center'})
    plt.title("Model Training Times")
    plt.savefig('./results/training_times_pie_chart.png')

    plt.figure(figsize=(8, 8))
    plt.pie(testing_times,
            labels=[f"{model_name}\n{time:.2f} seconds" for model_name, time in zip(model_names, testing_times)],
            labeldistance=0.5, textprops={'fontsize': 10, 'ha': 'center'})
    plt.title("Model Testing Times")
    plt.savefig('./results/testing_times_pie_chart.png')

    # Forming hybrid prediction using average of predicted probabilities
    average_probabilities = np.mean([dtc.predict_proba(X_test),
                                     rfc.predict_proba(X_test),
                                     lr.predict_proba(X_test),
                                     svc.predict_proba(X_test)], axis=0)
    hybrid_predictions = np.argmax(average_probabilities, axis=1)

    # Print and plot hybrid model performance
    import time
    start_time = time.time()
    print_and_plot_model_performance(None, "Hybrid", X_test, y_test, color_dict, hybrid_predictions)
    end_time = time.time()
    hybrid_testing_time = end_time - start_time
    testing_times.append(hybrid_testing_time)
    model_names.append("Hybrid")

    # Updating the pie chart for testing times
    plt.figure(figsize=(8, 8))
    plt.pie(testing_times,
            labels=[f"{model_name}\n{time:.2f} seconds" for model_name, time in zip(model_names, testing_times)],
            labeldistance=0.5, textprops={'fontsize': 10, 'ha': 'center'})
    plt.title("Model Testing Times including Hybrid Model")
    plt.savefig('./results/testing_times_pie_chart_updated.png')

    return hybrid_predictions


# Specify the CSV file path, feature columns, and label column
train_file_path = 'Borno-augmented.csv'
test_file_path = 'Borno.csv'
feature_columns = ['Wind speed (m/s)', 'Relative Humidity (%)', 'Pressure']
label_column = 'Label'

# Run the hybrid model
hybrid_predictions = hybrid_model(train_file_path, test_file_path, feature_columns, label_column)
