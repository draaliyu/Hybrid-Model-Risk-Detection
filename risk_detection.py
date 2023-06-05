import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.cm as cm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from itertools import cycle
import numpy as np
import time,os
from sklearn.decomposition import PCA

folders=['train confusion matrix','test confusion matrix','roc curves','pie chart incorrect training','pie chart incorrect testing',
         'pie chart correct training','pie chart correct testing','accuracy plots','prediction times','training metrics',
         'testing metrics','feature importance']
for folder in folders:
    os.makedirs(f'plots/{folder}',exist_ok=True)
def print_and_plot_model_performance(model, model_name, X_train, y_train, X_test, y_test, color_dict, predictions=None):
    # Define the range for each class
    class_ranges = {
    0: "NR",
    1: "LR",
    2: "MR",
    3: "MHR",
    4: "HR",
    5: "VHR",
    6: "EHR"}

    # Get the predictions from the model if they're not provided
    start_time = time.time()
    if predictions is None:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    else:
        y_train_pred, y_test_pred = predictions
    elapsed_time = time.time() - start_time

    # Add this line to get precision, recall, and F1-score
    train_precision, train_recall, train_fscore, _ = precision_recall_fscore_support(y_train, y_train_pred, average='weighted')
    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(y_test, y_test_pred,average='weighted')

    # Print classification report
    print(f"\n{model_name} Performance")
    print("Training Classification Report:")
    print(classification_report(y_train, y_train_pred))
    print("Testing Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Plotting Training confusion matrix
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Truth')
    plt.xticks(np.arange(len(class_ranges))+0.5,[f'{i} ({class_ranges[i]})' for i in class_ranges.keys()])
    plt.yticks(np.arange(len(class_ranges)) + 0.5, [f'{i} ({class_ranges[i]})' for i in class_ranges.keys()])
    ax1.set_title(f'{model_name} Training Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'./plots/train confusion matrix/{model_name}_train_confusion_matrix.png')
    plt.clf()

    # Plotting Testing confusion matrix
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Truth')
    plt.xticks(np.arange(len(class_ranges)) + 0.5, [f'{i} ({class_ranges[i]})' for i in class_ranges.keys()])
    plt.yticks(np.arange(len(class_ranges)) + 0.5, [f'{i} ({class_ranges[i]})' for i in class_ranges.keys()])
    ax2.set_title(f'{model_name} Testing Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'./plots/test confusion matrix/{model_name}_testing_confusion_matrix.png')
    plt.clf()

    # ROC Curve
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)
    y_test_pred_bin = lb.transform(y_test_pred)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(6, 4))
    colors = cycle(['yellow', 'darkorange', 'cornflowerblue', 'Red', 'Blue', 'Purple','pink'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} ({2}) (area = {1:0.3f})'.format(i, roc_auc[i], class_ranges[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROCs')
    plt.legend(loc="lower right")
    plt.savefig(f'./plots/roc curves/{model_name}_roc_curve.png')
    plt.clf()

    # Plotting the pie chart for correctly and incorrectly classified instances
    train_correct_counts = y_train[y_train_pred == y_train].value_counts()
    train_incorrect_counts = y_train[y_train_pred != y_train].value_counts()

    test_correct_counts = y_test[y_test_pred == y_test].value_counts()
    test_incorrect_counts = y_test[y_test_pred != y_test].value_counts()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.pie(train_correct_counts, labels=None, colors=[color_dict[i] for i in train_correct_counts.index],
            autopct=lambda p: '{:.0f}'.format(p * sum(train_correct_counts) / 100), pctdistance=0.75)
    ax1.legend([f'{i} ({class_ranges[i]})' for i in train_correct_counts.index], title="Classes", loc="center left",
               bbox_to_anchor=(0, 0, 0.5, 0))
    ax1.set_title(f'{model_name} - Correctly Predicted Instances (Training)')
    plt.tight_layout()
    plt.savefig(f'./plots/pie chart correct training/{model_name}_train_correct_classification_pie_chart.png')
    plt.clf()


    fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.pie(test_correct_counts, labels=None, colors=[color_dict[i] for i in test_correct_counts.index],
            autopct=lambda p: '{:.0f}'.format(p * sum(test_correct_counts) / 100), pctdistance=0.75)
    ax2.legend([f'{i} ({class_ranges[i]})' for i in test_correct_counts.index], title="Classes", loc="center left",
               bbox_to_anchor=(0, 0, 0.5, 0))
    ax2.set_title(f'{model_name} - Correctly Predicted Instances (Testing)')
    plt.tight_layout()
    plt.savefig(f'./plots/pie chart correct testing/{model_name}_test_correct_classification_pie_chart.png')

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.pie(train_incorrect_counts, labels=None, colors=[color_dict[i] for i in train_incorrect_counts.index],
            autopct=lambda p: '{:.0f}'.format(p * sum(train_incorrect_counts) / 100), pctdistance=0.75)
    ax1.legend([f'{i} ({class_ranges[i]})' for i in train_incorrect_counts.index], title="Classes", loc="center left",
               bbox_to_anchor=(0, 0, 0.5, 0))
    ax1.set_title(f'{model_name} - Incorrectly Predicted Instances (Training)')
    plt.tight_layout()
    plt.savefig(f'./plots/pie chart incorrect training/{model_name}_train_incorrect_classification_pie_chart.png')
    plt.clf()

    fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.pie(test_incorrect_counts, labels=None, colors=[color_dict[i] for i in test_incorrect_counts.index],
            autopct=lambda p: '{:.0f}'.format(p * sum(test_incorrect_counts) / 100), pctdistance=0.75)
    ax2.legend([f'{i} ({class_ranges[i]})' for i in test_incorrect_counts.index], title="Classes", loc="center left",
               bbox_to_anchor=(0, 0, 0.5, 0))
    ax2.set_title(f'{model_name} - Incorrectly Predicted Instances (Testing)')
    plt.tight_layout()
    plt.savefig(f'./plots/pie chart incorrect testing/{model_name}_test_incorrect_classification_pie_chart.png')
    plt.clf()
    return y_train_pred, y_test_pred, elapsed_time, train_precision, train_recall, train_fscore, test_precision, test_recall, test_fscore

def plot_accuracies(models, train_accuracies, test_accuracies):
    # Bar width
    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(train_accuracies))
    r2 = [x + barWidth for x in r1]
    plt.figure(figsize=(10, 6))
    # Make the plot
    train_bars = plt.bar(r1, train_accuracies, color='b', width=barWidth, edgecolor='grey', label='Training')
    test_bars = plt.bar(r2, test_accuracies, color='r', width=barWidth, edgecolor='grey', label='Testing')
    plt.xlabel('Model Name', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xticks([r + barWidth/2 for r in range(len(train_accuracies))], models)

    for bar in train_bars:
        yval = round(bar.get_height(),3)
        plt.text(bar.get_x()+bar.get_width()/2, yval+0.01, yval, ha='center',va='bottom')
    for bar in test_bars:
        yval=round(bar.get_height(),3)
        plt.text(bar.get_x() + bar.get_width()/2, yval+0.01,yval, ha='center',va='bottom')
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.05),fancybox=True,shadow=True,ncol=5)
    plt.title('Training and Testing Accuracy for each Model')
    plt.savefig('./plots/accuracy plots/classifier_accuracy.png')
    plt.clf()

def run_model(train_file_path, test_file_path, feature_columns, label_column):
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
    dtc = DecisionTreeClassifier(random_state=0, max_depth=3)
    rfc = RandomForestClassifier(max_depth=3, random_state=0)
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    nb = GaussianNB()

    # List of the base models
    level0 = list()
    level0.append(('dtc', dtc))
    level0.append(('rfc', rfc))
    level0.append(('xgb', lr))
    level0.append(('knn', knn))
    level0.append(('nn', nb))
    # Define the stacking ensemble
    hybrid = StackingClassifier(estimators=level0, final_estimator=lr, cv=5)

    # Train models
    dtc.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    hybrid.fit(X_train, y_train)

    # Define the colors for each class
    colors = cycle(['yellow', 'darkorange', 'cornflowerblue', 'Red', 'Blue', 'Purple','pink'])
    color_dict = {class_label: color for class_label, color in zip(y_test.unique(), colors)}

    # Print and plot model performance
    dtc_train_pred, dtc_test_pred, dtc_elapsed_time, dtc_train_precision, dtc_train_recall, dtc_train_fscore, dtc_test_precision, dtc_test_recall, dtc_test_fscore = print_and_plot_model_performance(
        dtc, "DTC", X_train, y_train, X_test, y_test, color_dict)
    rfc_train_pred, rfc_test_pred, rfc_elapsed_time, rfc_train_precision, rfc_train_recall, rfc_train_fscore, rfc_test_precision, rfc_test_recall, rfc_test_fscore = print_and_plot_model_performance(
        rfc, "RFC", X_train, y_train, X_test, y_test, color_dict)
    lr_train_pred, lr_test_pred, lr_elapsed_time, lr_train_precision, lr_train_recall, lr_train_fscore, lr_test_precision, lr_test_recall, lr_test_fscore = print_and_plot_model_performance(
        lr, "LogisticRegression", X_train, y_train, X_test, y_test, color_dict)
    knn_train_pred, knn_test_pred, knn_elapsed_time, knn_train_precision, knn_train_recall, knn_train_fscore, knn_test_precision, knn_test_recall, knn_test_fscore = print_and_plot_model_performance(
        knn, "KNN", X_train, y_train, X_test, y_test, color_dict)
    nb_train_pred, nb_test_pred, nb_elapsed_time, nb_train_precision, nb_train_recall, nb_train_f_score, nb_test_precision, nb_test_recall, nb_test_fscore = print_and_plot_model_performance(
    nb, "Naive Bayes", X_train, y_train, X_test, y_test, color_dict)
    hybrid_train_pred, hybrid_test_pred, hybrid_elapsed_time, hybrid_train_precision, hybrid_train_recall, hybrid_train_fscore, hybrid_test_precision, hybrid_test_recall, hybrid_test_fscore = print_and_plot_model_performance(
    hybrid, 'Hybrid Model', X_train, y_train, X_test, y_test, color_dict)

    #Plotting prediction times
    model_names = ['DTC','RFC','LogisticRegression','KNN','Naive Bayes','Hybrid']
    prediction_times = [dtc_elapsed_time,rfc_elapsed_time,lr_elapsed_time,knn_elapsed_time,nb_elapsed_time,hybrid_elapsed_time]
    plt.figure(figsize=(10,6))
    colors = ['blue','orange','green','red','purple','pink']
    bars = plt.bar(model_names, prediction_times,color=colors)
    plt.xlabel('Model Name')
    plt.ylabel('Prediction Time (in seconds)')
    plt.title('Prediction times for each model')
    plt.grid(True)
    for bar in bars:
        yval = round(bar.get_height(),5)
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), yval, ha='center',va='bottom')
    plt.savefig('./plots/prediction times/model_prediction_times.png')
    plt.clf()

    #Plot training precision, recall, and F1-score
    barwidth = 0.3
    r1 = np.arange(len(model_names))
    r2 = [x + barwidth for x in r1]
    r3 = [x + barwidth for x in r2]
    model_train_metrics = [[dtc_train_precision,rfc_train_precision,lr_train_precision,knn_train_precision,nb_train_precision,hybrid_train_precision],
                     [dtc_train_recall,rfc_train_recall,lr_train_recall,knn_train_recall,nb_train_recall,hybrid_train_recall],
                     [dtc_train_fscore,rfc_train_fscore,lr_train_fscore,knn_train_fscore,nb_train_recall,hybrid_train_recall]]
    plt.figure(figsize=(10,6))
    #Bar for precision , recall, and f1-score
    bars1 = plt.bar(r1, model_train_metrics[0],color='blue',width=barwidth,edgecolor='grey',label='Precision')
    bars2 = plt.bar(r2, model_train_metrics[1], color='orange', width=barwidth, edgecolor='grey', label='Recall')
    bars3 = plt.bar(r3, model_train_metrics[2], color='green', width=barwidth, edgecolor='grey', label='F1-score')
    plt.xlabel('Model Name',fontweight='bold')
    plt.ylabel('Scores')
    plt.xticks([r + barwidth for r in range(len(model_train_metrics[0]))], model_names)
    plt.legend()
    plt.title('Precision, Recall, and F1-score for each model')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.05),fancybox=True, shadow=True, ncol=5)
    plt.grid(True)
    #Loop over the bars, adjust theheight to reflect the data value
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval=round(bar.get_height(),2)
            plt.text(bar.get_x()+bar.get_width()/2.0,yval,yval,va='bottom',ha='center',color='black')
    plt.savefig('./plots/training metrics/model_training_metrics.png')
    plt.clf()

    # Plot testing precision, recall, and F1-score
    barwidth = 0.3
    r4 = np.arange(len(model_names))
    r5 = [x + barwidth for x in r4]
    r6 = [x + barwidth for x in r5]
    model_test_metrics = [
        [dtc_test_precision,rfc_test_precision,lr_test_precision,knn_test_precision,nb_test_precision,hybrid_test_precision],
        [dtc_test_recall,rfc_test_recall,lr_test_recall,knn_test_recall,nb_test_recall,hybrid_test_recall],
        [dtc_test_fscore,rfc_test_fscore,lr_test_fscore,knn_test_fscore,nb_test_fscore,hybrid_test_fscore]]
    plt.figure(figsize=(10, 6))
    # Bar for precision , recall, and f1-score
    bars4 = plt.bar(r4, model_test_metrics[0], color='blue', width=barwidth, edgecolor='grey', label='Precision')
    bars5 = plt.bar(r5, model_test_metrics[1], color='orange', width=barwidth, edgecolor='grey', label='Recall')
    bars6 = plt.bar(r6, model_test_metrics[2], color='green', width=barwidth, edgecolor='grey', label='F1-score')
    plt.xlabel('Model Name', fontweight='bold')
    plt.ylabel('Scores')
    plt.xticks([r + barwidth for r in range(len(model_test_metrics[0]))], model_names)
    plt.legend()
    plt.title('Precision, Recall, and F1-score for each model')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.grid(True)
    # Loop over the bars, adjust the height to reflect the data value
    for bars in [bars4, bars5, bars6]:
        for bar in bars:
            yval = round(bar.get_height(), 2)
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, yval, va='bottom', ha='center', color='black')
    plt.savefig('./plots/testing metrics/model_testing_metrics.png')
    plt.clf()

    #Feature Importnace
    importances = rfc.feature_importances_

    for feature, importance in zip(feature_columns, importances):
        print(f'Features: {feature}, Importance: {importance}')

    # Sort the feature importnaces in descending order and convert to list
    sorted_indices = list(np.argsort(importances)[::-1])
    # Get a color map
    cmap = cm.get_cmap('rainbow')
    # Generate array of colors
    colors = cmap(np.linspace(0, 1, X_train.shape[1]))
    plt.figure(figsize=(10, 4))
    # Create a bar plot with colors
    plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center', color=colors, width=0.5)
    # Replace the xticks with feature names
    plt.xticks(range(X_train.shape[1]), [feature_columns[i] for i in sorted_indices], rotation='horizontal')
    # Axis labels and title
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel('Importance',fontweight='bold')
    plt.grid(True)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('./plots/feature importance/feature_importance.png')
    plt.clf()

    # Get train and test accuracies
    dtc_train_accuracy = dtc.score(X_train, y_train)
    rfc_train_accuracy = rfc.score(X_train, y_train)
    lr_train_accuracy = lr.score(X_train, y_train)
    knn_train_accuracy = knn.score(X_train, y_train)
    nb_train_accuracy = nb.score(X_train, y_train)
    hybrid_train_accuracy = hybrid.score(X_train, y_train)

    dtc_test_accuracy = dtc.score(X_test, y_test)
    rfc_test_accuracy = rfc.score(X_test, y_test)
    lr_test_accuracy = lr.score(X_test, y_test)
    knn_test_accuracy = knn.score(X_test, y_test)
    nb_test_accuracy = nb.score(X_test, y_test)
    hybrid_test_accuracy = hybrid.score(X_test, y_test)

    train_accuracies = [dtc_train_accuracy, rfc_train_accuracy, lr_train_accuracy, knn_train_accuracy,
                        nb_train_accuracy, hybrid_train_accuracy]
    test_accuracies = [dtc_test_accuracy, rfc_test_accuracy, lr_test_accuracy, knn_test_accuracy, nb_test_accuracy,
                       hybrid_test_accuracy]
    plot_accuracies(model_names, train_accuracies, test_accuracies)

    return dtc_test_accuracy,rfc_test_accuracy,lr_test_accuracy,knn_test_accuracy,nb_test_accuracy,hybrid_test_accuracy


#Specify the CSV file path, feature columns, and label column
train_file_path = 'Yobe-aug_temperature.csv'
test_file_path = 'yobe_temperature_test.csv'
feature_columns = ['Wind speed (m/s)', 'Relative Humidity (%)', 'Pressure']
label_column = 'T_label'

#Run the models and evaluate performance
results_df = run_model(train_file_path, test_file_path, feature_columns, label_column)
