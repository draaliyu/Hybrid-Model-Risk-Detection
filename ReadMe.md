
Hybrid Classification Model

This Python script builds, trains, and evaluates a hybrid model using several classifiers from the sklearn library, including a Decision Tree Classifier, Random Forest Classifier, Logistic Regression model, and Support Vector Classifier (SVC).
Dependencies

    Python 3
    pandas
    scikit-learn
    seaborn
    matplotlib
    numpy
    scipy

You can install all dependencies using the following command:
pip install pandas scikit-learn seaborn matplotlib numpy scipy


Usage

Prepare your training and testing datasets in CSV format. The dataset should include the columns for the features you're interested in and a label column for the target variable. This script is currently configured for a multi-class problem with seven classes, but you can adjust this to suit your needs.

Update the train_file_path, test_file_path, feature_columns, and label_column variables at the bottom of the script to point to your datasets and the columns you're interested in.

# Specify the CSV file path, feature columns, and label column
train_file_path = 'your_training_dataset.csv'
test_file_path = 'your_testing_dataset.csv'
feature_columns = ['feature_1', 'feature_2', 'feature_3']  # replace with your features
label_column = 'label'  # replace with your label column


Run the script. It will train each model on the training dataset, test them on the testing dataset, and print the classification report and confusion matrix for each model. It will also create a hybrid model that averages the probabilities predicted by each individual model, and print the performance of the hybrid model.

The script will save multiple plots to a folder named results in your current directory. These include the confusion matrix, ROC curve, and pie charts of correctly and incorrectly classified instances for each model, as well as pie charts of training and testing times.

Make sure the results folder exists in your current directory, or modify the script to save the results elsewhere.


Note

This script is currently configured to handle a multi-class classification problem with seven classes, named NR (No Risk), MinR (Minimal Risk), LR (Low Risk), ModR (Moderate Risk), HR (High Risk), SR (Severe Risk), and ER (Extreme Risk). If you have a different number of classes or different class names, you will need to adjust the class_ranges variable in the print_and_plot_model_performance function, as well as the color mappings in the hybrid_model function.
