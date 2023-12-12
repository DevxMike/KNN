KNN Gesture Recognition Code

This C++ code implements the k-nearest neighbors (KNN) algorithm for gesture recognition based on data provided in training and testing files.

Data Structures:

    record_t: Represents a single gesture record, consisting of a vector of features and a gesture ID.
    matrix_type: A matrix of gesture records.

File Input:
The program loads training and testing data from files ("training.dat" and "testing.dat" respectively). Each line in the files contains a gesture ID followed by feature values.

Distance Metrics:
The code supports two distance metrics for KNN—Euclidean and Manhattan distances. These metrics are used to measure the similarity between feature vectors.

KNN Prediction:
The knn_predict function predicts the gesture ID for a given set of features using KNN. It calculates distances between the input features and those in the training data, selects the k-nearest neighbors, and determines the majority class.

Evaluation Metrics:
The code includes functions to calculate accuracy, confusion matrix counts, and confusion matrix percentages for both Euclidean and Manhattan distances.

Main Function:
The main function loads data, calculates accuracy for various k values, and outputs results to the console and a file named "result.txt". It also generates confusion matrices for the worst and best cases in terms of accuracy for both distance metrics.

Usage:

    Compile the code using a C++ compiler.
    Ensure the existence of "training.dat" and "testing.dat" files with appropriate gesture data.
    Run the compiled executable to obtain accuracy metrics and confusion matrices.

Output:
Results are printed to the console and stored in "result.txt," including accuracy values for different k values, and confusion matrices for the worst and best cases using both Euclidean and Manhattan distances.

Note:
Ensure the correct file paths and data formats to execute the code successfully.
