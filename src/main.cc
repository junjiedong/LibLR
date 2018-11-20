//
// Created by Junjie Dong on 11/18/18.
//

#include <iostream>
#include <armadillo>
#include "LogisticRegression.h"
#include "Utils.h"

using namespace arma;
using namespace std;

void testLR(int num_features, int num_train_examples, int num_val_examples, double noise_variance,
                int num_epoch, int batch_size, double learning_rate, double lambda) {
    // Randomly generate training set and validation set inputs
    mat trainX(num_train_examples, num_features);
    mat valX(num_val_examples, num_features);
    trainX.randn();
    valX.randn();

    // Randomly pick some 'ground-truth' weights and bias
    colvec weights(num_features);
    weights.randn();
    double bias = randn();

    // Generate the ground-truth weights by mixing signal and noise
    colvec trainY(num_train_examples);
    colvec valY(num_val_examples);
    for (int i = 0; i < num_train_examples; i++) {
        double logit = as_scalar(trainX.row(i) * weights) + bias + noise_variance * randn();
        if (logit > 0) trainY(i) = 1;
        else trainY(i) = 0;
    }
    for (int i = 0; i < num_val_examples; i++) {
        double logit = as_scalar(valX.row(i) * weights) + bias + noise_variance * randn();
        if (logit > 0) valY(i) = 1;
        else valY(i) = 0;
    }

    // Fit a logistic regression model
    LogisticRegression lr(num_features);
    lr.train(trainX, trainY, learning_rate, lambda, batch_size, num_train_examples / batch_size * num_epoch);

    // Evaluate performance on the validation set
    cout << endl << "Validation accuracy: " << Utils::accuracy(valY, lr.predict(valX)) << endl;
    lr.saveWeights("weights.txt");
}


int main(int argc, char **argv) {
    // Dataset options
    int num_features = 100;
    int num_train_examples = 50000;
    int num_val_examples = 10000;
    double noise_variance = 1;

    // Model options
    int num_epoch = 40;
    int batch_size = 500;
    double learning_rate = 0.02;
    double lambda = 0.005;

    testLR(num_features, num_train_examples, num_val_examples, noise_variance, num_epoch, batch_size, learning_rate, lambda);

    return 0;
}
