//
// Created by Junjie Dong on 11/18/18.
//

#include "LogisticRegression.h"
#include "Utils.h"

using namespace arma;
using namespace std;

LogisticRegression::LogisticRegression(int input_dimension) {
    this->input_dimension = input_dimension;

    // randomly initialize the weight vector
    theta.set_size(input_dimension);
    theta.randn(); // zero-mean unit-variance
    bias = 0; // initialize bias to be zero
}

LogisticRegression::~LogisticRegression() {}

void LogisticRegression::train(arma::mat X, arma::colvec Y, double learning_rate, double lambda, int batch_size, int num_iter) {
    if (X.n_cols != input_dimension) { // check input dimension
        cout << "Error: Input dimension mismatch!" << endl;
        return;
    }

    int num_examples = X.n_rows;
    if (num_examples != Y.n_rows) {
        cout << "Error: X and Y have different number of training examples!" << endl;
        return;
    }

    cout << "Starts training with learning_rate " << learning_rate << ", lambda " << lambda << ", batch_size " << batch_size  << endl;
    int evaluate_frequency = num_examples / batch_size + 1;
    for (int i = 1; i <= num_iter; i++) {
        // randomly sample 'batch_size' rows from X
        uvec index = randi<uvec>(batch_size, distr_param(0, num_examples - 1));
        mat X_sample = X.rows(index);
        colvec y_sample = Y.elem(index);

        // inference
        mat logits = X_sample * theta + bias;
        colvec y_hat = Utils::sigmoid(logits);
        colvec y_predict = (sign(logits) + 1) / 2;

        // perform mini-batch gradient descent update
        colvec y_diff = y_hat - y_sample;
        mat y_mask(batch_size, input_dimension);
        for (int j = 0; j < input_dimension; j++) {
            y_mask.col(j) = y_diff;
        }
        colvec gradient = mean(y_mask % X_sample, 0).t() + lambda * theta;
        theta = theta - learning_rate * gradient;
        bias = bias - learning_rate * mean(y_diff);

        // evaluate and log
        if (i % evaluate_frequency == 0 || i == num_iter) {
            double loss = mean(Utils::cross_entropy_loss(Y, Utils::sigmoid(X * theta + bias))) + 0.5 * lambda * sum(square(theta));
            double accuracy = Utils::accuracy(Y, predict(X));
            cout << "After iteration " << i << ": " << "loss " << loss << ", accuracy: " << accuracy << endl;
        }
    }

    cout << "Training completed!" << endl;
    cout << "Loss: " << mean(Utils::cross_entropy_loss(Y, Utils::sigmoid(X * theta + bias))) + 0.5 * lambda * sum(square(theta))
        << endl << "Accuracy: " << Utils::accuracy(Y, predict(X)) << endl;
}

arma::colvec LogisticRegression::getTheta() {
    return theta;
}

double LogisticRegression::getBias() {
    return bias;
}

arma::mat LogisticRegression::predict(arma::mat X) {
    return (sign(X * theta + bias) + 1) / 2;
}

void LogisticRegression::saveWeights(std::string file_name) {
    colvec weights(input_dimension + 1);
    weights.subvec(0, input_dimension-1) = theta;
    weights(input_dimension) = bias;
    weights.save(file_name, raw_ascii);
}

void LogisticRegression::loadWeights(std::string file_name) {
    colvec weights;
    weights.load(file_name, raw_ascii);
    theta = weights.subvec(0, input_dimension-1);
    bias = weights(input_dimension);
}
