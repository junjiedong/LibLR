//
// Created by Junjie Dong on 11/18/18.
//

#ifndef __LogisticRegression_H__
#define __LogisticRegression_H__

#include <string>
#include <armadillo>


class LogisticRegression {

    // Simple Logistic Regression Classifier with L2-regularization

public:
    LogisticRegression(int input_dimension);

    ~LogisticRegression();

    void train(arma::mat X, arma::colvec Y, double learning_rate, double lambda, int batch_size, int num_iter);

    arma::mat predict(arma::mat X);

    arma::colvec getTheta();

    double getBias();

    void saveWeights(std::string file_name);

    void loadWeights(std::string file_name);

private:
    int input_dimension;
    arma::colvec theta;  // LR Parameters
    double bias;  // LR bias term

};


#endif //__LogisticRegression_H__
