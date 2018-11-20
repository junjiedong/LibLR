//
// Created by Junjie Dong on 11/18/18.
//

#include "Utils.h"
using namespace arma;

arma::mat Utils::sigmoid(arma::mat logits) {
    return 1 / (1 + exp(-logits));
}

double Utils::accuracy(arma::colvec y_true, arma::colvec y_predict) {
    int num_rows = y_true.n_rows;
    double correct_count = 0;
    for (int i = 0; i < num_rows; i++) {
        if ((y_true(i) > 0.5 && y_predict(i) > 0.5) || (y_true(i) < 0.5 && y_predict(i) < 0.5)) {
            ++correct_count;
        }
    }

    return correct_count / num_rows;
}

arma::colvec Utils::cross_entropy_loss(arma::colvec y_true, arma::colvec y_hat) {
    return -(y_true % log(y_hat) + (1 - y_true) % log(1 - y_hat));
}
