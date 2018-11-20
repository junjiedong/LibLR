//
// Created by Junjie Dong on 11/18/18.
//

#ifndef __UTILS_H_
#define __UTILS_H_

#include <armadillo>

class Utils {

public:
    static arma::mat sigmoid(arma::mat logits);

    static double accuracy(arma::colvec y_true, arma::colvec y_predict);

    static arma::colvec cross_entropy_loss(arma::colvec y_true, arma::colvec y_hat);
};


#endif //__UTILS_H_
