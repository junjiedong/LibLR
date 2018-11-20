//
// Created by Junjie Dong on 11/19/18.
//

#include "LogisticRegression.h"

using namespace arma;
using namespace std;


void gen_random(char *s, int len) {
    const static char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < len-1; i++) {
        s[i] = alphanum[randi(distr_param(0, sizeof(alphanum) - 1))];
    }
    s[len-1] = '\0';
}

extern "C" {
    void train(double** X, double* Y, int num_examples, int num_features,
            double learning_rate, double lambda, int batch_size, int num_iter, char* ret_model_path) {
        // Set up training data
        mat X_train(num_examples, num_features);
        colvec Y_train(num_examples);
        for (int i = 0; i < num_examples; i++) {
            Y_train(i) = Y[i];
            for (int j = 0; j < num_features; j++) {
                X_train(i, j) = X[i][j];
            }
        }

        // Set up LR classifier
        LogisticRegression LR(num_features);
        LR.train(X_train, Y_train, learning_rate, lambda, batch_size, num_iter);

        // Save model weights to temp file and return the temp file name
        char* rand_file_name = new char[20];
        gen_random(rand_file_name, 20);
        string tmp_model_path = "/tmp/" + string(rand_file_name);
        LR.saveWeights(tmp_model_path);
        strcpy(ret_model_path, tmp_model_path.c_str());
    }

    void predict(double** X, int* ret_val, int num_examples, int num_features, char* tmp_model_path) {
        mat X_mat(num_examples, num_features);
        for (int i = 0; i < num_examples; i++) {
            for (int j = 0; j < num_features; j++) {
                X_mat(i, j) = X[i][j];
            }
        }

        LogisticRegression LR(num_features);
        LR.loadWeights(string(tmp_model_path));
        mat y_predict = LR.predict(X_mat);

        for (int i = 0; i < num_examples; i++) {
            ret_val[i] = y_predict(i) > 0.5 ? 1 : 0;
        }
    }

    void say_something(char* text) {
        printf("%s\n", text);
    }
}
