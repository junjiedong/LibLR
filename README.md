# LibLR
A basic Python Logistic Regression package with C++ backend

This light-weight project that references [wepe](https://github.com/wepe)'s [dive-into-ml-systems](https://github.com/wepe/dive-into-ml-system) is intended for practicing implementing ML algorithms in C++ with linear algebra libraries, and calling dynamic library functions in Python using `ctypes`. In comparison to the tools used in this project, Google's [TensorFlow](https://github.com/tensorflow/tensorflow) framework relies heavily on linear algebra library `Eigen` in its C++ kernels, and uses [SWIG](https://github.com/swig/swig) to build the Python-C++ interface. 

## Dependencies
- C++: Armadillo
- Python: ctypes, numpy

## Implementations
- `src/LogisticRegression.h` and `src/LogisticRegression.cc` implements a simple Logistic Regression classifier with L2 regularization using `Armadillo`, a popular linear algebra library that provides easy-to-use Matlab-like APIs.
- `src/CWrapper.cc` exports C-style functions that will be packaged into a dynamic library `liblr.so`
- `python/LibLR/LogisticClassifier.py` defines a LogisticClassifier class which performs training and inference by calling the dynamic library functions using `ctypes`
- `src/main.cc` tests the C++ implementations by training and evaluating on randomly (but carefully) generated datasets, while `python/test.py` tests the Python `LogisticClassifier` implementations in the same way

## Steps
- Install Armadillo, and properly set up g++ linking options in `Makefile`
- Build dynamic library and move it into `python/LibLR` folder
```
make lib
```
- Train and evaluate the Python `LogisticClassifier`
```
python2.7 python/test.py
```
