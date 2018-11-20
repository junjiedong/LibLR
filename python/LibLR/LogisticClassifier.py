from ctypes import *
import numpy as np
import os

dirPath = os.path.dirname(os.path.abspath(__file__))
dllPath = os.path.join(dirPath, "liblr.so")
LibLR = cdll.LoadLibrary(dllPath)
LibLR.say_something("LibLR loaded!")

LibLR.train.argtypes = [np.ctypeslib.ndpointer(dtype=np.uintp,ndim=1,flags='C'), POINTER(c_double), \
                        c_int, c_int, c_double, c_double, c_int, c_int, c_char_p]
LibLR.train.restype = None
LibLR.predict.argtypes = [np.ctypeslib.ndpointer(dtype=np.uintp,ndim=1,flags='C'), POINTER(c_int), \
                        c_int, c_int, c_char_p]
LibLR.predict.restype = None


class LogisticClassifier(object):
    def __init__(self, num_features):
        self.num_features = num_features
        self.tmp_model_path = None

    def train(self, X, Y, learning_rate=0.01, reg=0.0, batch_size=64, num_iter=100):
        # If inputs are arrays, convert them to numpy arrays
        X = np.asarray(X, dtype=np.double)
        Y = np.ascontiguousarray(np.asarray(Y, dtype=np.double), dtype=np.double)
        num_examples, num_cols = X.shape
        if num_cols != self.num_features:
            print "X does not have correct number of columns!"
            return
        if num_examples != Y.shape[0]:
            print "X and Y do not have the same number of examples!"
            return

        # Prepare Ctype function call arguments
        c_X = (X.ctypes.data + np.arange(num_examples) * X.strides[0]).astype(np.uintp)
        c_Y = cast(Y.ctypes.data, POINTER(c_double))
        ret_char = c_char_p("0" * 20)

        # Call LibLR
        LibLR.train(c_X, c_Y, c_int(num_examples), c_int(self.num_features), c_double(learning_rate), \
                    c_double(reg), c_int(batch_size), c_int(num_iter), ret_char)
        self.tmp_model_path = ret_char.value
        print "Model weights saved to:", self.tmp_model_path

    def predict(self, X):
        if self.tmp_model_path is None:
            print "Cannot make prediction without trained weights!"
            return None

        X = np.asarray(X, dtype=np.double)
        num_examples, num_cols = X.shape
        if num_cols != self.num_features:
            print "Feature dimension mismatch!"
            return None

        # Prepare Ctype function call arguments and call LibLR
        c_X = (X.ctypes.data + np.arange(num_examples) * X.strides[0]).astype(np.uintp)
        predict_results = cast((c_int * num_examples)(*([0] * num_examples)), POINTER(c_int))
        LibLR.predict(c_X, predict_results, c_int(num_examples), c_int(self.num_features), c_char_p(self.tmp_model_path))
        return np.array([predict_results[i] for i in xrange(num_examples)])

    def __del__(self):
        if self.tmp_model_path is not None:
            os.remove(self.tmp_model_path)
