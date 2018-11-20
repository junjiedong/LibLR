from LibLR import LogisticClassifier
import numpy as np

def accuracy(y, y_pred):
    m = y.shape[0]
    correct_cnt = 0.0
    for i in xrange(m):
        if (y[i] >= 0.5 and y_pred[i] >= 0.5) or (y[i] < 0.5 and y_pred[i] < 0.5):
            correct_cnt += 1
    return correct_cnt / m

# Artificial dataset configurations
num_features = 100
num_train_examples = 50000
num_val_examples = 10000
noise_variance = 1.0

# Generate artificial dataset
weights = np.random.randn(num_features, 1)
bias = np.random.randn()
X_train = np.random.randn(num_train_examples, num_features)
X_val = np.random.randn(num_val_examples, num_features)
logit_train = X_train.dot(weights) + bias + noise_variance * np.random.randn(num_train_examples, 1);
logit_val = X_val.dot(weights) + bias + noise_variance * np.random.randn(num_val_examples, 1);
Y_train = (logit_train > 0) + 0.0
Y_val = (logit_val > 0) + 0.0
print "Dataset generation finished!"

# Train LR classifier
LR = LogisticClassifier(num_features)
num_epoch = 40
batch_size = 500
num_iter = num_train_examples / batch_size * num_epoch
LR.train(X_train, Y_train, learning_rate=0.02, reg=0.005, batch_size=500, num_iter=num_iter)

# Make prediction on the validation set
y_predict = LR.predict(X_val)
print "Validation accuracy:", accuracy(Y_val, y_predict)
