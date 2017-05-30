from cifar_classifier import cifar_classifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from random import randint
def plot_prediction(network, X_test, y_test, model):
    # get predictions on the test set
    y_hat = model.predict(X_test)

    # define text labels
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # plot a random sample of test images, their predicted labels, and ground truth
    fig = plt.figure(figsize=(20, 8))
    for i, idx in enumerate(np.random.choice(X_test.shape[0], size=32, replace=False)):
        ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(X_test[idx]))
        pred_idx = np.argmax(y_hat[idx])
        true_idx = np.argmax(y_test[idx])
        ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
    #plt.show()
    fig.savefig(''.join([network,'.jpg']))

cc = cifar_classifier()
model = cc.get_model('mlp')
model.load_weights('cifar10.model.mlp.hdf5')
X_test, y_test = cc.get_test_data()
y_test, _ = cc._one_hot_encode(y_test,y_test)
X_test, _ = cc._rescale(X_test, X_test)

# predict 7 random digits using mlpt
plot_prediction('mlp',X_test, y_test,model)

model = cc.get_model('cnn')
model.load_weights('cifar10.model.cnn.hdf5')

plot_prediction('cnn', X_test, y_test,model)
