import os
import sys
import warnings
import keras.backend as k
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import load_mnist, preprocess, to_categorical
from art.defences.trainer import AdversarialTrainerMadryPGD
from art.estimators.classification.deep_partition_ensemble import DeepPartitionEnsemble


class AdvAttack:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.dpa_model = None

    def load_data(self):
        """
        Load the MNIST dataset and preprocess the data.
        """
        (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)
        n_train = np.shape(x_raw)[0]
        num_selection = 10000
        random_selection_indices = np.random.choice(n_train, num_selection)
        self.x_train = x_raw[random_selection_indices]
        self.y_train = y_raw[random_selection_indices]
        percent_poison = 0.33
        self.x_train, self.y_train = preprocess(self.x_train, self.y_train)
        self.x_train = np.expand_dims(self.x_train, axis=3)
        self.x_test, self.y_test = preprocess(x_raw_test, y_raw_test)
        self.x_test = np.expand_dims(self.x_test, axis=3)
        n_train = np.shape(self.y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        self.x_train = self.x_train[shuffled_indices]
        self.y_train = self.y_train[shuffled_indices]

    def create_model(self):
        """
        Create the convolutional neural network model using Keras.
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.x_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def create_poison_data(self):
        """
        Create poisoned data using a backdoor attack.
        """
        example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)
        pdata, plabels = backdoor.poison(self.x_test, y=example_target)
        return pdata, plabels

    def create_proxy_classifier(self):
        """
        Create a proxy classifier using adversarial training.
        """
        targets = to_categorical([9], 10)[0]
        proxy = AdversarialTrainerMadryPGD(KerasClassifier(self.model), nb_epochs=10, eps=0.15, eps_step=0.001)
        proxy.fit(self.x_train, self.y_train)
        return proxy.get_classifier(), targets

    def create_poisoned_data(self, backdoor, proxy_classifier, percent_poison):
        """
        Create poisoned data using a clean label backdoor attack.
        """
        attack = PoisoningAttackCleanLabelBackdoor(backdoor=backdoor, proxy_classifier=proxy_classifier,
                                                   target=targets, pp_poison=percent_poison, norm=2, eps=5,
                                                   eps_step=0.1, max_iter=200)
        pdata, plabels = attack.poison(self.x_train, self.y_train)
        poisoned = pdata[np.all(plabels == targets, axis=1)]
        poisoned_labels = plabels[np.all(plabels == targets, axis=1)]
        return poisoned, poisoned_labels

    def initialize_models(self):
        """
        Initialize the classification models.
        """
        self.model = KerasClassifier(self.create_model())
        self.dpa_model = DeepPartitionEnsemble(self.model, ensemble_size=50)

    def train_models(self, pdata, plabels):
        """
        Train the models on the poisoned data.
        """
        self.model.fit(pdata, plabels, nb_epochs=10)
        self.dpa_model.fit(pdata, plabels, nb_epochs=10)

    def evaluate_clean_data(self):
        """
        Evaluate the performance of the trained models on clean data.
        """
        clean_preds = np.argmax(self.model.predict(self.x_test), axis=1)
        clean_correct = np.sum(clean_preds == np.argmax(self.y_test, axis=1))
        clean_total = self.y_test.shape[0]
        clean_acc = clean_correct / clean_total
        print("Clean test set accuracy (model): %.2f%%" % (clean_acc * 100))
        c = 0
        i = 0
        c_idx = np.where(np.argmax(self.y_test, 1) == c)[0][i]
        plt.imshow(self.x_test[c_idx].squeeze())
        plt.show()
        clean_label = c
        print("Prediction: " + str(clean_preds[c_idx]))

        clean_preds = np.argmax(self.dpa_model.predict(self.x_test), axis=1)
        clean_correct = np.sum(clean_preds == np.argmax(self.y_test, axis=1))
        clean_total = self.y_test.shape[0]
        clean_acc = clean_correct / clean_total
        print("Clean test set accuracy (DPA model_50): %.2f%%" % (clean_acc * 100))
        c = 0
        i = 0
        c_idx = np.where(np.argmax(self.y_test, 1) == c)[0][i]
        plt.imshow(self.x_test[c_idx].squeeze())
        plt.show()
        clean_label = c
        print("Prediction: " + str(clean_preds[c_idx]))

    def evaluate_poisoned_data(self):
        """
        Evaluate the performance of the trained models on poisoned data.
        """
        not_target = np.logical_not(np.all(self.y_test == targets, axis=1))
        px_test, py_test = backdoor.poison(self.x_test[not_target], self.y_test[not_target])
        poison_preds = np.argmax(self.model.predict(px_test), axis=1)
        clean_correct = np.sum(poison_preds == np.argmax(self.y_test[not_target], axis=1))
        clean_total = self.y_test.shape[0]
        clean_acc = clean_correct / clean_total
        print("Poison test set accuracy (model): %.2f%%" % (clean_acc * 100))
        c = 0
        plt.imshow(px_test[c].squeeze())
        plt.show()
        clean_label = c
        print("Prediction: " + str(poison_preds[c]))

        poison_preds = np.argmax(self.dpa_model.predict(px_test), axis=1)
        clean_correct = np.sum(poison_preds == np.argmax(self.y_test[not_target], axis=1))
        clean_total = self.y_test.shape[0]
        clean_acc = clean_correct / clean_total
        print("Poison test set accuracy (DPA model_50): %.2f%%" % (clean_acc * 100))
        c = 0
        plt.imshow(px_test[c].squeeze())
        plt.show()
        clean_label = c
        print("Prediction: " + str(poison_preds[c]))

    def run(self):
        """
        Run the adversarial attack.
        """
        self.load_data()
        pdata, plabels = self.create_poison_data()
        proxy_classifier, targets = self.create_proxy_classifier()
        poisoned, poisoned_labels = self.create_poisoned_data(backdoor, proxy_classifier, percent_poison)
        self.initialize_models()
        self.train_models(poisoned, poisoned_labels)
        self.evaluate_clean_data()
        self.evaluate_poisoned_data()
