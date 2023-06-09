# Adversarial and Backdoor Attack + Defence 
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) 

![AI Hacking Defence](https://github.com/zlaabsi/adversarial-backdoor-attack-defence/assets/52045850/e428ef51-340d-4816-ab24-f01fa7d0a032)


## Demo (French version)


https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/f09dd014-b0c8-4c41-8aab-0e41868af40a



https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/c28b83de-c347-4b90-b694-25bbe1e4c6d2


---

The demo allows you to test the following preloaded datasets:

* **MNIST** (*digit recognition*) :  A large dataset of handwritten digits used for training and testing in the field of machine learning.

* **GTSRB** (*street sign recognition*) :  The German Traffic Sign Recognition Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. It's used in the development of algorithms for traffic sign recognition.

* **CIFAR-10** (*object recognition, small images*) : A dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images. This dataset is used for object recognition in smaller images.

* **ImageNet** (*object recognition, large images*) : A large dataset used in object recognition software research, consisting of millions of images with thousands of labeled categories. It's generally used for training deep learning models on large images.

---

Here are the adversarial attacks available in the demo and their associated research paper :

* **[Carlini & Wagner](https://arxiv.org/pdf/1608.04644.pdf) ($L_2$ attack)** : This is a powerful white-box adversarial attack method which focuses on creating adversarial examples with the smallest possible L2 distance between the original and perturbed inputs. 

* **[Jacobian-based Saliency Map Attack, One-Pixel](https://arxiv.org/pdf/1511.07528.pdf) ($L_0$ attack)** : This type of attack uses the Jacobian matrix to determine which pixels in an image, when changed, will have the highest impact on the output. The One-Pixel version aims to change just one pixel in the image, exploiting the L0 distance measure.

* **[Jacobian-based Saliency Map Attack](https://arxiv.org/pdf/1511.07528.pdf) ($L_0$ attack)** :  Similar to the above, this attack aims to change as few pixels as possible in an image, based on the impact on the output as calculated using the Jacobian matrix. The key difference is that it may modify more than one pixel.

* **[Basic Iterative Method](https://arxiv.org/pdf/1607.02533.pdf) ($L_{\infty}$ attack)** : This attack method, also known as the Projected Gradient Descent (PGD) attack, is an iterative version of the Fast Gradient Sign Method. It repeatedly applies FGSM and clips the perturbations to ensure they are within the ε-bounds (L∞ norm).

* **[Fast Gradient Sign Method](https://arxiv.org/pdf/1412.6572.pdf) ($L_{\infty}$ attack)** : This is one of the simplest and fastest methods for generating adversarial examples. It uses the gradients of the neural network with respect to the input data to create adversarial examples. 

The demo allows you to test the defense method against a backdoor attack:
* **[Deep Partition Aggregation](https://arxiv.org/pdf/2006.14768.pdf)** :  A defense strategy against poisoning attacks. It partitions data into subsets, trains individual models on each, then aggregates their predictions. This lessens the impact of poisoned data as it's dispersed across multiple models.


## Principles of adversarial and backdoor attacks

### Difference between backdoor and adversarial attack.

![Difference-between-backdoor-and-adversarial-attack](https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/af457ddd-a75e-4d53-98ae-169f3f50c5b4)


![Approaches-to-backdooring-a-neural-network-proposed-in-Gu-et-al-2019-The-backdoor](https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/19615f7d-8257-4f8b-809b-60bac28a70d0)

*Approaches to backdooring a neural network proposed in (Gu et al., 2019). The backdoor trigger is a pattern of pixels that appears on the bottom right corner of the image. (a) A benign network that correctly classifies its input. (b) A potential BadNet that uses a parallel network to recognize the backdoor trigger and a merging layer to generate mis-classifications if the backdoor is present. However, it is easy to identify the backdoor detector. (c) The BadNet has the same architecture as the benign network, but it still produces mis-classifications for backdoored inputs.*


Pan, Zhixin & Mishra, Prabhat. (2022). Backdoor Attacks on Bayesian Neural Networks using Reverse Distribution. 10.48550/arXiv.2205.09167. 

----
----

# Backdoor Attack with Adversarial Training [code]

Research indicates that deep neural networks are susceptible to discrete backdoor attacks. These networks usually perform well, but when backdoor triggers are inserted, they incorrectly classify manipulated examples. 

Earlier backdoor attacks were detectable, so a new approach creates undetectable backdoor triggers unique to each example. However, to evade "neural cleanse" detection, adversarial training is suggested. 

This refined backdoor attack strategy shows promising results, maintaining invisibility and successfully bypassing known defense mechanisms.

Everything that follows will use the **Adversarial Robustness Toolbox (ART)** Python library and this notebook : https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_deep_partition_aggregation.ipynb, with a few modifications.

![art_lfai](https://github.com/zlaabsi/adversarial-backdoor-attack-defence/assets/52045850/7a5ec989-3e36-40ee-8bcd-ec917c2b12cb)

## Initialize the Model Architecture

### Create Keras convolutional neural network - basic architecture from Keras examples
#### Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

```python
def create_model():    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```
![output_mini](https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/c6d12465-a401-4d73-9724-61276515da40)



## Set up the Model Backdoor

```python
backdoor = PoisoningAttackBackdoor(add_pattern_bd)
example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
pdata, plabels = backdoor.poison(x_test, y=example_target)

plt.imshow(pdata[0].squeeze())

```

- `backdoor = PoisoningAttackBackdoor(add_pattern_bd)`: This line of code is setting up a "Backdoor" attack (introduced in Gu et al., 2017.) with a given pattern that is defined by the add_pattern_bd function. A backdoor attack in machine learning typically refers to a type of attack where the attacker injects a "backdoor" or "trigger" into the model during training. When this trigger is present in the input, the model will produce a specific output, regardless of the actual input. Paper link: https://arxiv.org/abs/1708.06733

- `example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])`: Here, we are creating a numpy array example_target. This represents the target output class when the backdoor is present. In this case, it seems to be a 10-class classification problem and the backdoor trigger will cause the model to predict the last class.

## Create the poison data

In this scenario, we will choose the target class as 9. Consequently, the objective of the adversary is to contaminate the model in such a way that the addition of a trigger causes the trained model to misclassify the input as a 9.

Initially, the adversary will create a proxy classifier, which is a classifier similar to the target classifier. Since the clean label attack generates noise using PGD to encourage the trained classifier to rely on the trigger, it is crucial that the generated noise can be transferred. Therefore, adversarial training is employed.

### Poison some percentage of all non-nines to nines


```python
targets = to_categorical([9], 10)[0] 

proxy = AdversarialTrainerMadryPGD(KerasClassifier(create_model()), nb_epochs=10, eps=0.15, eps_step=0.001)
proxy.fit(x_train, y_train)

attack = PoisoningAttackCleanLabelBackdoor(backdoor=backdoor, proxy_classifier=proxy.get_classifier(),
                                           target=targets, pp_poison=percent_poison, norm=2, eps=5,
                                           eps_step=0.1, max_iter=200)
pdata, plabels = attack.poison(x_train, y_train)

poisoned = pdata[np.all(plabels == targets, axis=1)]
poisoned_labels = plabels[np.all(plabels == targets, axis=1)]

```

- ``AdversarialTrainerMadryPGD`` : Performing adversarial training following Madry’s Protocol. Paper link: https://arxiv.org/abs/1706.06083

- ``PoisoningAttackCleanLabelBackdoor`` : Implementation of Clean-Label Backdoor Attack introduced in Turner et al., 2018. Applies a number of backdoor perturbation functions and does not change labels. Paper link: https://people.csail.mit.edu/madry/lab/cleanlabel.pdf



![download](https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/7571fe65-982f-4f1c-b34a-362cf21d6c02)


---

## Initialize the classification models

We will initialize two models. The first model follows a single architecture, while the second model is a DPA model with an ensemble size of 50. The purpose is to showcase the tradeoff between clean accuracy and poison accuracy. Please note that the process may require some time due to model duplication.

```python

model = KerasClassifier(create_model())
dpa_model_50 = DeepPartitionEnsemble(model, ensemble_size=50)

````

## Train the models on the poisoned data

```python
model.fit(pdata, plabels, nb_epochs=10)
dpa_model_50.fit(pdata, plabels, nb_epochs=10)
````

---

## Evaluate clean data

Evaluate the performance of the trained models on clean data.

```python

clean_preds = np.argmax(model.predict(x_test), axis=1)
clean_correct = np.sum(clean_preds == np.argmax(y_test, axis=1))
clean_total = y_test.shape[0]
clean_acc = clean_correct / clean_total
print("Clean test set accuracy (model): %.2f%%" % (clean_acc * 100))
c = 0
i = 0
c_idx = np.where(np.argmax(y_test, 1) == c)[0][i]
plt.imshow(x_test[c_idx].squeeze())
plt.show()
clean_label = c
print("Prediction: " + str(clean_preds[c_idx]))

clean_preds = np.argmax(dpa_model.predict(x_test), axis=1)
clean_correct = np.sum(clean_preds == np.argmax(y_test, axis=1))
clean_total = y_test.shape[0]
clean_acc = clean_correct / clean_total
print("Clean test set accuracy (DPA model_50): %.2f%%" % (clean_acc * 100))
c = 0
i = 0
c_idx = np.where(np.argmax(y_test, 1) == c)[0][i]
plt.imshow(x_test[c_idx].squeeze())
plt.show()
clean_label = c
print("Prediction: " + str(clean_preds[c_idx]))

````
        
## Evaluate poisoned data     

 Evaluate the performance of the trained models on poisoned data.

```python

 
not_target = np.logical_not(np.all(y_test == targets, axis=1))
px_test, py_test = backdoor.poison(x_test[not_target], y_test[not_target])
poison_preds = np.argmax(model.predict(px_test), axis=1)
clean_correct = np.sum(poison_preds == np.argmax(y_test[not_target], axis=1))
clean_total = y_test.shape[0]
clean_acc = clean_correct / clean_total
print("Poison test set accuracy (model): %.2f%%" % (clean_acc * 100))
c = 0
plt.imshow(px_test[c].squeeze())
plt.show()
clean_label = c
print("Prediction: " + str(poison_preds[c]))

poison_preds = np.argmax(dpa_model.predict(px_test), axis=1)
clean_correct = np.sum(poison_preds == np.argmax(y_test[not_target], axis=1))
clean_total = y_test.shape[0]
clean_acc = clean_correct / clean_total
print("Poison test set accuracy (DPA model_50): %.2f%%" % (clean_acc * 100))
c = 0
plt.imshow(px_test[c].squeeze())
plt.show()
clean_label = c
print("Prediction: " + str(poison_preds[c]))
        
 
 ````


## References

- Credit for the demo in JavaScript : Kenny Song
https://github.com/kennysong/adversarial.js

- Nicolae, M.-I. et al. Adversarial Robustness Toolbox v1.0.0. [arXiv:1807.01069](https://arxiv.org/pdf/1807.01069.pdf) [cs, stat] (2019)

- Credit for the Python script : https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_deep_partition_aggregation.ipynb

- Pan, Zhixin & Mishra, Prabhat. (2022). Backdoor Attacks on Bayesian Neural Networks using Reverse Distribution. [doi:10.48550/arXiv.2205.09167](https://arxiv.org/abs/2205.09167)

- Levine, A., & Feizi, S. (2020). Deep Partition Aggregation: Provable Defense against General Poisoning Attacks. ArXiv, [abs/2006.14768](https://arxiv.org/abs/2006.14768)

- L. Feng, S. Li, Z. Qian and X. Zhang, "Stealthy Backdoor Attack with Adversarial Training," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, Singapore, 2022, pp. 2969-2973, [doi:10.1109/ICASSP43922.2022.9746008](https://ieeexplore.ieee.org/document/9746008)







