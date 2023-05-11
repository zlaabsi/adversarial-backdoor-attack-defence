# Adversarial and Backdoor Attack


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

## Create the poison data

For this example, we will select 9 as the target class. Thus, the adversary's goal is to poison the model so adding a trigger will result in the trained model misclassifying the #triggered input as a 9.

First, the adversary will create a proxy classifier (i.e., a classifier that is similar to the target classifier). As the clean label attack generates noise using PGD in order to encourage the trained classifier to rely on the trigger, it is important that the #generated noise be transferable. Thus, adversarial training is used.

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

![download](https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/7571fe65-982f-4f1c-b34a-362cf21d6c02)


---

## Initialize the classification models

We will initialize two models. The first is a single model architecture. The second is a DPA model with ensemble size (=50) to demonstrate the tradeoff between clean accuracy and poison accuracy. This make take some time because of the model copying.

```python

model = KerasClassifier(create_model())
dpa_model_50 = DeepPartitionEnsemble(model, ensemble_size=50)

````

## Train the models on the poisoned data

```python
model.fit(pdata, plabels, nb_epochs=10)
dpa_model_50.fit(pdata, plabels, nb_epochs=10)
````

![Difference-between-backdoor-and-adversarial-attack](https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/af457ddd-a75e-4d53-98ae-169f3f50c5b4)

![Approaches-to-backdooring-a-neural-network-proposed-in-Gu-et-al-2019-The-backdoor](https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/19615f7d-8257-4f8b-809b-60bac28a70d0)

Pan, Zhixin & Mishra, Prabhat. (2022). Backdoor Attacks on Bayesian Neural Networks using Reverse Distribution. 10.48550/arXiv.2205.09167. 

---
 ## Demo (French version)


https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/f09dd014-b0c8-4c41-8aab-0e41868af40a



https://github.com/zlaabsi/adversarial-backdoor-attack/assets/52045850/c28b83de-c347-4b90-b694-25bbe1e4c6d2

