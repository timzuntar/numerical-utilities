### Purpose

The file net_training.py contains python code to define and train a basic feed-forward neural network as well as evaluate its performance. Since the purpose of the exercise was to improve the understanding of basic concepts, it was (shoddily) implemented from scratch.

So far, it has only been tested on the MNIST handwriten digit dataset to try out image recognition.

### The basics

under construction

### Stochastic gradient descent

The dataset includes a total of 70 000 images, of which 10 000 are explicitely labelled as training data. This should be an arbirary distinction assuming the training data are representative - nevertheless, this "suggested" division of data was used at the start. The simplest way to update system weights and biases is fully stochastically after each training example, but before doing that, let us check if the training samples need to be shuffled.

![](https://github.com/timzuntar/numerical-utilities/blob/master/Neural_net/output/class_distribution.png?raw=true)

Evidently not all classes are equally common. The digit 1, for example, comes up in 11.2% of the dataset, while the digit 5 only occurs in 9% of the training data. Overall, though, the differences are minor, and more importantly, the trend is very linear, which means the examples are well shuffled and we can proceed with the training.

Four different values of the learning rate η spanning from 1e-4 (lowest) to 0.1 (highest) were tested.
