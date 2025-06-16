### Purpose

The file net_training.py contains python code to define and train a basic feed-forward neural network as well as evaluate its performance. Since the purpose of the exercise was to improve the understanding of basic concepts, it was (shoddily) implemented from scratch.

So far, it has only been tested on the MNIST handwriten digit dataset to try out image recognition.

### The basics

under construction

### Stochastic gradient descent

The dataset includes a total of 70 000 images, of which 10 000 are explicitely labelled as training data. This should be an arbirary distinction assuming the training data are representative - nevertheless, this "suggested" division of data was used at the start. The simplest way to update system weights and biases is fully stochastically after each training example, but before doing that, let us check if the training samples need to be shuffled.

![](https://github.com/timzuntar/numerical-utilities/blob/master/Neural_net/output/class_distribution.png?raw=true) ![](https://github.com/timzuntar/numerical-utilities/blob/master/Neural_net/output/class_distribution_scaled.png?raw=true)

Evidently not all classes are equally common. The digit 1, for example, comes up in 11.2% of the dataset, while the digit 5 only occurs in 9% of the training data. Overall, though, the differences are minor, and more importantly, the deviations from linearity are small - on the right-hand plot, they are plotted in units of the square root of the number of class members. The values never exceed one, so we have no reason to believe the slightly uneven distribution is caused by anything else but chance. This means the examples are well shuffled and we can proceed with training.

Training using the entire data set takes about a minute of real time on my machine. Of course, the code is unoptimized and only running on a single core - it would be interesting to see, for example, what kind of gains in speed a shift to batch gradient descent and parallelization the batches could achieve, but that is beyond the scope of this exercise. Instead, without further ado, the results.

Four different values of the learning rate η spanning from 1e-4 (lowest) to 0.1 (highest) were tested. The instantaneous loss function values (magnitudes of mismatches between the predicted and true answer) and the prediction accuracy of the network so far were continuously recorded. The below image of cumulative prediction accuracy of the network clearly shows that the choice of learning rate has a profound impact on performance; both too small and too large values lead to poor results. In the case of too quick learning, the network never stabilizes, while if the value is set too low, we run out of training examples before "convergence". In contrast, the middle values both achieved nearly the same accuracy by the end despite differing by an order of magnitude.

![](https://github.com/timzuntar/numerical-utilities/blob/master/Neural_net/output/accuracy_progression.png?raw=true)

It is more interesting to look at how the trained networks perform on the testing dataset; keep in mind that the above plots are not really accurate to the instantaneous performance, since poor performance at the start of training drags them down. But before getting to the results, let's use the example of the trained network and visualize the features it assigns the (near-)perfect number score to. In order to achieve that, the network's desired output and learned parameters are kept fixed and an arbitrary input image is introduced as its initial state. The backpropagation steps then act on and change the input image instead of weights and biases, causing it to gradually shift to the input which, given the trained parameters, would minimize the difference to the desired classification. An example of a set of classification-maximizing inputs is shown below.

| ![](https://github.com/timzuntar/numerical-utilities/blob/master/Neural_net/output/full_slow_learning_feature_vis_all.png?raw=true) |
|:--:|
| Visualization of features corresponding to optimal classification, computed through iterative back-propagation. The top row contains features corresponding to classification as digits 0-4 (in increasing order from left to right); features corresponding to digits 5-9 are contained in the bottom row. The input maps have been generated using the network trained with a learning rate of 1e-3 and a total error acceptance threshold of 0.6/(28 x 28).|
