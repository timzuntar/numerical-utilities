"""
Disclaimer: The datareader helper function has been reproduced from work by Yann LeCun.
"""

import sys
import struct
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np


## read images and labels from binary files, one by one (generator function) ##
def datareader(imagename,labelname):
    with open(imagename,"rb") as images, open(labelname,"rb") as labels:
        magic,length,w,h=struct.unpack('>IIII',images.read(4*4))
        magic,lengthL=struct.unpack('>II',labels.read(4*2))
        print("dataset of {} images of size {}x{}".format(length,w,h),file=sys.stderr)
        assert length == lengthL
        for i in range(length):
            yield np.frombuffer(images.read(w*h),dtype='uint8'),labels.read(1)[0]

def check_distribution(datareader):
    #Assumes that the classes are labeled with integers.
    all_labels = []
    for _,label in datareader:
        all_labels.append(label)
    
    all_labels = np.asarray(all_labels)
    unique_labels = np.unique(all_labels)
    cumulants = np.empty((len(all_labels),len(unique_labels)))

    for i in range(len(unique_labels)):
        cumulants[:,i] = (all_labels == unique_labels[i]).cumsum()

    return cumulants

def cumulant_plot(cumulants,expected_prevalences,scale_to_sqrtn = False):
    # shows the deviation from the expected (perfectly shuffled) distribution of the training data classes
    fig = plt.figure()
    ax = fig.add_subplot()
    
    for i in range(len(expected_prevalences)):
        if scale_to_sqrtn == True:
            plt.plot(np.arange(len(cumulants[:,i])),(cumulants[:,i]-np.arange(len(cumulants[:,i]))*expected_prevalences[i]/len(cumulants[:,i]))/np.sqrt(np.arange(len(cumulants[:,i]))+1),label="%d" % i)
        else:
            plt.plot(np.arange(len(cumulants[:,i])),cumulants[:,i],label="%d" % i)
    
    plt.title("Class distribution in the training set")
    plt.xlabel("Cumulative number of training samples")
    if scale_to_sqrtn == True:
        plt.ylabel(r"Deviation of class occurences from expectation ($\Delta n / \sqrt{n}$)")
    else:
        plt.ylabel("Cumulative occurences of each class")

    plt.legend(loc="lower right")
    plt.show()
    return 0

def softmax(weights,vector,bias):
    # activation function between network layers
    return np.tanh(np.matmul(weights,vector)+bias)

def step_fwd(v_input,w_matrices,bias_vectors):
    # propagation of a single input through the network
    num_layers = len(w_matrices)
    activation_values = []
    vector_values = [v_input]
    for i in range(num_layers):
        v_input = softmax(w_matrices[i],v_input,bias_vectors[i])
        activation_values.append(v_input)
        vector_values.append(v_input)

    return v_input,activation_values,vector_values

def backpropagate(output,control,w_matrices,activation_values):
    # propagation of a single prediction error back through the network
    # we take the final step explicitely
    diff = output-control
    E = 0.5*np.sum(np.square(diff)) # error function
    grad = np.multiply(-diff,np.ones(np.shape(activation_values[-1]))-np.square(activation_values[-1]))
    gradients = []
    gradients.append(grad)

    # stepping backwards
    for i in range(len(w_matrices)-1,0,-1):
        grad = np.multiply(np.matmul(grad,w_matrices[i]),np.ones(np.shape(activation_values[i-1]))-np.square(activation_values[i-1]))       
        gradients.append(grad)

    return E,list(reversed(gradients))

def update_weights(w_matrices,gradients,vector_values,learning_rate,bias_vectors=None):
    # If updates should not be passed to biases, they can be left out

    for i in range(len(w_matrices)):
        w_matrices[i] = w_matrices[i]+learning_rate*np.outer(gradients[i],vector_values[i])
    
    if bias_vectors is not None:
        for i in range(len(bias_vectors)-1): # the biases of the final layer must be equal and identical to 0
            bias_vectors[i] = bias_vectors[i] + learning_rate*gradients[i]

    return w_matrices,bias_vectors

def hallucinate(acceptable_diff,max_iters,dimensions,classification,learning_rate,weight_matrices,bias_vectors):
    # takes a trained NN and reconstructs the platonic ideal of one of its classifiers
    # classification is meant to just be the corresponding index in the output vector

    blank_slate_input = np.zeros(dimensions[0]*dimensions[1])

    output = np.full(10,-1.0)
    output[classification] = 1.0

    for k in range(max_iters):

        out_vec,act_val,vec_val = step_fwd(blank_slate_input,weight_matrices,bias_vectors)
        diff = np.sum(np.abs(output-out_vec))
        print(diff)
        if diff < acceptable_diff:
            print("Converged after %d iterations." % k)
            break
        E,grad_val = backpropagate(out_vec,output,weight_matrices,act_val)
        # now update the input image
        blank_slate_input = blank_slate_input + learning_rate*np.matmul(grad_val[0],weight_matrices[0])

    return blank_slate_input


#######################################################
# Let's define some constants and parameters

image_dimensions = [28,28]
parameter_set_name = "full_slow_learning"

layer_sizes = [784,784,28,10]   # input and output layers are included, so the number of hidden layers is len-2
learning_rate = 0.001

weight_matrices = []
bias_vectors = []

training_dataset = "train-images-idx3-ubyte"
training_labels = "train-labels-idx1-ubyte"
testing_dataset = "t10k-images-idx3-ubyte"
testing_labels = "t10k-labels-idx1-ubyte"

cumulants = check_distribution(datareader(training_dataset,training_labels))
print(cumulants[-1,:]/60000)
cumulant_plot(cumulants,cumulants[-1,:],scale_to_sqrtn = False)
exit()
# Here, the operator is trusted to not exceed the amount of training data actually included in the database
training_examples_to_use = 60000
test_examples_to_use = 100

# First, we will check if the network has already learned from these data and parameters
# If a saved set by the same name already exists, open and compare it

try:
    prev_result = []
    with open("output/"+parameter_set_name+"_trained.pkl", "rb") as f:
        for _ in range(pickle.load(f)):
            prev_result.append(pickle.load(f))

    # If the parameter set equals the currently defined one, the trained weights and biases are loaded  
    if [layer_sizes,learning_rate,[training_dataset,testing_dataset]] == prev_result[0:3] and training_examples_to_use == prev_result[3][0]:
        print("The network has already been trained for this set of parameters! Skipping to evaluation...")
        trained_network_exists = True
        weight_matrices = prev_result[4]
        bias_vectors = prev_result[5]
        error_values = prev_result[6]
        accuracies = prev_result[7]
        max_mismatches = prev_result[8]
    else:
        trained_network_exists = False
except:
    trained_network_exists = False

if trained_network_exists != True:
    # Generates initial random weights
    for i in range(0,len(layer_sizes)-1):
        weights = np.random.normal(0.0,2.0/layer_sizes[i+1],size=(layer_sizes[i+1],layer_sizes[i]))
        bias = np.full(layer_sizes[i+1],0.0)
        weight_matrices.append(weights)
        bias_vectors.append(bias)

    # The first implementation is going to be a fully stochastic gradient descent
    # Training

    error_values = []
    accuracies = []
    max_mismatches = []

    accuracy = 0

    k = 0
    for image,label in datareader(training_dataset,training_labels):

        rescaled = np.divide(image,255)
        control = np.full(10,-1.0)
        control[label] = 1.0

        out_vec,act_val,vec_val = step_fwd(rescaled,weight_matrices,bias_vectors)
        
        guess = np.argmax(out_vec)
        if (guess == label):
            correct = 1
        else: correct = 0
        accuracy = (accuracy*k+correct)/(k+1) #probability of correct prediction based on all predictions to this point

        E,grad_val = backpropagate(out_vec,control,weight_matrices,act_val)
        largest_mismatch = np.max(np.abs(out_vec-control))  # component with largest difference to correct values

        weight_matrices,bias_vectors = update_weights(weight_matrices,grad_val,vec_val,learning_rate,bias_vectors)

        error_values.append(E)
        accuracies.append(accuracy)
        max_mismatches.append(largest_mismatch)        

        k += 1
        if (k % (training_examples_to_use//100) == 0):
            print("%d %%" % ((100*k)//training_examples_to_use),end='\r')
        if k > training_examples_to_use:
            break

    # save the trained parameters for later use
    output_data = [
        layer_sizes,learning_rate,[training_dataset,testing_dataset],[training_examples_to_use,test_examples_to_use],
        weight_matrices,bias_vectors,error_values,accuracies,max_mismatches]

    with open("output/"+parameter_set_name+"_trained.pkl", "wb") as f:
        pickle.dump(len(output_data), f)
        for value in output_data:
            pickle.dump(value, f)

print("Training is finished. Testing network...")

""" acceptable_diff = 1.0
max_iters = 10000
ideal_zero = hallucinate(acceptable_diff,max_iters,image_dimensions,0,learning_rate,weight_matrices,bias_vectors)

plt.imshow(np.reshape(ideal_zero,(28,28)))
plt.show()
exit() """

# Testing
i_am_confusion = np.zeros((10,10))
class_numbers = np.zeros(10,dtype=int)

l = 0
for testimage,testlabel in datareader(testing_dataset,testing_labels):
    rescaled = np.divide(testimage,255)
    out_vec,_,_ = step_fwd(rescaled,weight_matrices,bias_vectors)
    class_numbers[testlabel] += 1   # increment the correct class
    i_am_confusion[testlabel,np.argmax(out_vec)] += 1
    l += 1
    if l > test_examples_to_use:
        break

for i in range(10):
    if class_numbers[i] > 0:
        i_am_confusion[i,:] = np.divide(i_am_confusion[i,:],class_numbers[i])

print(i_am_confusion)
print(class_numbers)

plt.imshow(i_am_confusion,cmap="Greys")
plt.show()
# evaluation of results, implementation of different hidden layer sizes as well as is to follow.