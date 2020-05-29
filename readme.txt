a simple program to make a convolutional neural network

----------------------

net = Network(3, 32, [16, 32, 64], 3, 0.25, [16, 8], range(10), torch.nn.ReLU(), 0.001, 30)

this command will create a networks that takes images of the shape (32, 32, 3)
there will be three convolution layers with 16, 32 and 64 units respectively with padding of 3 and a dropout of 0.25
two fully connected layers with units 16 and 8
the output layer will have 10 units i.e. 10 classes
the activation function to be used would be ReLU
the learning rage would be 0.001 and the training would run for 30 epochs

----------------------

net.forward()

completes a forward pass

----------------------

training_time, training_loss = net.train(dataloader_object)

command will train the network and return the training time and the training loss

----------------------

  accuracy, test_loss, confusion_matrix, f1_micro_score, f1_macro_score, f1_weighted_score = net.use(dataloader_object)

  testing the network


----------------------

**This program will only accept square images**
