
# A Neural Network From Scratch
This project was made for educational purposes to practice the skills and knowledge gained during the deep learning course and build my own neural network using minimum number of packages (numpy, pandas, matplotlib).

This neural network is for classifications tasks and it was mostly built for digit dataset from [kaggle](https://www.kaggle.com/competitions/digit-recognizer/overview) (**train.csv**, **test.csv** files).

# Usage
Open main.py file in any notebook or ide. 

``` python
test = NeuralNetwork(784 , [50, 200, 20] , 10,  'classification', batches = True)

test.prepare(gradient_method = 'gd', activation_func = 'leaky_relu', seed = None, alpha = 0.01, loss_function = 'cross_entropy_loss', val_metric = 'accuracy',  optimizer = 'accelerated_momentum', momentum = 0.9)

test.cosmetic(progress_bar = False, loss_display = True, loss_graphic = False,  iterations = 100)

test.train(train_batches, test_batches, 3)
```

As you can see, you may choose the number of inputs, outputs, hidden layers and number of neurons for every layer, gradient descent algorithm, activation function, alpha parameter etc.

**Note: This implementation of a neural network is highly scalable, unlike other user implementations.**

However, be careful when you increase the number of layers and neurons, as due to the high losses, the learning process becomes less controllable.

## Neural Net Architecture
```python
 (784 , [50, 200, 50] , 10)
 ```
 ![nn graph](https://github.com/TimaGitHub/NeuralNetwork-from-Scratch/assets/70072941/1b1e2350-11f0-4103-b2a6-95a6348320f9)

 ### progress_bar and loss_display
 ![gh1](https://github.com/TimaGitHub/NeuralNetwork-from-Scratch/assets/70072941/d4484b22-655b-437a-a53f-897ebad3b8f2)

 ### loss_graphic
 ![gh3](https://github.com/TimaGitHub/NeuralNetwork-from-Scratch/assets/70072941/14317df1-68cf-4086-b107-e79e9dbbf55e)

 You can also save your model and download it's parameters next time.
 ```python
 test.save(path = 'model_params.npy')
test.load('model_params.npy')
 ```

 ### Short brief about .py files
 - **dataloader.py**
 
    analog of ```torch.utils.data.DataLoader```, but self-made

- **functions.py**
 
   contains most popular activation functions: sigmoid, tanh, relu, leaky relu, cross-entropy loss function, softmax function

- **gradient_steps.py**

    contains 3 most popular gradient descent algorithms: normal, stochastic and stochastic average

- **augmentation.py**

    provides data augmentation for digit images (rotate, resize, noise etc.)

    ![image](https://i.sstatic.net/EaKBb.png)

- **neural_network.py**

    main file with neural network implementation


## Predict your own hand-written images

- open **draw_digit.py**
- train your model
- draw a digit in a Tkinter GUI window
- make prediction
  
  ![gh2](https://github.com/TimaGitHub/NeuralNetwork-from-Scratch/assets/70072941/606d6c19-da9c-41bf-9647-3eea56f27295)



## To-Do List
- [X] expand neural network for regression puprose (look for new project - PyCandle)
- [X] add L1, L2 regularization (look for new project - PyCandle)
- [X] make it more robust for large number of layers and neurons (look for new project - PyCandle)
- [X] add Batch Normalization (look for new project - PyCandle)
- [X] add optimizers ( RMSProp, Adam etc.) (look for new project - PyCandle)
- [X] make class more pytorch-like (look for new project - PyCandle)
- [ ] win Kaggle competition with self-made neural network :sunglasses:


## Supplementary materials

- very clear [explanation](https://colab.research.google.com/drive/1I-yxouAvKTOifFVfuaDYIS2x8LNiMQN4?usp=sharing#scrollTo=OQ7yPMeZjd9K) of how neural network is made
- nice 3Blue1Brown [videos](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=vZ3tJjTqXa9iSfBE)
- very cool [network](https://www.youtube.com/watch?v=hfMk-kjRv4c&t=2708s&pp=ygUYbmV1cmFsIG5ldHdvcmsgc2VsZiBtYWRl) made from scratch by Sebastian Lague

- cool and very clear [model](https://youtu.be/w8yWXqWQYmU?si=p1C-AUBRy7XWIQoM) from scratch made by Samson Zhang

- chatgpt, to my surprise, can also make network from scratch (he builds it for XOR classification)
- some very usefull videos to understand classification problem ( [1](https://youtu.be/hkj3OoSWQGo?si=M0RA1rXhU4f0Ae8p), [2](https://youtu.be/ftddLO6KvSo?si=gwSjI3bCOl3KNH_z), [3](https://youtu.be/U3PPDmc15Bc?si=cXjbROcGL6VCApij) )
