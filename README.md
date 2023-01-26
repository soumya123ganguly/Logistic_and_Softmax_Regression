
<h1 align="center">
  Logistic and Softmax Regression for classifying MNIST dataset
</h1>

<p align="center">
  <a href="#Introduction">Introduction</a> •
  <a href="#Usage">Usage</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Introduction

The project aims at implementing simple regression models for classifying MNIST dataset. The project is divided into data.py, network.py and main.py. Data loading, preprocessing is implemented in the data.py file. The network functions, loss function and activation functions are implemented in network.py file. The main.py file implements the training loop.

## Usage

```bash
# Clone this repository
$ https://github.com/nishanthrachakonda/PA-1

# Go into the repository
$ cd PA-1

# Install dependencies
$ pip3 install numpy
$ pip3 install matplotlib
$ pip3 install scikit-learn

Change the network object with softmax activation function and multiclass_cross_entropy for 10 class classification and update the network object with sigmoid activation function and binary_cross_entropy for 2 class classification of 2 and 7, 5 and 8. Also uncomment the lines for choosing the data from these classes
# Run the app
$ python main.py --p 100 --learning-rate 0.01 --batch-size 32 --k 10 
```

## Credits

Soumya and Nishanth have collaborated in implemting the project from ground up.

## License

MIT

