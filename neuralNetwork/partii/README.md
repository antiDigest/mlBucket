# Assignment done by:

	Antriksh Agarwal
	Sarvesh Pandit


## Assignment Part-2

### Pre-Processing Part

files = preprocess.py, Dataset.py

run as:

		python preprocess.py <input-path> <store-path>

### Neural Network Part

files = Network.py, run.py, Dataset.py

usage: run.py [-h] dataset percent iterations hiddenLayers hidden

Pre Processing Dataset.

positional arguments:
  dataset       Complete path of the post-processed input dataset.
  percent       Percentage of the dataset to be used for training
  iterations    Maximum number of iterations that your algorithm will run.
                This parameter is used so that your program terminates in a
                reasonable time.
  hiddenLayers  Number of hidden layers
  hidden        number of neurons in each hidden layer

optional arguments:
  -h, --help    show this help message and exit

run as:

		python run.py <dataset-path> <percentage> <number-of-iterations> <number-of-hidden-layers> "<layer-sizes-separated-with-spaces>"

example command:

		python run.py ../data/iris/iris.csv 30 1000 3 "10 8 10"