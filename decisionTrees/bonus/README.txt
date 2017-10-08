#README.txt

Machine Learning CS6375 -- Assignment 2

PART 2

Programming Language used: Python

************************************************************ TO RUN *************************************************************

# Main file to run: "partii/run.py"

```
	usage: run.py [-h] --train TRAIN --validation VALIDATION --test TEST
	              [--prune PRUNE] [--choice {info_gain,random}]

	Training Decision Trees.

	optional arguments:
	  -h, --help            show this help message and exit
	  --train TRAIN         Supply a training dataset.
	  --validation VALIDATION
	                        Supply a validation dataset.
	  --test TEST           Supply a test dataset.

```

* The "choice" argument is added for the bonus assignment, which can either make your decision tree on random choices or best choices using Information Gain.


# Example run command on bash shell:

```
	python run.py --train "../data/data_sets2/training_set.csv" --validation "../data/data_sets2/validation_set.csv" --test "../data/data_sets2/test_set.csv"
```

OR

```
	python ./partii/run.py --train "../data/data_sets1/training_set.csv" --validation "../data/data_sets1/validation_set.csv" --test "../data/data_sets1/test_set.csv"
```