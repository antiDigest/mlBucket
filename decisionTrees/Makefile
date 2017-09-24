# Makefile
TRAIN="../data/data_sets2/training_set.csv"
VAL="../data/data_sets2/validation_set.csv"
TEST="../data/data_sets2/test_set.csv"
PRUNE=0.2
CHOICE="info_gain"

all: run

run: ./partii/run.py
	python ./partii/run.py --train $(TRAIN) --validation $(VAL) --test $(TEST) --prune $(PRUNE) --choice $(CHOICE)

clean:
	rm -rf */*.pyc
	rm -rf *.zip

zip: clean .
	zip -r parti.zip parti.pdf
	zip -r partii.zip partii/
	zip -r decisionTree.zip CS6375_CoverPage.docx parti.zip partii.zip