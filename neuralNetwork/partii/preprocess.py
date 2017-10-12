from Dataset import Dataset

import argparse


def preprocess(dataset, store):
    data = Dataset(FILE=dataset)
    data.removeNull()
    data.toNumeric()
    data.normalize()
    data.save(store)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre Processing Dataset.')
    parser.add_argument('dataset', action='store',
                        help='Supply a dataset to pre-process.')
    parser.add_argument('store', action='store',
                        help='Supply a location to store the processed dataset.')
    args = parser.parse_args()

    if args.dataset is None:
        print "Please supply a data set to preprocess."
        print parser.print_help()
        exit()

    if args.store is None:
        print "No storage location given, replacing old data location."
        preprocess(args.dataset, args.dataset)
        exit()
    else:
        preprocess(args.dataset, args.store)
        exit()
