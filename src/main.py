#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

""" Twitter Sentiment Analysis
# Ricardo Sousa
# rsousa at rsousa.org

# 2015 Ricardo Sousa
"""

import argparse
from data_handling import process_data
from sklearn.ensemble.forest import RandomForestClassifier

__author__ = "Ricardo Sousa"
__copyright__ = "GPL 2015"
__credits__ = []
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Ricardo Sousa"
__email__ = "rsousa@rsousa.org"
__status__ = "Dev"


def main(args):

    if args.analyse != None:
        train_data_x, test_data_x,train_data_y, test_data_y  = process_data(args.analyse)

        RT = RandomForestClassifier(n_estimators=100)
        RT.fit(train_data_x, train_data_y)
        print RT.score(test_data_x, test_data_y)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twitter Sentiment Analysis")
    parser.add_argument('--analyse', dest='analyse', default=None, type=str)

    args = parser.parse_args()
    main(args)
