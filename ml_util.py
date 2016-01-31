# coding: utf-8
import random


def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results


def train_test_split(data, test_pct):
    # split the dataset of pairs 
    train_data, test_data = split_data(data, 1 - test_pct)   
    return train_data, test_data