#!/usr/bin/env python

import os, errno
import random
from math import floor

SEED=11


def train_test_split_from_dir(datadir, train_dir, test_dir, split=0.8):
    '''take base_dir'''
    n=0
    for dirname in os.listdir(datadir): #make new directories if needed
        try:
            os.makedirs(os.path.join(train_dir, dirname) + '/')

        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            print(dirname, 'already exists')
            pass
        try:
            os.makedirs(os.path.join(test_dir,dirname) + '/')
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            print(dirname, 'already exists')
            pass
    for dirname in os.listdir(datadir):
        filenames = os.listdir(os.path.join(datadir, dirname)) #list files in class
        data_files = list(filter(lambda file: file.endswith('.jpeg'), filenames)) #jpeg
        random.Random(SEED).shuffle(data_files)
        split_index = floor(len(data_files) * split)
        training = data_files[:split_index]
        testing = data_files[split_index:]

        #move files to train
        for file in training:
            try:
                os.rename(os.path.join(datadir,dirname,file), os.path.join(train_dir, dirname, file))
                n+=1
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                print(file, 'already exists')
                continue

        #move files to test
        for file in testing:
            try:
                os.rename(os.path.join(datadir,dirname,file), os.path.join(test_dir, dirname, file))
                n+=1
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                print(file, 'already exists') 
                continue 
    return str(n) + ' files moved'

# base_dir = '/home/ubuntu/open sprayer/weeds/'
# train_dir = '/home/ubuntu/open sprayer/shuffled_weeds/train/'
# test_dir = '/home/ubuntu/open sprayer/shuffled_weeds/test/'
# train_test_split_from_dir(base_dir, train_dir, test_dir)

base_dir = '/home/ubuntu/open sprayer/plantnet/'
train_dir = '/home/ubuntu/open sprayer/shuffled_plantnet/train/'
test_dir = '/home/ubuntu/open sprayer/shuffled_plantnet/test/'
train_test_split_from_dir(base_dir, train_dir, test_dir)