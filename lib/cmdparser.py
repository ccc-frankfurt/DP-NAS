"""
Command line argument options parser.

Usage with two minuses "- -". Options are written with a minus "-" in command line, but
appear with an underscore "_" in the attributes' list.

You can use     python3 main.py --help    to get this printed out in terminal.
"""

########################
# importing libraries
########################
# system libraries
import argparse

parser = argparse.ArgumentParser(description='Q-learning for deep prior architecture search')

# architecture search | train fixed net
parser.add_argument('-t', '--task', type=int, default=1, help='1: architecture search, 2: train fixed net. Default: 1')
parser.add_argument('--full-training', default=False, type=bool,
                    help='turn on to use train the embedding. Default: False.')

# replay dictionary from a search and the net index no. for training search net (for continuing search or training
# fixed net)
parser.add_argument('--replay-buffer-csv-path', default=None, help='path to replay buffer. Default: None')
parser.add_argument('--fixed-net-index-no', type=int, default=-1, help='index of fixed net in replay buffer. '
                                                                       'Default: -1')
parser.add_argument('--net', default='', type=str, help='state string for testing the performance of a chosen net')
parser.add_argument('--store-embedding', default=False, type=bool, help='in task 2, whether to store the random'
                                                                        ' embedding, e.g. for further CL experiments')

# dataset and loading
parser.add_argument('--dataset', default='FashionMNIST', help='name of dataset. Default: FashionMNIST')
parser.add_argument('--gray-scale', default=True, type=bool,
                    help='use gray scale images. Default: True. If false, single channel images will be repeated '
                         'to three channels.')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers. Default: 4')
parser.add_argument('-p', '--patch-size', default=28, type=int, help='patch size for crops. Default: 28')

# path to replay dictionary, Q-values, epsilon value and iteration no. to continue search from
parser.add_argument('--continue-search', type=bool, default=False, help='set the flag to continue search '
                                                                        'Default: False')
parser.add_argument('--q-values-csv-path', default=None, help='path to stored Q-values to continue search'
                                                              'Default: None')
parser.add_argument('--continue-epsilon', default=1.0, type=float, help='epsilon to continue search from '
                                                                        'Default: 1.0')
parser.add_argument('--continue-ite', default=1, type=int, help='iteration to continue search from. Default: 1')

# weight initialization for created net
parser.add_argument('--weight-init', default='kaiming-normal', help='weight-initialization scheme. '
                                                                    'Default: kaiming-normal')

# precision for early stopping of a net while training
parser.add_argument('--early-stopping-thresh', default=0.15, type=float, help='threshold for early stopping. '
                                                                              'Default: 0.15')
parser.add_argument('--early-stopping-epoch', default=10, type=int, help='epoch for comparing with early '
                                                                         'stopping threshold. Default: 10')

# individual net training and validation hyper-parameters
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs to run. Default: 70')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size. Default: 128')
parser.add_argument('-wd', '--weight-decay', default=0.0, type=float, metavar='W', help='weight decay. Default: 0.0')
parser.add_argument('-bn', '--batch-norm', default=1e-4, type=float, metavar='BN', help='batch normalization '
                                                                                        'Default 1e-4')
parser.add_argument('--drop-out-drop', default=0.0, type=float, help='drop out drop probability. Default 0.0')
parser.add_argument('-pf', '--print-freq', default=200, type=int, help='print every pf mini-batches. Default: 200')

# learning rate schedule
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate '
                                                                                           'Default: 0.001')

# MetaQNN hyperparameters
parser.add_argument('--q-learning-rate', default=0.1, type=float, help='Q-learning rate. Default: 0.1')
parser.add_argument('--q-discount-factor', default=1.0, type=float, help='Q-learning discount factor. Default: 1.0')

# designed network parameters
parser.add_argument('-min-conv', '--conv-layer-min-limit', default=2, type=int,
                    help='Minimum amount of conv layers in model. Default: 2)')
parser.add_argument('-max-conv', '--conv-layer-max-limit', default=12, type=int,
                    help='Maximum amount of conv layers in model. Default: 12')
parser.add_argument('--max-fc', default=0, type=int, help='maximum number of fully connected layers. Default: 0')
