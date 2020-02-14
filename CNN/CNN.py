import numpy as np
from time import time
import math
from keras.datasets import fashion_mnist


from include.data import get_data_set
from include.model import model, lr


(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

