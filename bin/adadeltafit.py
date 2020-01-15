import argparse
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
import tqdm
import numpy as np
from iminuit import Minuit
import time

import b_meson_fit as bmf

signal_count = 24000
signal_coeffs = bmf.coeffs.signal("SM")
print("Input Signals:", [i.numpy() for i in signal_coeffs])
train = Adadelta(learning_rate = 0.001,
                 rho = 0.95,
                 epsilon = 5e-7)
signal_events = bmf.signal.generate(signal_coeffs, events_total=signal_count)

