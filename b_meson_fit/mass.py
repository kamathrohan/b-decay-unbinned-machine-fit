import numpy as np 
import tensorflow.compat.v2 as tf

def signal_mass_generator(n):
    signal_mass = np.random.normal(loc = 4, scale = 1,size =n )
    signal_mass = tf.convert_to_tensor(signal_mass)
    return signal_mass


def background_mass_generator(n):
    background_mass = np.random.exponential(scale = 1,size =n )
    return signal_mass


