import numpy as np 
import tensorflow.compat.v2 as tf

def signal_mass_generator(n):
    signal_mass = np.random.normal(loc = 4, scale = 1,size =n )
    signal_mass = tf.reshape(tf.convert_to_tensor(signal_mass, dtype = tf.float32),[n,1])
    return signal_mass


def background_mass_generator(n):
    background_mass = np.random.exponential(scale = 1,size =n )
    background_mass = tf.reshape(tf.convert_to_tensor(background_mass, dtype = tf.float32),[n,1])
    return background_mass


def signal_mass(mass,mean,sig):
    return (np.exp(-np.power(mass - mean, 2.) / (2 * np.power(sig, 2.))))/(sig*np.sqrt(2*np.pi))

def background_mass(mass,scale):
    return np.exp(-1*mass/scale)/scale
    