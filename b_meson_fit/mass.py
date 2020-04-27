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


def signal_mass(mass,mean = 4.,sig = 1. ):
    massprob = tf.exp(-tf.math.square(mass - mean) / (2 * tf.math.square(sig)))/(sig*tf.sqrt(2*np.pi)) 
    return tf.reshape(tf.convert_to_tensor(massprob, dtype = tf.float32),[tf.shape(mass)[0],])


def background_mass(mass,scale = 1. ):
    return tf.exp(-1*mass/scale)/scale
