import itertools
import tensorflow.compat.v2 as tf



def pdf(coeffs, events):
    [q2, cos_theta_k, cos_theta_l, phi] = tf.unstack(events, axis=1)
    return (coeffs[0][0] + q2*coeffs[0][1]
            +coeffs[1][0] + cos_theta_k*coeffs[1][1]    
            +coeffs[2][0] + cos_theta_l*coeffs[2][1]
            +coeffs[3][0] + phi*coeffs[3][1])


 
            
