import itertools
import tensorflow.compat.v2 as tf
import math

CoeffBCK0 = [
    [100.,-10],
    [100.,-80.],
    [100.,-80.],
    [100,-15]
]

back_coeffs_fit = [tf.constant(i, dtype = 'float32')  for j in CoeffBCK0 for i in j]
def pdf(coeffs, events):
    [q2, cos_theta_k, cos_theta_l, phi] = tf.unstack(events, axis=1)
    return ((coeffs[0] + q2*coeffs[1])*(coeffs[2] + cos_theta_k*coeffs[3])*( coeffs[4] + cos_theta_l*coeffs[5])*(coeffs[6] + phi*coeffs[7]))

            
def pdfnorm(coeffs,events):
    coeffs = [tf.constant(i , dtype='float32')  for j in coeffs for i in j]
    qmax = 8
    qmin = 1
    norm = 8*math.pi*coeffs[2]*coeffs[4]*coeffs[6]*((coeffs[0]*(qmax-qmin))+(coeffs[1]*(qmax**2-qmin**2)))
    return pdf(coeffs,events)/norm


