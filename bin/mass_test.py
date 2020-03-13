import b_meson_fit.mass as mass
import b_meson_fit as bmf
import b_meson_fit.background as background
import numpy as np
import tensorflow.compat.v2 as tf

sig_coeffs = bmf.coeffs.fit(initialization=bmf.coeffs.fit_initialization_same, current_signal_model="SM")
back_coeffs = background.back_coeffs_fit
alpha = tf.constant(0.8, dtype = 'float32')
ndat = tf.constant(10000, dtype = 'float32')
nback = tf.Variable(2000, dtype = 'float32')

coeffs = [*sig_coeffs,*back_coeffs,ndat,nback,alpha]



events = bmf.signal.generate_all(sig_coeffs,back_coeffs,events_total=10000)
print(bmf.signal.exnll(coeffs,events))
