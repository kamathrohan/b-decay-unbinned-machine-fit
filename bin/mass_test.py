import b_meson_fit.mass as mass
import b_meson_fit as bmf
import b_meson_fit.background as background
import numpy as np
import tensorflow.compat.v2 as tf

sig_coeffs = bmf.coeffs.fit(initialization=bmf.coeffs.fit_initialization_same, current_signal_model="SM")
back_coeffs = background.back_coeffs_fit
alpha = tf.constant(0.8)
ndat = tf.constant(1000000)
nback = tf.Variable(10000)

"""
coeffs = [*sig_coeffs,*back_coeffs,alpha,ndat,nback]
for i in coeffs:
    print(bmf.coeffs.is_trainable(i))
"""

events = bmf.signal.generate_all(sig_coeffs,back_coeffs,events_total=10000)
print(events)