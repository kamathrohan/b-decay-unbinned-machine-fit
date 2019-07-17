import math
import numpy.testing as nt
import tensorflow.compat.v2 as tf
# Import this separately as its old Tensorflow v1 code
from tensorflow.contrib import integrate as tf_integrate
import unittest

import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs

tf.enable_v2_behavior()


class TestSignal(unittest.TestCase):

    # Fields are: name, list of coefficients, "true" decay rate (As generated by odeint_fixed() attempt across all vars)
    # To add more to this list:
    #  1. Add a new line with "true" decay rate set to something like 1.0.
    #  2. Next uncomment the skip() in test_unit_test_coeff_data_provider_decay_rates()
    #     and run the test to generate the decay rates.
    #  3. Finally add the value here and add the skip() again.
    test_coeffs = [
        ('signal', bmfc.signal(), 602.09033, ),
        ('ones', [tf.constant(1.0)] * 36, 3125.6858, ),
        ('integers', [tf.constant(float(i)) for i in range(-18, 18)], 335771.25,),
        ('_minus_point_ones', [tf.constant(-0.1)] * 36, 31.256865,),
    ]

    def test_decay_rate_integration_methods_approx_equal(self):
        """
        Check that the _integrate_decay_rate() method that integrates a previously angle-integrated decay rate
        over q^2 returns something approximately equal to running odeint_fixed() over all variables.

        Only checks to within 1% as both methods use bins and add errors.
        """
        # Check for different lists of coefficients
        for c_name, coeffs, expected_decay_rate in self.test_coeffs:
            with self.subTest(c_name=c_name):
                actual = bmfs._integrate_decay_rate(coeffs)
                # Check values are the same to within 2%
                nt.assert_allclose(expected_decay_rate, actual.numpy(), atol=0, rtol=0.01)

    def test_integral_decay_rate_within_tolerance(self):
        """
        Check that the tolerances set in _integrate_decay_rate() have not been relaxed so much that
        they mess up the accuracy more than 0.1% from using odeint() on the previously angle integrated decay rate.
        """
        for c_name, coeffs, _ in self.test_coeffs:
            with self.subTest(c_name=c_name):
                true = tf_integrate.odeint(
                    lambda _, q2: bmfs._decay_rate_angle_integrated(coeffs, q2),
                    0.0,
                    tf.stack([bmfs.q2_min, bmfs.q2_max]),
                )[1]

                ours = bmfs._integrate_decay_rate(coeffs)

                nt.assert_allclose(true.numpy(), ours.numpy(), atol=0, rtol=0.001)

    def test_generate_returns_correct_shape(self):
        """Check generate() returns a tensor of shape (events_total, 4)"""
        events = bmfs.generate(bmfc.signal(), 123_456)
        self.longMessage = True
        self.assertEqual(123_456, tf.shape(events)[0].numpy())
        self.assertEqual(4, tf.shape(events)[1].numpy())

    def test_unit_test_coeff_data_provider_decay_rates(self):
        """
        Go through our test_coeffs data provider and use odeint_fixed () to across all variables to work out a
        decay rate.

        Ideally this would use odeint(), in which case the test_decay_rate_integration_methods_approx_equal tolerance
        could be reduced. However odeint() results in a 'underflow in dt' error with this data so the value is
        approximated through odeint_fixed().

        Normally skipped. Only meant to be used for adding new cases to the data provider.

        Will take a long while to run.
        """
        self.skipTest('Only meant to be used to generate new coefficient test cases')

        # Ranges we want to integrate over
        q2_range = tf.cast([bmfs.q2_min, bmfs.q2_max], dtype=tf.float32)
        cos_k_range = tf.constant([-1.0, 1.0], dtype=tf.float32)
        cos_l_range = tf.constant([-1.0, 1.0], dtype=tf.float32)
        phi_range = tf.constant([-math.pi, math.pi], dtype=tf.float32)

        # When integrating approximate each variable into bins. Bin sizes found through trial and error
        def dt(r, bins): return (r[1] - r[0]) / bins
        q2_dt = dt(q2_range, 20)
        cos_k_dt = dt(cos_k_range, 10)
        cos_l_dt = dt(cos_l_range, 10)
        phi_dt = dt(phi_range, 6)

        # Massively improve the speed of the test by autographing our decay_rate() function. This does
        #  unfortunately make the test harder to debug
        decay_rate = tf.function(bmfs._decay_rate)

        # Check for different lists of coefficients
        for c_name, coeffs, expected_decay_rate in self.test_coeffs:
            with self.subTest(c_name=c_name):

                with tf.device('/device:GPU:0'):
                    # Integrate decay_rate() over the 4 independent variables
                    # odeint_fixed() is used as the faster and more accurate odeint() resulted in a float underflow
                    full_integrated_rate = tf_integrate.odeint_fixed(
                        lambda _, q2: tf_integrate.odeint_fixed(
                            lambda _, cos_theta_k: tf_integrate.odeint_fixed(
                                lambda _, cos_theta_l: tf_integrate.odeint_fixed(
                                    lambda _, phi: decay_rate(
                                        coeffs,
                                        tf.expand_dims(tf.stack([q2, cos_theta_k, cos_theta_l, phi]), 0)
                                    )[0],
                                    0.0,
                                    phi_range,
                                    phi_dt,
                                    method='midpoint'
                                )[1],
                                0.0,
                                cos_l_range,
                                cos_l_dt,
                                method='midpoint'
                            )[1],
                            0.0,
                            cos_k_range,
                            cos_k_dt,
                            method='midpoint'

                        )[1],
                        0.0,
                        q2_range,
                        q2_dt,
                        method='midpoint'
                    )[1]

                self.assertEqual(expected_decay_rate, full_integrated_rate.numpy())


if __name__ == '__main__':
    unittest.main()
