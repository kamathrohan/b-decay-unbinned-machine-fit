#!/usr/bin/env python
"""Fit amplitude coefficients to signal events"""

#coucou

import argparse
import shutil
import tensorflow.compat.v2 as tf
import tqdm
from iminuit import Minuit


import b_meson_fit as bmf

tf.enable_v2_behavior()

def fit_init_value(arg):  # Handle --fit-init argument
    if arg in bmf.coeffs.fit_init_schemes:
        return arg
    try:
        init_value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            '{} is not one of {}'.format(arg, ",".join(bmf.coeffs.fit_init_schemes + ['FLOAT']))
        )
    return init_value


columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Fit coefficients to generated toy signal(s).',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
)
parser.add_argument(
    '-c',
    '--csv',
    dest='csv_file',
    help='write results to this CSV file'
)
parser.add_argument(
    '-d',
    '--device',
    dest='device',
    default=bmf.Script.device_default,
    help='use this device e.g. CPU:0, GPU:0, GPU:1 (default: {})'.format(bmf.Script.device_default),
)
parser.add_argument(
    '-f',
    '--fit-init',
    dest='fit_init',
    type=fit_init_value,
    metavar='FIT_INIT',
    default=bmf.coeffs.fit_initialization_scheme_default,
    help='fit coefficient initialization. FIT_INIT should be one of {} (default: {})'.format(
        bmf.coeffs.fit_init_schemes + ['FLOAT'],
        bmf.coeffs.fit_initialization_scheme_default
    )
)
parser.add_argument(
    '-i',
    '--iterations',
    dest='iterations',
    type=int,
    default=1,
    help='number of iterations to run (default: 1)'
)
parser.add_argument(
    '-l',
    '--log',
    dest='log',
    action='store_true',
    help='store logs for Tensorboard (has large performance hit)'
)
parser.add_argument(
    '-m',
    '--max-step',
    dest='max_step',
    type=int,
    default=20000,
    help='restart iteration if not converged after this many steps (default: 20000)'
)
parser.add_argument(
    '-o',
    '--opt-name',
    dest='opt_name',
    default=bmf.Optimizer.opt_name_default,
    help='optimizer algorithm to use (default: {})'.format(bmf.Optimizer.opt_name_default),
)
parser.add_argument(
    '-p',
    '--opt-param',
    nargs=2,
    dest='opt_params',
    action='append',
    metavar=('PARAM_NAME', 'VALUE'),
    help='additional params to pass to optimizer - can be specified multiple times'
)
parser.add_argument(
    '-P',
    '--grad-clip',
    dest='grad_clip',
    type=float,
    help='clip gradients by this global norm'
)
parser.add_argument(
    '-r',
    '--learning-rate',
    dest='learning_rate',
    type=float,
    default=bmf.Optimizer.learning_rate_default,
    help='optimizer learning rate (default: {})'.format(bmf.Optimizer.learning_rate_default),
)
parser.add_argument(
    '-s',
    '--signal-count',
    dest='signal_count',
    type=int,
    default=2400,
    help='number of signal events to generated per fit (default: 2400)'
)
parser.add_argument(
    '-S',
    '--signal-model',
    dest='signal_model',
    choices=bmf.coeffs.signal_models,
    default=bmf.coeffs.SM,
    help='signal model (default: {})'.format(bmf.coeffs.SM)
)
parser.add_argument(
    '-u',
    '--grad-max-cutoff',
    dest='grad_max_cutoff',
    type=float,
    default=bmf.Optimizer.grad_max_cutoff_default,
    help='count fit as converged when max gradient is less than this ' +
         '(default: {})'.format(bmf.Optimizer.grad_max_cutoff_default),
)
args = parser.parse_args()

# Convert optimizer params to dict
opt_params = {}
if args.opt_params:
    for idx, _ in enumerate(args.opt_params):
        # Change any opt param values to floats if possible
        try:
            args.opt_params[idx][1] = float(args.opt_params[idx][1])
        except ValueError:
            pass
        opt_params[args.opt_params[idx][0]] = args.opt_params[idx][1]

iteration = 0
with bmf.Script(device=args.device) as script:
    if args.log:
        log = bmf.Log(script.name)

    signal_coeffs = bmf.coeffs.signal(args.signal_model)

    if args.csv_file is not None:
        writer = bmf.FitWriter(args.csv_file, signal_coeffs)
        if writer.current_id > 0:
            bmf.stdout('{} already contains {} iteration(s)'.format(args.csv_file, writer.current_id))
            bmf.stdout('')
            if writer.current_id >= args.iterations:
                bmf.stderr('Nothing to do')
                exit(0)
            iteration = writer.current_id

    # Show progress bar for fits
    for iteration in tqdm.trange(
            iteration + 1,
            args.iterations + 1,
            initial=iteration,
            total=args.iterations,
            unit='fit'
    ):
        # Time each iteration for CSV writing
        script.timer_start('fit')

        signal_events = bmf.signal.generate(signal_coeffs, events_total=args.signal_count)

        attempt = 1
        converged = False
        while not converged:
            fit_coeffs = bmf.coeffs.fit(args.fit_init, args.signal_model)
            optimizer = bmf.Optimizer(
                fit_coeffs,
                signal_events,
                opt_name=args.opt_name,
                learning_rate=args.learning_rate,
                opt_params=opt_params,
                grad_clip=args.grad_clip,
                grad_max_cutoff=args.grad_max_cutoff
            )

            while True:
                optimizer.minimize()

                # print the log likelihood as we minimize it 
                #print(optimizer.normalized_nll)

                
                if args.log:
                    log.coefficients('fit_{}'.format(iteration), optimizer, signal_coeffs)
                if optimizer.converged():
                    converged = True
                    if args.csv_file is not None:

                        writer.write_coeffs(optimizer.normalized_nll.numpy(), fit_coeffs, script.timer_elapsed('fit'))
                    break
                if optimizer.step >= args.max_step:
                    bmf.stderr('No convergence after {} steps. Restarting iteration'.format(args.max_step))
                    attempt = attempt + 1
                    if args.fit_init not in bmf.coeffs.fit_init_schemes_with_randomization:
                        # If this scheme doesn't randomise coefficients, then restarting with the same signal will
                        #  lead to the same result.
                        bmf.stderr('{} initialisation used so generating new signal'.format(args.fit_init))
                        signal_events = bmf.signal.generate(signal_coeffs, events_total=args.signal_count)
                    break


# First attempt to return the Hessian of the LL for error computation 
#print(optimizer.trainables)

bol0=0
if bol0==1 : 
    # Returns the Hessian of the NLL
    HessianLL = optimizer.get_hessian()
    
    Sig = tf.linalg.inv(HessianLL)
    
    errors = tf.linalg.diag_part(Sig)

    """
    print("The covariant matrix is given by :")
    print(Sig)
    """

    print("The diagonal components of the covariant matrix are :")
    print(errors)


    """
    print("Diagonal Hessian components are is given by :")
    for i in range(10):
        print(Hessian[i][i])
    """


    EIGEN=tf.linalg.eigh(HessianLL)
    print("Eigenvalues of Hessian are :")
    print(EIGEN[0])


# Second try, computing individual single and double derivatives for w.r.t. parameter of index i 
# Comparing to Hessian i bol0 seems OK 
bol1=0
if bol1==1 : 
    #calculate single and double derivatives with respect to the nth parameter 
    for i in range(10) : 
        with tf.GradientTape() as tape1 : 
            with tf.GradientTape() as tape2 :
                normalized_nll = optimizer._normalized_nll()
            grad1=tape2.gradient(normalized_nll , optimizer.trainables[i])
            grad2=tape1.gradient(grad1 , optimizer.trainables[i])
        PRINT="double derivative along "+ str(i)
        print(PRINT)
        print(grad2)
optcoeff = [i.numpy() for i in optimizer.fit_coeffs]

def minuitnll(coeffs):
    return bmf.signal.nll(coeffs, signal_events).numpy()
m = Minuit.from_array_func(minuitnll,optcoeff,errordef=0.5, pedantic = False)
m.migrad()
m.hesse()
print(optcoeff)
print(m.values)
print(m.errors)
