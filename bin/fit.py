#!/usr/bin/env python
"""Fit amplitude coefficients to signal events"""

import argparse
import os
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

# Only do plots if running PyCharm
if 'PYCHARM_HOSTED' in os.environ:
    import matplotlib.pylab as plt
    import seaborn as sns

tf.enable_v2_behavior()

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
    '-i',
    '--iterations',
    dest='iterations',
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
    default=20_000,
    help='restart iteration if not converged after this many steps (default: 20000)'
)
parser.add_argument(
    '-s',
    '--signal-count',
    dest='signal_count',
    default=2400,
    help='number of signal events to generated per fit (default: 2400)'
)
args = parser.parse_args()

iteration = 0
with bmf.Script() as script:
    if args.log:
        log = bmf.Log(script.name)

    if args.csv_file is not None:
        writer = bmf.CsvWriter(args.csv_file)
        if writer.current_id >= args.iterations:
            bmf.stdout('{} already contains {} iteration(s)'.format(args.csv_file, args.iterations))
            exit(0)
        iteration = writer.current_id

    while iteration < args.iterations:
        iteration = iteration + 1
        bmf.stdout('Starting iteration {}/{}'.format(iteration, args.iterations))

        signal_coeffs = bmf.coeffs.signal()
        signal_events = bmf.signal.generate(signal_coeffs, events_total=args.signal_count)

        # If running if PyCharm, plot our signal distributions for each independent variable
        if 'PYCHARM_HOSTED' in os.environ:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            fig.suptitle('Signal (Iteration {}/{})'.format(iteration, args.iterations))
            titles = [
                r'$q^2$',
                r'$\cos{\theta_k}$',
                r'$\cos{\theta_l}$',
                r'$\phi$'
            ]
            for ax, feature, title in zip(axes.flatten(), signal_events.numpy().transpose(), titles):
                sns.distplot(feature, ax=ax, bins=20)
                ax.set(title=title)
            plt.show()

        attempt = 1
        converged = False
        while not converged:
            fit_coeffs = bmf.coeffs.fit()
            optimizer = bmf.Optimizer(fit_coeffs, signal_events)

            def print_step():
                bmf.stdout(
                    "Iteration:", "{}/{}". format(iteration, args.iterations),
                    "Attempt", attempt,
                    "Step:", optimizer.step,
                    "normalized_nll:", optimizer.normalized_nll,
                    "grad_max:", optimizer.grad_max,
                )
                bmf.stdout("fit:   ", bmf.coeffs.to_str(fit_coeffs))
                bmf.stdout("signal:", bmf.coeffs.to_str(signal_coeffs))

            print_step()

            while True:
                optimizer.minimize()
                if args.log:
                    log.coefficients('fit_{}'.format(iteration), optimizer, signal_coeffs)
                if optimizer.converged():
                    converged = True
                    print_step()
                    if args.csv_file is not None:
                        writer.write_coeffs(optimizer.normalized_nll, fit_coeffs)
                    break
                if optimizer.step.numpy() % 100 == 0:
                    print_step()
                if optimizer.step >= args.max_step:
                    bmf.stderr('No convergence after {} steps. Restarting iteration'.format(args.max_step))
                    attempt = attempt + 1
                    break
