#!/usr/bin/env python
"""
Plot pull distributions for given CSV result files.

Will also output mean, std err and pull mean for each coefficient.
"""

import argparse
import csv
import itertools
import matplotlib
import numpy as np
import os
import scipy.stats
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()


def filename_and_name(arg):
    try:
        _filename, _name = arg.split(":")
    except ValueError:
        _filename = arg
        _name = os.path.splitext(os.path.basename(_filename))[0]
    return _filename, _name


columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot pull distributions.',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
)
parser.add_argument(
    '-d',
    '--device',
    dest='device',
    default=bmf.Script.device_default,
    help='use this device e.g. CPU:0, GPU:0, GPU:1 (default: {})'.format(bmf.Script.device_default),
)
parser.add_argument(
    '-w',
    '--write-svg',
    dest='write_svg',
    metavar='SVG_PATH',
    help='write plots as SVGs using this filepath. this string must contain \'%%name%%\''
)
parser.add_argument(
    nargs='+',
    dest='plot_list',
    action='append',
    type=filename_and_name,
    metavar='FILENAME[:NAME]',
    help='Filename and optional name to plot (e.g. NP_0.15.csv or NP_0.15.csv:"NP 0.15")'
)
args = parser.parse_args()
if args.write_svg and '%name%' not in args.write_svg:
    parser.error('-w/--write-svg must contain \'%name%\'')

with bmf.Script(device=args.device) as script:
    if args.write_svg is not None:
        matplotlib.use('SVG')

    # Import these after we optionally set SVG backend - otherwise matplotlib may bail on a missing TK backend when
    #  running from the CLI
    import matplotlib.pylab as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    import seaborn as sns

    # Load inputs
    data_points = {}
    signal_coeffs = {}
    for plot in args.plot_list[0]:
        filename, p_name = plot
        with open(filename, newline='') as csv_file:
            reader = csv.DictReader(csv_file)

            # Load signal
            if p_name not in signal_coeffs:
                signal_coeffs[p_name] = {}
            signal = next(reader)
            for c_name in bmf.coeffs.names:
                signal_coeffs[p_name][c_name] = float(signal[c_name])

            # Load fit coefficients
            for row in reader:
                for c_name in bmf.coeffs.names:
                    if row[c_name] == "0.0":
                        continue
                    if c_name not in data_points:
                        data_points[c_name] = {}
                    if p_name not in data_points[c_name]:
                        data_points[c_name][p_name] = []
                    data_points[c_name][p_name].append(float(row[c_name]))

    for c_name in data_points.keys():
        plt.figure()
        # Set style as well as font to Computer Modern Roman to match LaTeX output
        sns.set(style='ticks', font='cmr10', rc={'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

        plt.title(bmf.coeffs.latex_names[bmf.coeffs.names.index(c_name)])

        colors = itertools.cycle(sns.color_palette())

        for name, points in data_points[c_name].items():
            if not all(elem == 0.0 for elem in points):
                mean = np.mean(points)
                std_err = scipy.stats.sem(points, axis=None)
                pull = list(map(lambda p: (p - signal_coeffs[name][c_name]) / std_err, points))
                pull_mean = np.mean(pull)

                bmf.stdout(
                    '{}/{} signal: {} mean: {} std err: {} pull mean: {}'.format(
                        c_name,
                        name,
                        signal_coeffs[name][c_name],
                        mean,
                        std_err,
                        pull_mean,
                    )
                )
                color = next(colors)
                sns.kdeplot(pull, cut=0, color=color, label=name)
                # Draw a dotted line to represent the pull mean
                plt.gca().axvline(pull_mean, color=color, linestyle=':')

        plt.xlabel('Pull')
        plt.ylabel('Density')

        if len(data_points[c_name]) > 1:
            plt.legend()
        else:
            plt.gca().get_legend().set_visible(False)

        if args.write_svg is not None:
            filepath = args.write_svg.replace('%name%', c_name)
            bmf.stdout('Writing {}'.format(filepath))
            plt.savefig(filepath, format='svg', bbox_inches='tight')
        else:
            plt.show()
