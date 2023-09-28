import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import warnings
import pandas as pd

from .utils.chrono import get_time_transforms, get_input_and_response
from .utils.scale import get_scale_prefix, get_scale_factor, get_factor_from_prefix
from .utils.eis import construct_eis_df
from .utils import validation
from .preprocessing import identify_steps, estimate_rp


def plot_chrono(data, chrono_mode=None, step_times=None, axes=None, plot_func='scatter', area=None,
                transform_time=False, trans_functions=None, linear_time_axis=False, display_linear_ticks=False,
                linear_tick_kw=None, plot_i=True, plot_v=True, i_sign_convention=1,
                scale_prefix=None, tight_layout=True, **kw):
    times, i_signal, v_signal = process_chrono_plot_data(data)

    if i_signal is None and v_signal is None:
        raise ValueError('At least one of i_signal and v_signal must be provided')
    elif i_signal is None:
        y_vals = [v_signal]
        y_label_parts = [('$v$', 'V')]
    elif v_signal is None:
        y_vals = [i_signal * i_sign_convention]
        y_label_parts = [('$i$', 'A')]
    else:
        y_vals = []
        y_label_parts = []
        if plot_i:
            if area is None:
                y_vals.append(i_signal * i_sign_convention)
                y_label_parts.append(('$i$', 'A'))
            else:
                y_vals.append(i_signal * i_sign_convention / area)
                y_label_parts.append(('$j$', r'A$\cdot \mathrm{cm}^{-2}$'))
        if plot_v:
            y_vals.append(v_signal)
            y_label_parts.append(('$v$', 'V'))
    num_col = len(y_vals)

    # Create figure and axes as needed
    if axes is None:
        fig, axes = plt.subplots(1, num_col, figsize=(4 * num_col, 3))
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].get_figure()

    # Get x values
    if transform_time:
        if trans_functions is not None:
            x = trans_functions[1](times)
        else:
            if chrono_mode is None and step_times is None:
                raise ValueError('One of trans_functions, chrono_mode, or step_times must be specified if '
                                 'transform_time=True')
            if step_times is None:
                # Get step times
                input_signal, response_signal = get_input_and_response(i_signal, v_signal, chrono_mode)
                step_indices = identify_steps(input_signal, allow_consecutive=False)
                step_times = times[step_indices]
            x, trans_functions = get_transformed_plot_time(times, step_times)
    else:
        x = times

    # Set plot_func defaults
    if plot_func == 'scatter':
        defaults = {'s': 10, 'alpha': 0.5}
    else:
        defaults = {}
    # Update with user-supplied kws
    defaults.update(kw)

    # Plot y_val
    for ax, y_val, y_label_tuple in zip(axes, y_vals, y_label_parts):
        # Get scale for y_val
        if scale_prefix is None:
            scale_prefix_y = get_scale_prefix(y_val)
        else:
            scale_prefix_y = scale_prefix
        scale_factor = get_factor_from_prefix(scale_prefix_y)

        func = getattr(ax, plot_func)
        func(x, y_val / scale_factor, **defaults)

        if transform_time:
            ax.set_xlabel('$f(t)$')
        else:
            ax.set_xlabel('Time (s)')

        y_label = f'{y_label_tuple[0]} ({scale_prefix_y}{y_label_tuple[1]})'
        ax.set_ylabel(y_label)

    # Add linear time axis
    if linear_time_axis and transform_time:
        for ax in axes:
            axt = add_linear_time_axis(ax, times, step_times, trans_functions)

    if transform_time and display_linear_ticks:
        if linear_tick_kw is None:
            linear_tick_kw = {}
        for ax in axes:
            display_linear_time_ticks(ax, times, step_times, trans_functions, **linear_tick_kw)

    if tight_layout:
        fig.tight_layout()

    return axes


def process_chrono_plot_data(data):
    times, i_signal, v_signal = None, None, None
    if type(data) in (list, tuple):
        if len(data) == 3:
            times, i_signal, v_signal = data
        else:
            raise ValueError('If data is a tuple, it must be a 3-tuple of time, i_signal, and v_signal arrays')
    elif type(data) == pd.core.frame.DataFrame:
        is_valid_df = True
        time_intersect = np.intersect1d(['elapsed', 'Time', 'T'], list(data.columns))
        if len(time_intersect) == 0:
            is_valid_df = False
        else:
            time_column = time_intersect[-1]
            required_columns = ['Vf', 'Im']
            intersection = np.intersect1d(required_columns, list(data.columns))
            if len(intersection) == len(required_columns):
                times = data[time_column].values
                i_signal = data['Im'].values
                v_signal = data['Vf'].values
            else:
                is_valid_df = False

        if not is_valid_df:
            raise ValueError("If data is a DataFrame, it must contain columns 'Time' or 'T', 'Vf', and 'Im'. "
                             f"Provided data contains columns: {list(data.columns)}")

    validation.check_chrono_data(times, i_signal, v_signal)

    return times, i_signal, v_signal


def get_transformed_plot_time(times, step_times):
    trans2time, time2trans = get_time_transforms(times, step_times)
    x = time2trans(times)
    functions = (trans2time, time2trans)

    return x, functions


def add_linear_time_axis(ax, times, step_times, trans_functions):
    axt = ax.secondary_xaxis('top', functions=trans_functions)
    # if len(step_times) > 1:
    #     step_increment = np.mean(np.diff(step_times))
    # else:
    #     step_increment = step_times[0]
    t_ticks = np.insert(step_times, len(step_times), times[-1])
    # Don't exceed the number of ticks on the primary x-axis
    max_nticks = len(ax.get_xticks())
    factor = int(np.ceil(len(t_ticks) / max_nticks))
    t_ticks = t_ticks[::factor]
    axt.set_xticks(t_ticks)
    axt.set_xlabel('$t$ (s)')

    # axt = ax.twiny()
    # print(ax.get_xlim(), axt.get_xlim())
    # axt.set_xlim(ax.get_xlim())
    # t_ticks = np.round(np.insert(step_times, len(step_times), times[-1]), 2)
    # # Don't exceed the number of ticks on the primary x-axis
    # max_nticks = len(ax.get_xticks())
    # factor = int(np.ceil(len(t_ticks) / max_nticks))
    # t_ticks = t_ticks[::factor]
    #
    # trans2time, time2trans = trans_functions
    # print(t_ticks, time2trans(t_ticks))
    # axt.set_xticks(time2trans(t_ticks))
    # axt.set_xticklabels(t_ticks)
    # print(ax.get_xlim(), axt.get_xlim())
    # axt.set_xlabel('$t$ (s)')

    return axt


def display_linear_time_ticks(ax, times, step_times, trans_functions, step_increment=1, ticks_per_step=9,
                              major_tick_format='.1f'):
    """
    Display linear time ticks and labels on transformed time axis
    :param ax:
    :param times:
    :param step_times:
    :param trans_functions:
    :return:
    """
    trans2time, time2trans = trans_functions

    step_times = step_times[::step_increment]

    major_ticks = np.insert(step_times, len(step_times), times[-1])
    minor_ticks = np.concatenate(
        [np.linspace(major_ticks[i], major_ticks[i + 1], ticks_per_step + 2)[1:-1]
         for i in range(len(major_ticks) - 1)]
    )
    # minor_ticks = np.concatenate(
    #     [major_ticks[i] + np.logspace(, major_ticks[i + 1], ticks_per_step + 2)[1:-1]
    #      for i in range(len(major_ticks) - 1)]
    # )
    # print(major_ticks, minor_ticks)

    major_trans = time2trans(major_ticks)
    # minor_trans = [np.linspace(major_trans[i], major_trans[i + 1], ticks_per_step + 2)[1:-1]
    #                 for i in range(len(major_trans) - 1)]
    minor_trans = time2trans(minor_ticks)

    # minor_times = [trans2time(mt) for mt in minor_trans]
    # minor_deltas = [minor_times[i] - major_ticks[i] for i in range(len(minor_times))]
    # print(minor_deltas)
    # # Round to nearest integer power
    # minor_delta_powers = [np.round(np.log10(md), 0).astype(int) for md in minor_deltas]
    # minor_deltas = [10.0 ** mdp for mdp in minor_delta_powers]
    # # recalculate times with rounded deltas
    # minor_times = [minor_deltas[i] + major_ticks[i] for i in range(len(minor_deltas))]
    # minor_times = np.concatenate(minor_times)
    # minor_deltas = np.concatenate(minor_deltas)
    # minor_delta_powers = np.concatenate(minor_delta_powers)
    # print(minor_times)
    #
    # minor_trans = time2trans(minor_times)

    ax.set_xticks(major_trans)
    # Set tick labels manually. Add small positive to prevent "-0.0" label
    ax.set_xticklabels(['{:{}}'.format(mt + 1e-10, major_tick_format) for mt in major_ticks])

    ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_trans))
    # ax.set_xticklabels(['$+10^{{{}}}$'.format(mdp) for mdp in minor_delta_powers], minor=True)

    # Update axis label to reflect linear time tick values
    ax.set_xlabel('$t$ (s)')


# def plot_drt_result(data, drt, tau_plot=np.logspace(-7, 2, 200), axes=None):
#     if axes is None:
#         fig, axes = plt.subplots(1, 3, figsize=(10, 3))
#     else:
#         fig = axes[0].get_figure()
#
#     if type(data) == pd.core.frame.DataFrame:
#         times = data['T'].values
#         i_signal = data['Im'].values
#         v_signal = data['Vf'].values
#     elif type(data) in (list, tuple):
#         times, i_signal, v_signal = data
#
#     # Plot DRT
#     axes[0].plot(tau_plot, drt.predict_distribution(tau_plot))
#     axes[0].set_xscale('log')
#     axes[0].set_xlabel(r'$\tau$ (s)')
#     axes[0].set_ylabel(r'$\gamma$ ($\Omega$)')
#
#     # Plot fit of response data
#     t0 = 1e-5
#     period = 1
#     factor = period / t0
#     t_plot = (factor ** (times // period)) * (np.mod(times, period) + t0 / 10)
#     # t_plot = times
#     axes[1].scatter(t_plot, v_signal, s=10, alpha=0.5)
#     axes[1].plot(t_plot, drt.predict_response(times), c='k')
#     axes[1].set_xscale('log')
#     axes[1].set_xlabel('$t$ (s)')
#     axes[1].set_ylabel('$V$ (V)')
#
#     # Plot residuals
#     axes[2].scatter(t_plot, drt.predict_response(times) - v_signal)
#     axes[2].set_xscale('log')
#     axes[2].axhline(0, color='k', lw=0.5)
#     axes[2].set_xlabel('$t$ (s)')
#     axes[2].set_ylabel('$\hat{V} - V_{\mathrm{exp}}$')
#
#     fig.tight_layout()
#
#     return axes


def plot_distribution(tau, f, ax=None, area=None, scale_prefix=None, normalize_by=None,
                      freq_axis=False, return_info=False, **kw):
    """
    Generic function for plotting a distribution as a function of tau in log space
    :param tau:
    :param f:
    :param ax:
    :param area:
    :param scale_prefix:
    :param normalize_by:
    :param kw:
    :return:
    """
    if area is not None and normalize_by is not None:
        warnings.warn('If both area and normalize_by are provided, the normalization will not hold')

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.get_figure()

    # Normalize by area
    if area is not None:
        f *= area

    # Normalize by R_p
    if normalize_by is not None:
        f /= normalize_by

    # Scale to appropriate magnitude
    if scale_prefix is None:
        scale_prefix = get_scale_prefix(f)
    scale_factor = get_factor_from_prefix(scale_prefix)

    # Plot distribution
    line = ax.plot(tau, f / scale_factor, **kw)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$ (s)')

    if normalize_by is not None:
        y_label = r'$\gamma \, / \, R_p$'
    else:
        if area is not None:
            y_units = r'$\Omega \cdot \mathrm{cm}^2$'
        else:
            y_units = r'$\Omega$'
        y_label = fr'$\gamma$ ({scale_prefix}{y_units})'
    ax.set_ylabel(y_label)

    # Add frequency axis
    if freq_axis:
        def ft_trans(x):
            return 1 / (2 * np.pi * x)

        freq_ax = ax.secondary_xaxis('top', functions=(ft_trans, ft_trans))
        freq_ax.set_xlabel('$f$ (Hz)')

    if return_info:
        return ax, (line, scale_prefix, scale_factor)
    else:
        return ax


# --------------------
# EIS plotting
# --------------------
def process_eis_plot_data(data):
    if type(data) == tuple:
        if len(data) == 2:
            validation.check_eis_data(*data)
        else:
            raise ValueError('If data is a tuple, must be a 2-tuple of frequency and complex impedance arrays')
    elif type(data) == pd.core.frame.DataFrame:
        required_columns = ['Freq', 'Zreal', 'Zimag', 'Zmod', 'Zphz']
        intersection = np.intersect1d(required_columns, list(data.columns))
        if len(intersection) != len(required_columns):
            raise ValueError(f'If data is a DataFrame, must contain columns: {required_columns}')
    else:
        raise ValueError('data must either be a pandas DataFrame or a 2-tuple of of frequency and complex impedance '
                         'arrays')

    if type(data) == tuple:
        data_out = construct_eis_df(*data)
    else:
        data_out = data.copy()

    return data_out


def plot_nyquist(data, area=None, ax=None, label='', plot_func='scatter', scale_prefix=None, set_aspect_ratio=True,
                 normalize=False, normalize_rp=None, draw_zero_line=True, tight_layout=True, **kw):
    """
    Generate Nyquist plot.

    Parameters
    ----------
    :param data: DataFrame of impedance data or 2-tuple of frequency and complex impedance arrays
    area : float, optional (default: None)
        Active area in cm^2. If provided, plot area-normalized impedance
    ax : matplotlib axis, optional (default: None)
        Axis on which to plot. If None, axis will be created.
    label : str, optional (default: '')
        Label for data
    plot_func : str, optional (default: 'scatter')
        Name of matplotlib.pyplot plotting function to use. Options: 'scatter', 'plot'
    scale_prefix: str, optional (default: 'auto')
        Scaling unit prefix. If 'auto', determine from data.
        Options are 'mu', 'm', '', 'k', 'M', 'G'
    set_aspect_ratio : bool, optional (default: True)
        If True, ensure that visual scale of x and y axes is the same.
        If False, use matplotlib's default scaling.
    kw:
        Keywords to pass to matplotlib.pyplot.plot_func
    """
    df = process_eis_plot_data(data)

    if normalize and area is not None:
        raise ValueError('normalize and area should not be used together')

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.75))
    else:
        fig = ax.get_figure()

    if area is not None:
        # if area given, convert to ASR
        df['Zreal'] *= area
        df['Zimag'] *= area

    if normalize:
        if normalize_rp is None:
            normalize_rp = estimate_rp(None, None, None, None, None, df['Zreal'].values + 1j * df['Zimag'].values)
        df['Zreal'] /= normalize_rp
        df['Zimag'] /= normalize_rp

        if scale_prefix is None:
            scale_prefix = ''

    # get/set unit scale
    if scale_prefix is None:
        z_concat = np.concatenate((df['Zreal'], df['Zimag']))
        scale_prefix = get_scale_prefix(z_concat)
    scale_factor = get_factor_from_prefix(scale_prefix)

    # scale data
    df['Zreal'] /= scale_factor
    df['Zimag'] /= scale_factor

    if plot_func == 'scatter':
        scatter_defaults = {'s': 10, 'alpha': 0.5}
        scatter_defaults.update(kw)
        ax.scatter(df['Zreal'], -df['Zimag'], label=label, **scatter_defaults)
    elif plot_func == 'plot':
        ax.plot(df['Zreal'], -df['Zimag'], label=label, **kw)
    else:
        raise ValueError(f'Invalid plot type {plot_func}. Options are scatter, plot')

    if area is not None:
        ax.set_xlabel(fr'$Z^\prime$ ({scale_prefix}$\Omega\cdot \mathrm{{cm}}^2$)')
        ax.set_ylabel(fr'$-Z^{{\prime\prime}}$ ({scale_prefix}$\Omega\cdot \mathrm{{cm}}^2$)')
    elif normalize:
        ax.set_xlabel(r'$Z^\prime \, / \, R_p$')
        ax.set_ylabel(r'$-Z^{\prime\prime} \, / \, R_p$')
    else:
        ax.set_xlabel(fr'$Z^\prime$ ({scale_prefix}$\Omega$)')
        ax.set_ylabel(fr'$-Z^{{\prime\prime}}$ ({scale_prefix}$\Omega$)')

    if label != '':
        ax.legend()

    # Apply tight_layout before setting aspect ratio
    if tight_layout:
        fig.tight_layout()

    if set_aspect_ratio:
        # make scale of x and y axes the same

        # if data extends beyond axis limits, adjust to capture all data
        ydata_range = df['Zimag'].max() - df['Zimag'].min()
        xdata_range = df['Zreal'].max() - df['Zreal'].min()
        if np.min(-df['Zimag']) < ax.get_ylim()[0]:
            if np.min(-df['Zimag']) >= 0:
                # if data doesn't go negative, don't let y-axis go negative
                ymin = max(0, np.min(-df['Zimag']) - ydata_range * 0.1)
            else:
                ymin = np.min(-df['Zimag']) - ydata_range * 0.1
        else:
            ymin = ax.get_ylim()[0]
        if np.max(-df['Zimag']) > ax.get_ylim()[1]:
            ymax = np.max(-df['Zimag']) + ydata_range * 0.1
        else:
            ymax = ax.get_ylim()[1]
        ax.set_ylim(ymin, ymax)

        if df['Zreal'].min() < ax.get_xlim()[0]:
            if df['Zreal'].min() >= 0:
                # if data doesn't go negative, don't let x-axis go negative
                xmin = max(0, df['Zreal'].min() - xdata_range * 0.1)
            else:
                xmin = df['Zreal'].min() - xdata_range * 0.1
        else:
            xmin = ax.get_xlim()[0]
        if df['Zreal'].max() > ax.get_xlim()[1]:
            xmax = df['Zreal'].max() + xdata_range * 0.1
        else:
            xmax = ax.get_xlim()[1]
        ax.set_xlim(xmin, xmax)

        # # get data range
        # yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
        # xrng = ax.get_xlim()[1] - ax.get_xlim()[0]
        #
        # # get axis dimensions
        # bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # width, height = bbox.width, bbox.height
        #
        # yscale = yrng / height
        # xscale = xrng / width
        #
        # if yscale > xscale:
        #     # expand the x axis
        #     diff = (yscale - xscale) * width
        #     xmin = max(0, ax.get_xlim()[0] - diff / 2)
        #     mindelta = ax.get_xlim()[0] - xmin
        #     xmax = ax.get_xlim()[1] + diff - mindelta
        #
        #     ax.set_xlim(xmin, xmax)
        # elif xscale > yscale:
        #     # expand the y axis
        #     diff = (xscale - yscale) * height
        #     if min(np.min(-df['Zimag']), ax.get_ylim()[0]) >= 0:
        #         # if -Zimag doesn't go negative, don't go negative on y-axis
        #         ymin = max(0, ax.get_ylim()[0] - diff / 2)
        #         mindelta = ax.get_ylim()[0] - ymin
        #         ymax = ax.get_ylim()[1] + diff - mindelta
        #     else:
        #         negrng = abs(ax.get_ylim()[0])
        #         posrng = abs(ax.get_ylim()[1])
        #         negoffset = negrng * diff / (negrng + posrng)
        #         posoffset = posrng * diff / (negrng + posrng)
        #         ymin = ax.get_ylim()[0] - negoffset
        #         ymax = ax.get_ylim()[1] + posoffset
        #
        #     ax.set_ylim(ymin, ymax)

        set_nyquist_aspect(ax, data=df)

    if draw_zero_line and ax.get_ylim()[0] < 0:
        ax.axhline(0, c='k', lw=0.5, zorder=-10, alpha=0.75)

    return ax


def set_nyquist_aspect(ax, set_to_axis=None, data=None, center_coords=None):
    fig = ax.get_figure()

    # get data range
    yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
    xrng = ax.get_xlim()[1] - ax.get_xlim()[0]

    # Center on the given coordinates
    if center_coords is not None:
        x_offset = center_coords[0] - (ax.get_xlim()[0] + 0.5 * xrng)
        y_offset = center_coords[1] - (ax.get_ylim()[0] + 0.5 * yrng)
        ax.set_xlim(*np.array(ax.get_xlim()) + x_offset)
        ax.set_ylim(*np.array(ax.get_ylim()) + y_offset)

    # get axis dimensions
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    yscale = yrng / height
    xscale = xrng / width

    if set_to_axis is None:
        if yscale > xscale:
            set_to_axis = 'y'
        else:
            set_to_axis = 'x'
    elif set_to_axis not in ['x', 'y']:
        raise ValueError(f"If provided, set_to_axis must be either 'x' or 'y'. Received {set_to_axis}")

    if set_to_axis == 'y':
        # expand the x axis
        diff = (yscale - xscale) * width
        xmin = max(0, ax.get_xlim()[0] - diff / 2)
        mindelta = ax.get_xlim()[0] - xmin
        xmax = ax.get_xlim()[1] + diff - mindelta

        ax.set_xlim(xmin, xmax)
    else:
        # expand the y axis
        diff = (xscale - yscale) * height
        if data is None:
            data_min = 0
        else:
            data_min = np.min(-data['Zimag'])

        if min(data_min, ax.get_ylim()[0]) >= 0:
            # if -Zimag doesn't go negative, don't go negative on y-axis
            ymin = max(0, ax.get_ylim()[0] - diff / 2)
            mindelta = ax.get_ylim()[0] - ymin
            ymax = ax.get_ylim()[1] + diff - mindelta
        else:
            negrng = abs(ax.get_ylim()[0])
            posrng = abs(ax.get_ylim()[1])
            negoffset = negrng * diff / (negrng + posrng)
            posoffset = posrng * diff / (negrng + posrng)
            ymin = ax.get_ylim()[0] - negoffset
            ymax = ax.get_ylim()[1] + posoffset

        ax.set_ylim(ymin, ymax)


def plot_bode(data, area=None, axes=None, label='', plot_func='scatter', cols=['Zmod', 'Zphz'], scale_prefix=None,
              invert_phase=True, invert_Zimag=True, log_mod=True, normalize=False, normalize_rp=None,
              tight_layout=True, **kw):
    """
    Generate Bode plots.

    Parameters
    ----------
    :param data: DataFrame of impedance data or 2-tuple of frequency and complex impedance arrays
    area : float, optional (default: None)
        Active area in cm^2. If provided, plot area-normalized impedance.
    axes : array, optional (default: None)
        List or array of axes on which to plot. If None, axes will be created.
    label : str, optional (default: '')
        Label for data
    plot_func : str, optional (default: 'scatter')
        Name of matplotlib.pyplot plotting function to use. Options: 'scatter', 'plot'
    cols : list, optional (default: ['Zmod', 'Zphz'])
        List of data columns to plot. Options: 'Zreal', 'Zimag', 'Zmod', 'Zphz'
    scale_prefix: str, optional (default: 'auto')
        Scaling unit prefix. If 'auto', determine from data.
        Options are 'mu', 'm', '', 'k', 'M', 'G'
    invert_phase : bool, optional (default: True)
        If True, plot negative phase
    invert_Zimag : bool, optional (default: True)
        If True, plot negative Zimag
    kw:
        Keywords to pass to matplotlib.pyplot.plot_func
    """
    df = process_eis_plot_data(data)

    if normalize and area is not None:
        raise ValueError('normalize and area should not be used together')

    # formatting for columns
    col_dict = {'Zmod': {'units': '$\Omega$', 'label': '$|Z|$', 'scale': 'log'},
                'Zphz': {'units': '$^\circ$', 'label': r'$\theta$', 'scale': 'linear'},
                'Zreal': {'units': '$\Omega$', 'label': '$Z^\prime$', 'scale': 'linear'},
                'Zimag': {'units': '$\Omega$', 'label': '$Z^{\prime\prime}$', 'scale': 'linear'}
                }
    if not log_mod:
        col_dict['Zmod']['scale'] = 'linear'

    # if type(axes) not in [list, np.ndarray, tuple] and axes is not None:
    #     axes = [axes]

    if axes is None:
        fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 2.75))
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].get_figure()

    if area is not None:
        for col in ['Zreal', 'Zimag', 'Zmod']:
            if col in df.columns:
                df[col] *= area

    if normalize:
        if normalize_rp is None:
            normalize_rp = estimate_rp(None, None, None, None, None, df['Zreal'].values + 1j * df['Zimag'].values)
        df['Zreal'] /= normalize_rp
        df['Zimag'] /= normalize_rp
        df['Zmod'] /= normalize_rp

        if scale_prefix is None:
            scale_prefix = ''

    # get/set unit scale
    if scale_prefix is None:
        z_concat = np.concatenate((df['Zreal'], df['Zimag']))
        scale_prefix = get_scale_prefix(z_concat)
    scale_factor = get_factor_from_prefix(scale_prefix)

    # scale data
    for col in ['Zreal', 'Zimag', 'Zmod']:
        if col in df.columns:
            df[col] /= scale_factor

    if invert_Zimag:
        df['Zimag'] *= -1

    if invert_phase:
        df['Zphz'] *= -1

    if plot_func == 'scatter':
        scatter_defaults = {'s': 10, 'alpha': 0.5}
        scatter_defaults.update(kw)
        for col, ax in zip(cols, axes):
            ax.scatter(df['Freq'], df[col], label=label, **scatter_defaults)
    elif plot_func == 'plot':
        for col, ax in zip(cols, axes):
            ax.plot(df['Freq'], df[col], label=label, **kw)
    else:
        raise ValueError(f'Invalid plot type {plot_func}. Options are scatter, plot')

    for ax in axes:
        ax.set_xlabel('$f$ (Hz)')
        ax.set_xscale('log')

    def ax_title(column):
        cdict = col_dict.get(column, {})
        if area is not None and cdict.get('units', '') == '$\Omega$':
            title = '{} ({}{}$\cdot\mathrm{{cm}}^2)$'.format(cdict.get('label', column), scale_prefix,
                                                             cdict.get('units', ''))
        elif normalize and cdict.get('units', '') == '$\Omega$':
            title = '{} $\, / \, R_p$'.format(cdict.get('label', column))
        elif cdict.get('units', '') == '$\Omega$':
            title = '{} ({}{})'.format(cdict.get('label', column), scale_prefix, cdict.get('units', ''))
        else:
            title = '{} ({})'.format(cdict.get('label', column), cdict.get('units', 'a.u.'))

        if column == 'Zimag' and invert_Zimag:
            title = '$-$' + title
        elif column == 'Zphz' and invert_phase:
            title = '$-$' + title
        return title

    for col, ax in zip(cols, axes):
        ax.set_ylabel(ax_title(col))
        ax.set_yscale(col_dict.get(col, {}).get('scale', 'linear'))
        if col_dict.get(col, {}).get('scale', 'linear') == 'log':
            # if y-axis is log-scaled, manually set limits
            # sometimes matplotlib gets it wrong
            ymin = min(ax.get_ylim()[0], df[col].min() / 2)
            ymax = max(ax.get_ylim()[1], df[col].max() * 2)
            ax.set_ylim(ymin, ymax)

    for ax in axes:
        # manually set x axis limits - sometimes matplotlib doesn't get them right
        fmin = min(df['Freq'].min(), ax.get_xlim()[0] * 5)
        fmax = max(df['Freq'].max(), ax.get_xlim()[1] / 5)
        ax.set_xlim(fmin / 5, fmax * 5)

    if tight_layout:
        fig.tight_layout()

    return axes


def plot_eis(data, plot_type='all', area=None, axes=None, label='', plot_func='scatter', scale_prefix=None,
             bode_cols=['Zmod', 'Zphz'], set_aspect_ratio=True, normalize=False, normalize_rp=None,
             tight_layout=True, nyquist_kw=None, bode_kw=None, **kw):
    """
    Plot eis data in Nyquist and/or Bode plot(s)
    Parameters
    ----------
    :param data: DataFrame of impedance data or 2-tuple of frequency and complex impedance arrays
    plot_type : str, optional (default: 'all')
        Type of plot(s) to create. Options:
            'all': Nyquist and Bode plots
            'nyquist': Nyquist plot only
            'bode': Bode plots only
    area : float, optional (default: None)
        Active area in cm^2. If provided, plot area-normalized impedance
    axes : array, optional (default: None)
        Axes on which to plot. If None, axes will be created
    label : str, optional (default: '')
        Label for data
    plot_func : str, optional (default: 'scatter')
        Name of matplotlib.pyplot function to use. Options: 'scatter', 'plot'
    scale_prefix: str, optional (default: 'auto')
        Scaling unit prefix. If 'auto', determine from data.
        Options are 'mu', 'm', '', 'k', 'M', 'G'
    bode_cols : list, optional (default: ['Zmod', 'Zphz'])
        List of data columns to plot in Bode plots. Options: 'Zreal', 'Zimag', 'Zmod', 'Zphz'
        Only used if plot_type in ('all', 'bode')
    set_aspect_ratio : bool, optional (default: True)
        If True, ensure that visual scale of x and y axes is the same for Nyquist plot.
        Only used if plot_type in ('all', 'nyquist')
    kw :
        Keywords to pass to matplotlib.pyplot.plot_func

    Returns
    -------

    """
    # Process data
    df = process_eis_plot_data(data)

    if nyquist_kw is None:
        nyquist_kw = {}
    if bode_kw is None:
        bode_kw = {}

    if plot_type == 'bode':
        axes = plot_bode(df, area=area, axes=axes, label=label, plot_func=plot_func, cols=bode_cols,
                         scale_prefix=scale_prefix, normalize=normalize, normalize_rp=normalize_rp,
                         tight_layout=tight_layout, **bode_kw, **kw)
    elif plot_type == 'nyquist':
        axes = plot_nyquist(df, area=area, ax=axes, label=label, plot_func=plot_func, scale_prefix=scale_prefix,
                            set_aspect_ratio=set_aspect_ratio, normalize=normalize, normalize_rp=normalize_rp,
                            tight_layout=tight_layout, **nyquist_kw, **kw)
    elif plot_type == 'all':
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(9, 2.75))
            ax1, ax2, ax3 = axes.ravel()
        else:
            ax1, ax2, ax3 = axes.ravel()
            fig = axes.ravel()[0].get_figure()

        # Nyquist plot
        plot_nyquist(df, area=area, ax=ax1, label=label, plot_func=plot_func, scale_prefix=scale_prefix,
                     set_aspect_ratio=set_aspect_ratio, normalize=normalize, normalize_rp=normalize_rp,
                     tight_layout=tight_layout, **nyquist_kw, **kw)

        # Bode plots
        plot_bode(df, area=area, axes=(ax2, ax3), label=label, plot_func=plot_func, cols=bode_cols,
                  scale_prefix=scale_prefix, normalize=normalize, normalize_rp=normalize_rp, tight_layout=tight_layout,
                  **bode_kw, **kw)

        # fig.tight_layout()
    else:
        raise ValueError(f'Invalid plot_type {plot_type}. Options: all, bode, nyquist')

    return axes
