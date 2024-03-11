import argparse
import copy
import datetime
import sys
from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
import uuid
import warnings

import numpy as np
import polars as pl
import pandas as pd  # Polars is broken for reading Excel, need to use Pandas to load.
import matplotlib.pyplot as plt
import prql_python

import data
import utils
import wavelet


def transform(x: np.ndarray, freqs_hz: np.ndarray, scales_s: np.ndarray, nphase: int, dt: float, log2_pres: bool,
              mother: wavelet.Wavelet, mesaclip_min_cycles: int,
              fig_filename: Path, **kwargs) -> (np.ndarray, np.ndarray):

    nfreq = len(freqs_hz)
    phase_edges = np.linspace(-np.pi, np.pi, nphase + 1)
    dphase = phase_edges[1] - phase_edges[0]
    cplx_dtype = np.promote_types(x.dtype, np.complex64)

    nchan, nsamp = x.shape

    # transform pressures with log2
    if log2_pres:
        x[x < 1] = 1
        x = np.log2(x)

    # cwt
    w = np.zeros((nchan, nfreq, nsamp), dtype=cplx_dtype)
    for i in range(nchan):
        w[i, :, :], coimask, _, _ = wavelet.cwt(x[i, :], dt, scales_s, mother, freqs_hz, mesaclip_min_cycles)

    # xwt
    pairs = list(zip(range(nchan), range(1, nchan)))
    npair = len(pairs)
    wxy = np.zeros((npair, nfreq, nsamp), dtype=cplx_dtype)
    for pair_idx, (chan0, chan1) in enumerate(pairs):
        wx = w[chan0, :, :]
        wy = w[chan1, :, :]
        wxy[pair_idx, :, :] = wx * wy.conj()

    # calculate freq_phases based on histogram
    phase = np.angle(wxy)
    cross_power = np.abs(wxy)
    freq_phase_power = np.zeros((nfreq, nphase))
    for fi in range(nfreq):
        h, _ = np.histogram(phase[:, fi, :].flatten(), bins=phase_edges, weights=cross_power[:, fi, :].flatten())
        freq_phase_power[fi, :] = h / (dphase * npair * nsamp)

    # power over frequency
    freq_power = np.mean(np.abs(w)**2, axis=(0, 2))

    # plot for reference
    if fig_filename is not None:
        utils.ensure_path(fig_filename.parent)

        fig = plot_wxy(x, dt, w, freq_power, freq_phase_power, freqs_hz * 60, log2_pres, sep=1,
                       title=fig_filename.with_suffix('').name.replace('_', ' '), **kwargs)
        fig.savefig(fig_filename, dpi=kwargs.get('dpi', None))
        plt.close(fig)

    return freq_power, freq_phase_power


def plot_wxy(x: np.ndarray, dt: float, w: np.ndarray, freq_power: np.ndarray, freq_phase_power: np.ndarray,
             freqs_cpm: np.ndarray, log2_pres: bool,
             title=None, sep=None, x_range=None, w_range=None, f_range=None, fp_range=None, **_):

    gs_kwargs = dict(height_ratios=[0.5, 0.5, 1.5], top=0.93, bottom=0.07, hspace=0.4)

    fig = plt.figure(figsize=(8, 9))
    gs = fig.add_gridspec(3, 2, left=0.1, right=0.85, wspace=0.2, width_ratios=[0.4, 1], **gs_kwargs)
    gsc = fig.add_gridspec(3, 1, left=0.88, right=0.9, **gs_kwargs)

    ax_x = fig.add_subplot(gs[0, :])
    ax_w = fig.add_subplot(gs[1, :])
    ax_f = fig.add_subplot(gs[2, 0])
    ax_fp = fig.add_subplot(gs[2, 1])

    cax_x = fig.add_subplot(gsc[0, 0])
    cax_w = fig.add_subplot(gsc[1, 0])
    cax_fp = fig.add_subplot(gsc[2, 0])

    t = np.arange(x.shape[1]) * dt

    nfreq, nphase = freq_phase_power.shape
    phase_edges = np.linspace(-np.pi, np.pi, nphase + 1)
    phase_centers = utils.edges_to_centers(phase_edges)

    # plot data
    chans = np.arange(x.shape[0])
    grid_time, grid_chan = utils.edge_meshgrid(t / 60, chans + 1)
    if x_range is None:
        x_range = (np.min(x), np.max(x))
    h_x = ax_x.pcolormesh(grid_time, grid_chan, x, vmin=x_range[0], vmax=x_range[1], rasterized=True)
    cb_x = fig.colorbar(h_x, cax=cax_x, use_gridspec=True)
    if log2_pres:
        cb_x.set_label('Amplitude\n(log₂(mmHg))')
    else:
        cb_x.set_label('Amplitude\n(mmHg)')
    ax_x.set_xlabel('Time (minute)')
    ax_x.set_ylabel('Channel')
    ax_x.invert_yaxis()
    ax_x.set_title('Pressures')
    if title is not None:
        fig.suptitle(title, x=0.025, ha='left', fontweight='bold')

    # create ticks, labels, and grid for freq and phase
    freq_order1 = int(round(np.log2(freqs_cpm[0])))
    freq_order2 = int(round(np.log2(freqs_cpm[-1])))
    freq_exp = np.arange(freq_order1, freq_order2 + 1, dtype=np.int32)
    # TODO ugly hacky increase tick resolution for now, really need to do a proper job later on
    if len(freq_exp) < 4:
        freq_ticks = np.arange(round(freqs_cpm[0]), round(freqs_cpm[-1]) + 1)
        freq_labels = [f'{f:.0f}' for f in freq_ticks]
    else:
        freq_ticks = 2 ** freq_exp.astype('f8')
        freq_labels = ['{}'.format(2**fe) if fe >= 0 else '1/{}'.format(2**(-fe)) for fe in freq_exp]
    phase_ticks = np.pi * np.array([-1, -0.5, 0, 0.5, 1])
    phase_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    grid_phase, grid_freq = utils.edge_meshgrid(phase_centers, freqs_cpm, logy=True)
    freq_edges = utils.centers_to_edges(freqs_cpm, log=True)

    # plot frequency over time
    grid_ft_time, grid_ft_freq = utils.edge_meshgrid(t / 60, freqs_cpm, logy=True)
    f = np.sqrt(np.mean(np.abs(w)**2, axis=0))
    if w_range is None:
        w_range = (np.min(f), np.max(f))
    h_w = ax_w.pcolormesh(grid_ft_time, grid_ft_freq, f, vmin=w_range[0], vmax=w_range[1], rasterized=True)
    cb_w = fig.colorbar(h_w, cax=cax_w, use_gridspec=True)
    if log2_pres:
        cb_w.set_label('Power\n(log₂(mmHg)²)')
    else:
        cb_w.set_label('Power\n(mmHg²)')
    ax_w.set_yscale('log', base=2)
    ax_w.set_ylim(freq_edges[0], freq_edges[-1])
    ax_w.set_yticks(freq_ticks)
    ax_w.set_yticklabels(freq_labels)
    ax_w.set_xlabel('Time (minute)')
    ax_w.set_title('Wavelet Cross-Spectrum (Channel average)')

    # plot frequency amplitudes
    ax_f.plot(freq_power, freqs_cpm)
    ax_f.set_yscale('log', base=2)
    ax_f.set_ylim(freq_edges[0], freq_edges[-1])
    ax_f.set_yticks(freq_ticks)
    ax_f.set_yticklabels(freq_labels)
    if f_range is not None:
        ax_f.set_xlim(f_range)
    if log2_pres:
        ax_f.set_xlabel('Power (log₂(mmHg)²)')
    else:
        ax_f.set_xlabel('Power (mmHg²)')
    ax_f.set_title('Global Wavelet\nSpectrum')

    # plot freq_phase_amp
    if fp_range is None:
        good_idxs = np.isfinite(freq_phase_power)
        fp_range = freq_phase_power[good_idxs].min(), freq_phase_power[good_idxs].max()
    h_fp = ax_fp.pcolormesh(grid_phase, grid_freq, freq_phase_power,
                            vmin=fp_range[0], vmax=fp_range[1], rasterized=True)
    cb_fp = fig.colorbar(h_fp, cax=cax_fp, use_gridspec=True)
    if log2_pres:
        cb_fp.set_label('Power (log₂(mmHg)²)')
    else:
        cb_fp.set_label('Power (mmHg²)')
    ax_fp.set_title('Global Wavelet Cross-Spectrum')
    ax_fp.set_xlabel('Phase Difference (rad)')

    # plot isovelocity lines
    if sep is not None:
        velocities = [1, 3, 10, 30, 100]  # cm/min when sep is in cm
        for vel in velocities:
            vel_phases = 2 * np.pi * sep * freq_edges / vel
            ax_fp.plot(vel_phases, freq_edges, c='w', lw=1, ls=':', alpha=0.5)
            ax_fp.plot(-vel_phases, freq_edges, c='w', lw=1, ls=':', alpha=0.5)

    ax_fp.axvline(0, color='w', alpha=0.2)
    ax_fp.set_yscale('log', base=2)
    ax_fp.set_yticks(freq_ticks)
    ax_fp.set_yticklabels(freq_labels)
    ax_fp.set_xticks(phase_ticks)
    ax_fp.set_xticklabels(phase_labels)
    ax_fp.set_xlim(phase_edges[0], phase_edges[-1])  # must limit here since isovelocity lines extend past +-pi

    for ax in [ax_w, ax_f]:
        ax.set_ylabel('Frequency (cpm)')

    return fig


def time_to_seconds(x: str | datetime.time) -> float:
    ty = type(x)
    if ty is datetime.time:
        h, m, s = x.hour, x.minute, x.second
        seconds = float(h) * 3600 + float(m) * 60 + float(s)
    elif ty is str:
        if ':' in x:
            h, m, s = x.split(':')
            seconds = float(h) * 3600 + float(m) * 60 + float(s)
        else:
            # Force whitespace separation between elements.
            x = x.replace('h', 'h ').replace('m', 'm ').replace('s', 's ')

            # Split on whitespace, and parse individual items.
            seconds = 0
            for x in x.split():
                match x[-1]:
                    case 'h':
                        seconds += float(x[:-1]) * 60 * 60
                    case 'm':
                        seconds += float(x[:-1]) * 60
                    case 's':
                        seconds += float(x[:-1])
                    case _:
                        raise RuntimeError(f'could not parse as time the entry "{x}"')
    else:
        raise RuntimeError(f'unknown time type: {ty}')
    return seconds


@dataclass
class AssembledTable:
    df: pl.DataFrame
    time_value_column: str
    chan_value_column: str


def assemble_table(config: dict, df: pl.DataFrame) -> AssembledTable:
    # Parse time columns.
    time_columns = config['data']['time_columns']
    for column in time_columns:
        df = df.with_columns(pl.col(column).map_elements(time_to_seconds))

    # Parse [time].
    time_variable = config['time']['name']
    time_def = config['time']['level']
    time_uuid = str(uuid.uuid4()).replace('-', '')  # Removing '-' so that eval works.
    value_vars_to_melt = []  # Keeps a record of new rows created holding the time ranges to melt.

    # Each level in the categorical variable will contain the level str identifier and a time range.
    # The column name will contain the level and the values contain the ranges... until melted later.
    for level, level_def in time_def.items():
        if type(level_def) is not list or len(level_def) != 2:
            raise RuntimeError(f'{time_variable}.{level} needs to be a list of 2 items, '
                               'the start and end time')

        time_start_column = f'{level}_start{time_uuid}'
        time_end_column = f'{level}_end{time_uuid}'

        # `level_dev` is a list of two strings, each of which may contain an expression,
        # so we evaluate each expression using SQL.
        df_values = pl.SQLContext(frame=df).execute(
            prql_python.compile(f'''
                from frame
                derive {{
                    {time_start_column} = {level_def[0]},
                    {time_end_column} = {level_def[1]},
                }}
                select {{{time_start_column}, {time_end_column}}}
                ''')).collect()

        # Combine the start and end columns into a single column, and add it to the dataframe.
        value_var = f'{level}{time_uuid}'
        df = df.with_columns(df_values.select(pl.concat_list(
            pl.col(time_start_column, time_end_column)).alias(value_var)))

        # Append the column name for melting.
        value_vars_to_melt.append(value_var)

    # Melt all category levels into a single column (keeping time-ranges in another column).
    time_value_column = f'{time_variable}_{time_uuid}'
    df = df.melt(id_vars=[c for c in df.columns if c not in value_vars_to_melt], value_vars=value_vars_to_melt,
                 variable_name=time_variable, value_name=time_value_column)

    # Strip off uuids in the categorical column.
    df = df.with_columns(pl.col(time_variable).map_elements(lambda x: x[:-len(time_uuid)]))

    # Parse [chan]
    # Channels must be in the format "start-end", so just melt for now.
    chan_variable = config['chan']['name']
    chan_def = config['chan']['level']
    chan_uuid = str(uuid.uuid4()).replace('-', '')  # Removing '-' so that eval works.
    chan_value_column = f'{chan_variable}_{chan_uuid}'
    df = df.melt(id_vars=[c for c in df.columns if c not in chan_def], value_vars=chan_def,
                 variable_name=chan_variable, value_name=chan_value_column)

    # Drop missing records that were melted into a null.
    null_times = df[time_value_column].map_elements(lambda x: x.is_null().all())
    null_chans = df[chan_value_column].is_null()
    nulls = null_times | null_chans
    df = df.filter(~nulls)

    # Parse chan_value_column from "start-end" to (start, end).
    df = df.with_columns(pl.col(chan_value_column).map_elements(lambda x: [int(c) for c in x.split('-')]))

    # Combine filename, chan, and time to make a unit identifier which will be used to store wavelet data and
    # retrieve it for sampling.
    filenames_column = config['data']['filenames_column']
    df = df.with_columns(pl.concat_str(pl.col(filenames_column, chan_variable, time_variable), separator='_').alias(
        config['data']['unit_id_column']))

    return AssembledTable(df=df, chan_value_column=chan_value_column, time_value_column=time_value_column)


def main():
    # pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(100)
    pl.Config.set_tbl_width_chars(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename')
    args = parser.parse_args()

    # Load config toml and change the cwd to its location.
    config_filename = Path(args.config_filename)
    print(f'reading wavelet config file: "{config_filename}"')
    config = tomllib.loads(config_filename.read_text(encoding='utf-8'))
    os.chdir(config_filename.parent)

    input_path = Path(config['data']['input_path'])
    if 'outpath' in config['data']:
        output_path = Path(config['data']['output_path'])
    else:
        output_path = Path(config_filename.stem)
    output_images_path = output_path / 'images'
    utils.ensure_path(output_images_path)

    # Load table.
    table_filename = Path(config['data']['table_filename'])
    if table_filename.suffix == '.csv':
        df = pl.read_csv(table_filename)
    else:
        # Polars does not read Excel properly, neither does it convert from Pandas properly, so we need to do this
        # awkward conversion.
        df = pd.read_excel(table_filename)
        df_strs = dict()
        for col in df.columns:
            strs = df[col].astype(str)
            strs[df[col].isnull()] = None
            df_strs[col] = list(strs)
        df = pl.DataFrame(df_strs)

    # Assemble table for processing data.
    table = assemble_table(config, df)
    df = table.df
    unit_id_col = config['data']['unit_id_column']

    filename_for_dt = None
    check_dt = None

    blr_config = config['preproc']['baseline_removal']
    blr_sigma = blr_config['sigma']
    blr_iters = blr_config['iters']
    assert blr_config['method'] == 'gauss'  # TODO add other options

    sync_rem = config['preproc']['sync_rem'] == 'joint'  # TODO add other options

    # Load wavelet config, using deepcopy since we modify params for make_freqs_hz.
    wavelet_params = copy.deepcopy(config['wavelet'])
    if type(wavelet_params['freq_min_cpm']) is str:
        wavelet_params['freq_min_cpm'] = eval(wavelet_params['freq_min_cpm'])
    if type(wavelet_params['freq_max_cpm']) is str:
        wavelet_params['freq_max_cpm'] = eval(wavelet_params['freq_max_cpm'])

    freqs_hz = wavelet.make_freqs_hz(**wavelet_params)
    nphase = len(freqs_hz) + 1  # +1 so that phase and freq have different sizes for sanity-checking plots
    mother = eval('wavelet.' + wavelet_params['family'])(**wavelet_params['params'])

    # Write wavelet parameters so that we can recreate freqs and corretly label plots for pressure log2.
    params_filename = output_path / 'params.toml'
    with open(params_filename, 'w') as f:
        # Write table params.
        f.write('[table]\n')
        f.write(f'unit_id_column = "{config["data"]["unit_id_column"]}"\n\n')

        # Write wavelet params.
        f.write('[wavelet]\n')
        for key in ['freq_min_cpm', 'freq_max_cpm', 'intervals_per_octave', 'log2_pres']:
            value = config['wavelet'][key]
            if type(value) is str:
                value = f'"{value}"'
            if type(value) is bool:
                value = 'true' if value else 'false'
            f.write(f'{key} = {value}\n')
        f.write(f'nphase = {nphase}\n')
    print(f'wrote "{params_filename}"')

    # For saving peaks to table.
    peak_measures = dict()
    freqs_cpm = freqs_hz * 60
    phase_edges = np.linspace(-np.pi, np.pi, nphase + 1)
    phase_centres = utils.edges_to_centers(phase_edges)

    # For displaying progress.
    unit_i = 0
    nunits = len(df)

    for (data_basename, ), df_recording in df.group_by([config['data']['filenames_column']], maintain_order=True):

        # TODO check that we don't already have values for this.

        print(f'{unit_i}/{nunits}: computing {data_basename}:')

        # load filename
        filename = (input_path / data_basename).with_suffix('.txt')
        t, _, x = data.load_hrm_txt(filename)

        diff_t = np.diff(t)
        dt = np.median(diff_t)

        if check_dt is None:
            check_dt = dt
            filename_for_dt = filename

        # Ensure sample rate is consistent throughout.
        if (np.min(diff_t) / np.max(diff_t) - 1.) > 1e-6:
            raise RuntimeError('Inconsistent sampling rate')

        # Ensure expected sampling rate.
        if np.abs(dt / check_dt - 1.0) > 1e-6:
            raise RuntimeError(f'Time-step in "{filename}" ({dt}) '
                               f'inconsistent with that in "{filename_for_dt}" ({check_dt})')

        # Ensure no nans.
        if np.sum(np.isnan(x)) > 0:
            raise RuntimeError('There are NaNs in the pressures')

        # pre-process with baseline and synchronous anomaly removal
        x = data.clean_pressures(x, sigma_samples=blr_sigma/dt, iters=blr_iters, sync_rem=sync_rem)

        # For each unit.
        for row in df_recording.rows(named=True):
            unit_id = row[unit_id_col]

            unit_i += 1
            print(f'{unit_i}/{nunits}:   {unit_id}')

            # Extract the unit's pressures.
            chan_start, chan_end = row[table.chan_value_column]  # 1-based end-inclusive indexing
            sec_start, sec_end = row[table.time_value_column]
            samp_start = np.searchsorted(t, sec_start, side='left')
            samp_end = np.searchsorted(t, sec_end, side='right')

            # Ensure chan_start is positive.
            if chan_start < 1:
                raise RuntimeError(f'Unit {unit_id} has chan start at {chan_start}, minimum needs to be 1')

            # Ensure we're within recording bounds.
            if sec_start < t[0]:
                raise RuntimeError(f'Unit {unit_id} has start time at {sec_start}, but min is {t[0]}')
            if sec_end > t[-1]:
                raise RuntimeError(f'Unit {unit_id} has end time at {sec_end}, but max is {t[-1]}')

            # `-1` to switch to 0-based indexing, leaving chan_end alone since we also switch to end exclusive.
            x_unit = x[(chan_start-1):chan_end, samp_start:samp_end]

            # Compute the wavelet transform.
            scales_s = wavelet.make_scales_s(wavelet_params['intervals_per_octave'], sec_end - sec_start, dt)
            fig_filename = (output_images_path / unit_id).with_suffix('.png')
            f, fp = transform(x=x_unit, freqs_hz=freqs_hz, scales_s=scales_s, nphase=nphase, mother=mother, dt=dt,
                              fig_filename=fig_filename, **wavelet_params)

            # Calculate peak measures.
            f_i = np.argmax(f)
            fp_i, fp_j = np.unravel_index(np.argmax(fp), fp.shape)  # i = freq_idx, j = phase_idx
            peak_measures[unit_id] = (
                freqs_cpm[f_i],       # 1D peak frequency
                f[f_i],               # 1D peak power
                freqs_cpm[fp_i],      # 2D peak frequency
                phase_centres[fp_j],  # 2D peak phase
                fp[fp_i, fp_j]        # 2D peak power
            )

            # Pickle result.
            pkl_filename = (output_path / 'data' / unit_id).with_suffix('.pkl')
            utils.pkl_save((f, fp), pkl_filename)

    # Append to table for use in Stan.
    if 'nchan' not in df.columns:
        df = df.with_columns(pl.col(table.chan_value_column).map_elements(lambda v: v[1] - v[0] + 1).alias('nchan'))
    else:
        print(f'column "nchan" already exists in table, skipping', file=sys.stderr)
    if 'hours' not in df.columns:
        df = df.with_columns(
            pl.col(table.time_value_column).map_elements(lambda v: (v[1] - v[0]) / 3600).alias('hours'))
    else:
        print(f'column "hours" already exists in table, skipping', file=sys.stderr)
    if 'minutes' not in df.columns:
        df = df.with_columns(
            pl.col(table.time_value_column).map_elements(lambda v: (v[1] - v[0]) / 60).alias('minutes'))
    else:
        print(f'column "minutes" already exists in table, skipping', file=sys.stderr)
    df = df.drop([table.chan_value_column, table.time_value_column])

    # Append peak measures.
    with warnings.catch_warnings():
        # Polars does not correctly identify the mapping below, so filter the unnecessary warning.
        warnings.filterwarnings('ignore', category=pl.exceptions.PolarsInefficientMapWarning)
        for i, label in enumerate(['peak1d_freq', 'peak1d_power', 'peak2d_freq', 'peak2d_phase', 'peak2d_power']):
            if label not in df.columns:
                df = df.with_columns(pl.col(unit_id_col).map_elements(lambda u: peak_measures[u][i]).alias(label))
            else:
                print(f'column "{label}" already exists in table, skipping', file=sys.stderr)

    # Write table for use in Stan.
    output_table_filename = output_path / 'table.csv'
    df.write_csv(output_table_filename)
    print(f'wrote "{output_table_filename}"')


if __name__ == '__main__':

    # TODO
    #  - Perform a fast check on all metadata and data to ensure everything is correct before running the slow
    #    wavelet transform - e.g. ensure all data files are present, ensure channels are within range, show
    #    warnings for channel and time overlaps, ensure times are within range, etc.

    main()
