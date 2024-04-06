import argparse
import os
from pathlib import Path
import tomllib
import uuid

import numpy as np
import polars as pl
import formulaic

import utils
import wavelet
import gp_kron_stan
import gp_kron_stan_simple


def main():
    pl.Config.set_tbl_rows(1000)
    pl.Config.set_tbl_cols(100)
    pl.Config.set_tbl_width_chars(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename')
    parser.add_argument('-dryrun', action='store_true')
    args = parser.parse_args()

    # Load config toml and change the cwd to its location.
    config_filename = Path(args.config_filename)
    note = f'  Running Stan: {config_filename}  '
    print('\n')
    print("=" * len(note))
    print(note)
    print("=" * len(note))
    config = tomllib.loads(config_filename.read_text(encoding='utf-8'))
    os.chdir(config_filename.parent)

    # Read input and output path, or if no output path then set default output to the name of the config file.
    input_path = Path(config['data']['input_path'])
    if 'outpath' in config['data']:
        output_path = Path(config['data']['output_path'])
    else:
        output_path = Path(config_filename.stem)
    utils.ensure_path(output_path)

    # Read params written in wavelet step.
    from_wavelet_params_filename = input_path / 'params.toml'
    from_wavelet_params = tomllib.loads(from_wavelet_params_filename.read_text(encoding='utf-8'))
    unit_id_column = from_wavelet_params['table']['unit_id_column']
    wavelet_params = from_wavelet_params['wavelet']
    if type(wavelet_params['freq_min_cpm']) is str:
        wavelet_params['freq_min_cpm'] = eval(wavelet_params['freq_min_cpm'])
    if type(wavelet_params['freq_max_cpm']) is str:
        wavelet_params['freq_max_cpm'] = eval(wavelet_params['freq_max_cpm'])
    log2_freqs_cpm = wavelet.make_log2_freqs_cpm(**wavelet_params)
    nfreq = len(log2_freqs_cpm)
    nphase = wavelet_params['nphase']

    # Formulas.
    fe_mu_formula = config['design']['population']['mean']
    fe_noise_formula = config['design']['population']['variance']
    if 'group' in config['design']:
        re_mu_formulas = config['design']['group']['mean']
    else:
        re_mu_formulas = None

    # Read table written in wavelet step.
    df = pl.read_csv(input_path / 'table.csv')

    # Strip whitespace in String columns. This is a recurring problem in researchers' tables.
    for col in df.columns:
        if df[col].dtype == pl.String:
            df = df.with_columns(pl.col(col).map_elements(lambda s: s.strip()))

    # TODO based on the formulas, plot data at least for the 1d case. Or, should we just plot data in the dryrun?
    # Collect all variables that appear in the formulas.
    df_pandas = df.to_pandas()
    variables = formulaic.model_matrix(fe_mu_formula, df_pandas).model_spec.variables_by_source['data']
    variables |= formulaic.model_matrix(fe_noise_formula, df_pandas).model_spec.variables_by_source['data']
    if re_mu_formulas:
        for formula in re_mu_formulas:
            spec = formulaic.model_matrix(formula, df_pandas).model_spec
            variables |= spec[0].variables
            variables |= spec[1].variables
    del df_pandas

    # # Drop null entries for all variables that appear in the formulas.
    # nulls = np.zeros(len(df), dtype=bool)
    # for variable in variables:
    #     v_nulls = df[variable].is_null().to_numpy()
    #     n_nulls = np.sum(v_nulls)
    #     if n_nulls > 0:
    #         print(f'dropping {n_nulls} nulls from {variable}')
    #         nulls |= v_nulls
    # df = df.filter(~nulls)

    # Filter (keep) based on design.
    if 'filter' in config['design']:
        for variable, value in config['design']['filter'].items():
            if type(value) is list:
                df = df.filter(pl.col(variable).is_in(value))
            else:
                df = df.filter(pl.col(variable) == value)

    # Remove (filter-out).
    if 'remove' in config['design']:
        for variable, value in config['design']['remove'].items():
            if type(value) is list:
                df = df.filter(pl.col(variable).is_in(value).not_())
            else:
                df = df.filter(pl.col(variable) != value)

    # Add index column to table.
    idx_uuid = str(uuid.uuid4()).replace('-', '')  # Removing '-' so that eval works.
    idx_column = f'idx_{idx_uuid}'
    df = df.with_columns(pl.Series(name=idx_column, values=list(range(len(df)))))

    # Load wavelet data.
    fs = np.nan + np.zeros((len(df), nfreq))
    fps = np.nan + np.zeros((len(df), nfreq, nphase))
    for row in df.rows(named=True):
        idx = row[idx_column]
        unit_id = row[unit_id_column]
        f, fp = utils.pkl_load((input_path / 'data' / unit_id).with_suffix('.pkl'))
        fs[idx] = f
        fps[idx] = fp

    simple_version = 'lengthscale_freq' in config['design']

    phase_edges = np.linspace(-np.pi, np.pi, nphase + 1)
    phases = utils.edges_to_centers(phase_edges)

    dims = config['design']['dims']
    if type(dims) is int:
        dims = [dims]

    for dim in dims:
        if simple_version:
            print(f'running {dim}d (simple)')
        else:
            print(f'running {dim}d (full)')

        # Prepare data.
        y = {1: fs, 2: fps}[dim]
        y = np.log(y)
        icpt_tx = np.exp  # Store transformation to undo the log above.
        assert np.all(np.isfinite(y))

        # Setup model.
        if simple_version:
            kern_func_freq = lambda x1, x2: gp_kron_stan_simple.kern_func_sqr_exp(
                config['design']['lengthscale_freq'], x1, x2)
            if dim == 1:
                model = gp_kron_stan_simple.GPFreqModel(df.to_pandas(), fe_mu_formula, re_mu_formulas, fe_noise_formula,
                                                        priors=config['priors'], kern_func_freq=kern_func_freq)
                model.set_data(y, freqs=log2_freqs_cpm)
            elif dim == 2:
                kern_func_phase = lambda x1, x2: gp_kron_stan_simple.kern_func_periodic_sqr_exp(
                    config['design']['lengthscale_phase'], x1, x2)
                model = gp_kron_stan_simple.GPFreqPhaseModel(df.to_pandas(),
                                                             fe_mu_formula, re_mu_formulas, fe_noise_formula, sep=1,
                                                             priors=config['priors'], kern_func_freq=kern_func_freq,
                                                             kern_func_phase=kern_func_phase)
                model.set_data(y, freqs=log2_freqs_cpm, phases=phases)
        else:
            if dim == 1:
                model = gp_kron_stan.GPFreqModel(df.to_pandas(), fe_mu_formula, re_mu_formulas, fe_noise_formula,
                                                 priors=config['priors'])
                model.set_data(y, freqs=log2_freqs_cpm)
            elif dim == 2:
                model = gp_kron_stan.GPFreqPhaseModel(df.to_pandas(), fe_mu_formula, re_mu_formulas, fe_noise_formula,
                                                      priors=config['priors'], sep=1)
                model.set_data(y, freqs=log2_freqs_cpm, phases=phases)

        # Write out design matrices.
        model.dmat_mu.__wrapped__.to_csv(output_path / f'design_mu.csv', index=False)
        model.dmat_noise.__wrapped__.to_csv(output_path / f'design_noise.csv', index=False)
        with open(output_path / f'design_columns.txt', 'w') as f:
            print('dmat_mu:', file=f)
            for c in model.dmat_mu.model_spec.column_names:
                print(f'  {c}', file=f)
            print('dmat_noise:', file=f)
            for c in model.dmat_noise.model_spec.column_names:
                print(f'  {c}', file=f)

        # Perform sampling, or retrieve samples if nothing has changed.
        samples_path = output_path / f'samples_{dim}d'
        samples, is_resampled = model.sample(samples_path, show_progress=True, show_console=True, refresh=10,
                                             **config['stan'])

        timer = utils.Timer()

        if is_resampled:
            timer.restart()
            model.plot_coeffs(samples, samples_path)
            print(f'Plot coeffs elapsed: {timer}')

        # Choose intercept label.
        if from_wavelet_params['wavelet']['log2_pres']:
            icpt_label = 'Power (log₂(mmHg)²)'
        else:
            icpt_label = 'Power (mmHg²)'

        alpha = 0.05
        print(f'alpha level: {100 * alpha:.2f}%')

        # Plot.
        default_plots = config['defaults']['plot']
        for plot_config in config['plot']:
            plot_config = utils.merge_dicts(plot_config, default_plots)

            filename = output_path / f'results_{dim}d' / plot_config['filename']
            utils.ensure_path(filename.parent)
            print(f'plotting: {filename}')

            # Interpolate to high-resolution.
            timer.restart()
            subdiv = int(np.ceil(plot_config.get('subdivision_resolution', 100) / nfreq))
            if subdiv > 0:
                print(f'  subdividing {subdiv} times ({nfreq} -> {(nfreq - 1) * subdiv + 1})... ', end='')
                samples_hr, log2_freqs_cpm_hr = model.interp_samples_subdivide(samples, subdiv)
                freqs_cpm_hr = 2**log2_freqs_cpm_hr
                print(f'elapsed: {timer}')
            else:
                samples_hr = samples
                freqs_cpm_hr = 2**log2_freqs_cpm

            # Fit declarations ("label" entries in toml file).
            decs = dict()
            print('  fitting:', end='')
            for dec, defn in plot_config['label'].items():
                if 'constants' in plot_config:
                    defn = utils.merge_dicts(defn, plot_config['constants'])
                print(f' {dec}', end='')
                decs[dec] = model.fit(samples_hr, 'eta', data=defn)
            print('')

            eqns = plot_config['equations']
            if 'titles' in plot_config:
                titles = plot_config['titles']
            else:
                titles = [[elt.strip() for elt in row] for row in eqns]
            peaks = plot_config.get('peaks', False)

            # Plot and save.
            print('  rendering')
            fig = model.plot(freqs_cpm_hr, decs, eqns, titles=titles, force_diff=False,  # TODO implement force_diff
                             icpt_tx=icpt_tx, icpt_value_label=icpt_label, alpha=alpha, peaks=peaks,
                             suptitle=config_filename)
            fig.savefig(filename, dpi=plot_config.get('dpi', 150))


if __name__ == '__main__':
    main()
