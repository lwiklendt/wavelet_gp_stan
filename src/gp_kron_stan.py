from abc import ABC, abstractmethod
from collections import OrderedDict
import functools
import inspect
from pathlib import Path
import re
from typing import List
import sys

from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import scipy.linalg
import pandas as pd
import formulaic

import stan_utils


class GPModel(ABC):

    def __init__(self, df, fe_mu_formula=None, re_mu_formulas=None, fe_noise_formula=None, re_noise_formulas=None,
                 priors=None):

        self.df = df
        self.fe_mu_formula = fe_mu_formula or '1'
        self.re_mu_formulas = re_mu_formulas or []
        self.fe_noise_formula = fe_noise_formula or '1'
        self.re_noise_formulas = re_noise_formulas or []

        self.priors = (priors or dict()).copy()  # priors will be modified, so copy

        # init variables that will hold frequencies once set_data is called
        self.freqs = None

        # set default priors
        varnames = ['lambda_noise', 'lambda_gamma', 'lambda_beta',
                    'tau_sigma', 'tau_gamma', 'tau_beta', 'sigma_noise']
        for varname in varnames:
            self.priors[varname] = self.priors.get(varname, 'gamma(2, 1)')

        # create design matrices
        self.x, self.zs, self.dmat_mu, self.dmats_mu_re = make_design_matrices(df, fe_mu_formula, re_mu_formulas)
        self.w, self.us, self.dmat_noise, self.dmats_noise_re = make_design_matrices(df, fe_noise_formula,
                                                                                     re_noise_formulas)

        # setup simplified coefficient and level names to facilitate plotting (this might change when fit is added)
        self.fe_mu_coeffs = list(self.dmat_mu.columns)
        self.fe_noise_coeffs = list(self.dmat_noise.columns)
        self.re_mu_coeffs = dict()
        self.re_mu_levels = dict()
        self.re_noise_coeffs = dict()
        self.re_noise_levels = dict()
        for re_formula in self.re_mu_formulas:
            re_dmat, factor_dmat = self.dmats_mu_re[re_formula]
            self.re_mu_coeffs[re_formula] = list(re_dmat.columns)
            self.re_mu_levels[re_formula] = list(factor_dmat.columns)
        for re_formula in self.re_noise_formulas:
            re_dmat, factor_dmat = self.dmats_noise_re[re_formula]
            self.re_noise_coeffs[re_formula] = list(re_dmat.columns)
            self.re_noise_levels[re_formula] = list(factor_dmat.columns)

        # create stan input data
        self.stan_input_data = OrderedDict()
        self.stan_input_data['N'] = self.x.shape[0]
        self.stan_input_data['P'] = self.x.shape[1]
        self.stan_input_data['Q'] = self.w.shape[1]
        self.stan_input_data['X'] = self.x
        self.stan_input_data['W'] = self.w

        # get subclass-specific variables
        template = self.get_template()
        self.params = self.get_params_fe().copy()
        params_re = self.get_params_re()

        # extract template blueprints
        blueprints = dict(onecol=dict(), multicol=dict())
        locs = dict(data            ='/*** {edge} data {colspec} ***/',
                    params          ='/*** {edge} parameters {colspec} ***/',
                    xfrm_param_decs ='/*** {edge} transformed parameter declarations {colspec} ***/',
                    xfrm_param_defs ='/*** {edge} transformed parameter definitions {colspec} ***/',
                    model           ='/*** {edge} model {colspec} ***/',
                    genqaunt_decs   ='/*** {edge} generated quantities declarations {colspec} ***/',
                    genqaunt_defs   ='/*** {edge} generated quantities definitions {colspec} ***/')
        blueprint_line_idxs = {k: [None, None] for k in locs.keys()}
        for colspec in blueprints.keys():
            for loc, loc_str in locs.items():

                # extract blueprint
                str_start = loc_str.format(edge='start', colspec=colspec)
                str_end   = loc_str.format(edge='end',   colspec=colspec)
                line_start = next(line_idx for line_idx, line in enumerate(template) if str_start in line)
                line_end   = next(line_idx for line_idx, line in enumerate(template) if str_end   in line)
                blueprint = '\n'.join(line for line in template[line_start+1:line_end-1] if len(line.strip()) > 0)
                blueprints[colspec][loc] = blueprint

                # keep line indexes to know which parts of the template to replace with generated code
                if colspec == 'onecol':
                    blueprint_line_idxs[loc][0] = line_start
                else:
                    blueprint_line_idxs[loc][1] = line_end

        # make random-effect code from blueprints to inject back into template
        syringe = {k: [] for k, _ in locs.items()}
        prior_varnames = ['lambda', 'sigma']
        for zs, dpar, pre_vname in [(self.zs, 'eta', 'mu'), (self.us, 'log_omega', 'noise')]:
            for ti, (z, l, term) in enumerate(zs, 1):

                v = f'{pre_vname}_b{ti}'  # variable name
                _, ncol = z.shape         # num columns in the random-effect design matrix
                nlev = l.max()

                # add default priors for specific terms
                for varname in prior_varnames:
                    prior_name_generic = f'{varname}_{pre_vname}_'
                    prior_name_specific = f'{prior_name_generic}b{ti}'
                    if prior_name_specific not in self.priors:
                        self.priors[prior_name_specific] = self.priors.get(prior_name_generic, 'gamma(2, 1)')

                self.params.append(v)
                self.params += [param.format(v) for param in params_re]
                self.stan_input_data[f'l_{v}'] = l
                if ncol == 1:
                    self.stan_input_data[f'Z_{v}'] = z[:, 0].T
                else:
                    self.stan_input_data[f'Z_{v}'] = z
                    self.params += [f'chol_corr_{v}']

                blueprint_kwargs = dict(ncol=ncol, nlev=nlev, v=v, term=term, dpar=dpar)
                for loc, blueprint in blueprints['onecol' if ncol == 1 else 'multicol'].items():
                    syringe[loc].append(blueprint.format(**blueprint_kwargs))

        # inject code
        self.code = template.copy()
        for k in syringe.keys():
            contents = '\n'.join(syringe[k])  # combine list into a single string
            line_start, line_end = blueprint_line_idxs[k]
            self.code[line_start] = contents
            self.code[line_start+1:line_end+1] = [None, ] * (line_end - line_start)
        self.code = '\n'.join([line for line in self.code if line is not None])

        # inject priors
        for varname, prior in self.priors.items():

            # check if varname represents a generic prior specification, in which case ignore it
            if varname.endswith('_'):
                continue

            locator = f'prior_{varname}'
            if locator not in self.code:
                raise RuntimeError(f'unknown prior key: {varname}')
            self.code = self.code.replace(locator + ';', prior + ';')

    def __str__(self):
        s = ''

        # plot general sizes
        if 'y' in self.stan_input_data:
            y = self.stan_input_data['y']
            s += f'num units = {y.shape[0]}\n'
            s += f'response shape = {np.array(y.shape[1:]).squeeze()}\n'

        # plot design matrix schemas
        s += f'fe_mu_formula:\n  {self.fe_mu_formula}\n    = '
        s += ' + '.join(simplify_column_names(self.fe_mu_coeffs)) + '\n'
        s += f'fe_noise_formula:\n  {self.fe_noise_formula}\n    = '
        s += ' + '.join(simplify_column_names(self.fe_noise_coeffs)) + '\n'
        if len(self.re_mu_formulas) > 0:
            s += 're_mu_formulas:\n  '
            for formula in self.re_mu_formulas:
                coeffs = simplify_column_names(self.re_mu_coeffs[formula])
                levels = simplify_column_names(self.re_mu_levels[formula])
                s += f'{formula}\n    = ' + ' + '.join(coeffs)
                s += '\n    | ' + ' + '.join(levels)
        if len(self.re_noise_formulas) > 0:
            s += 're_noise_formulas:\n  '
            for formula in self.re_noise_formulas:
                coeffs = simplify_column_names(self.re_noise_coeffs[formula])
                levels = simplify_column_names(self.re_noise_levels[formula])
                s += f'{formula}\n    = ' + ' + '.join(coeffs)
                s += '\n    | ' + ' + '.join(levels)

        return s

    @classmethod
    def get_template(cls) -> List[str]:
        pass

    @classmethod
    def get_params_fe(cls) -> List[str]:
        return ['beta', 'gamma', 'offset_eta', 'sigma_noise', 'lambda_beta', 'lambda_gamma',
                'tau_beta', 'tau_gamma', 'tau_sigma', 'noise']

    @classmethod
    def get_params_re(cls) -> List[str]:
        pass

    @abstractmethod
    def set_data(self, y, **kwargs):
        pass

    @abstractmethod
    def _plot_axes(self, ax, samples, freq_cpm, icpt=False, vmin=None, vmax=None, alpha=0.05, value_label=None):
        pass

    def _plot_peaks_axes(self, ax, samples, freq_cpm, icpt, vmin, vmax, alpha=0.05, value_label=None):
        pass

    @abstractmethod
    def interp_samples_subdivide(self, samples, subdivision):
        pass

    def sample(self, outpath, **kwargs):
        """Calls stan_utils.sample with self.code and self.stan_input_data"""
        return stan_utils.sample(src_stan_code=self.code,
                                 data=self.stan_input_data,
                                 output_dirname=outpath,
                                 sample_kwargs=kwargs)

    def fit(self, samples, component, data):
        if component == 'eta':
            return self._fit(samples['beta'], self.dmat_mu, data) + samples['offset_eta'][:, np.newaxis]
        elif component == 'omega':
            return self._fit(samples['gamma'], self.dmat_noise, data)
        else:
            raise RuntimeError(f'unknown component "{component}", components should be "eta" or "omega"')

    # posterior predictive
    def pred(self, samples, data: dict):

        # eta fixed effects
        eta = self._fit(samples['beta'], self.dmat_mu, data) + samples['offset_eta'][:, np.newaxis]

        # eta random effects
        for ti, (re_term, (dmat_re, _)) in enumerate(self.dmats_mu_re.items(), 1):
            eta += self._fit(samples[f'new_mu_b{ti}'], dmat_re, data)

        # omega fixed effects
        log_omega = self._fit(samples['gamma'], self.dmat_noise, data)

        # omega random effects
        for ti, (re_term, (dmat_re, _)) in enumerate(self.dmats_noise_re.items(), 1):
            log_omega += self._fit(samples[f'new_noise_b{ti}'], dmat_re, data)

        # noise
        eta += samples['noise'] * np.exp(log_omega)

        return eta

    @staticmethod
    def _fit(samples, dmat: formulaic.ModelMatrix, data: dict):
        df_data = pd.DataFrame({k: [v] for k, v in data.items()})
        dmat_predict = np.array(dmat.model_spec.get_model_matrix(df_data))
        return np.einsum('ska, k -> sa', samples, dmat_predict[0])

    def plot(self, freq_cpm, decs, eqns, titles=None, icpt_tx=None, diff_tx=None, alpha=0.05, simplify_coeffs=True,
             icpt_value_label=None, diff_value_label='Ratio', offset_eta=None, force_diff=False,
             col_width=None, row_height=None, peaks=False, suptitle=None):
        """
        Plot results.
        @param freq_cpm: frequency in log2 cpm.
        @param decs: either coefficients such as samples['beta'] or samples['gamma'], or a dict str->samples containing
                     named variables that are accessed by the equations in eqns.
        @param eqns: list of lists specifying a row-major matrix of equations as str, where variables' values are
                     looked up in the decs parameter, where each equation is plotted on a single axes.
        @param titles: list of lists of the same size as eqns, specifying the titles of the axes specified in eqns.
        @param icpt_tx: intercept transform, for example, if log responses are used can be np.exp to transform back.
        @param diff_tx: comparison transform. Note: by default the identity is used where a non-intercept equation
                        is considered a difference on the log-scale, and thus a ratio is plotted.
        @param alpha: value for plotting "significant" contours or band lines.
        @param simplify_coeffs: whether change the name of coefficients to be more human readable.
        @param icpt_value_label: axis label referring to intercept-based values.
        @param diff_value_label: axis label referring to comparison-based values, defaults to "Ratio".
        @param offset_eta: only supply if decs doesn't contain samples from fit, which already include offset_eta.
        @param force_diff: whether to force considering all values as comparisons.
        @param col_width: width in inches per column, or None for the default of 3.5 for 1d and 4.2 for 2d plots.
        @param row_height: height in inches per row, or None for the default of 3.35.
        @param peaks: True to plot peaks-only version, False to print usual full-frequency version.
        @param suptitle: Overall title for the figure.
        @return: matplotlib figure
        """

        # default transforms are identities
        icpt_tx = icpt_tx or (lambda x: x)
        diff_tx = diff_tx or (lambda x: x)

        # no decs dict specified, so decs must be ndarray, where decs.shape[1] == len(eqns)
        is_marginal = type(decs) is not dict
        if is_marginal:
            assert decs.shape[1] == len(eqns)
            if simplify_coeffs:
                eqns = simplify_column_names(eqns)
            keys = eqns
            values = [decs[:, i, ...] for i in range(len(keys))]
            if titles is None:
                titles = list(keys)

            # generate a new set of keys for dec_list such that they can be used in eval, keeping 'Intercept'
            i = 0
            new_keys = []
            for key in keys:
                if key == 'Intercept':
                    new_keys.append(key)
                else:
                    new_keys.append(f'a{i}')
                    i += 1

            # recreate decs with new keys
            decs = dict(zip(new_keys, values))
            eqns = new_keys

            titles = to_grid_list(titles)
            eqns = to_grid_list(eqns)

        # test for 1d or 2d data
        # Note: 2d freq-phase is flattenend into shape (nsamples, nfreq * nphase),
        #       but freq_freq is (nsamples, nfreq, nfreq)
        samples0 = next(iter(decs.values()))
        is_1d = samples0.shape[-1] == len(freq_cpm) and len(samples0.shape) == 2
        if is_1d:
            extrema_func = lambda x: x
        else:
            extrema_func = lambda x: np.median(x, axis=0)

        # Peaks version, convert samples from f to [max_f, log2_freq_cpm[argmax_f]]
        if peaks:
            if not is_1d:
                raise NotImplementedError('peaks currently only implemented for 1d')
            for d in decs.keys():
                idxs = np.argmax(decs[d], axis=1)
                max_f = decs[d][np.arange(len(idxs)), idxs]
                log2_freq_cpm_argmax_f = np.log2(freq_cpm[idxs])
                decs[d] = np.c_[max_f, log2_freq_cpm_argmax_f]

        nrows = len(eqns)
        ncols = max(map(len, eqns))

        col_width = col_width or (3.5 if is_1d else 4.2)
        row_height = row_height or 3.35
        fig = plt.figure(figsize=(ncols * col_width, nrows * row_height))
        gs = gridspec.GridSpec(nrows, ncols)

        # first pass: parse equations, calculate extrema and generate axes
        cells = []
        if peaks:
            icpt_min, icpt_max = np.zeros(2) + np.inf, np.zeros(2) - np.inf
            diff_min, diff_max = np.zeros(2) + np.inf, np.zeros(2) - np.inf
        else:
            icpt_min, icpt_max = np.inf, -np.inf
            diff_min, diff_max = np.inf, -np.inf
        for ri, row_eqn in enumerate(eqns):
            if row_eqn is None or len(row_eqn) == 0:
                continue
            for ci, cell_eqn in enumerate(row_eqn):
                if cell_eqn is None or cell_eqn.strip() == '':
                    continue

                if force_diff:
                    is_icpt = False
                else:
                    # identify intercept vs diff
                    if is_marginal:
                        is_icpt = cell_eqn == 'Intercept'
                    else:
                        is_icpt = ('-' not in cell_eqn)

                # Evaluate the equation under the variables given in decs.
                # Copy is required since a cell_eqn containing just a single decs key will simply return the decs
                #   value, which we may modify later.
                samples = eval(cell_eqn, {}, decs).copy()

                # calculate extrema
                if is_icpt:
                    if peaks:
                        if offset_eta is not None:
                            samples[:, 0] = samples[:, 0] + offset_eta
                        samples[:, 0] = icpt_tx(samples[:, 0])
                        icpt_min = np.minimum(icpt_min, np.min(samples, axis=0))
                        icpt_max = np.maximum(icpt_max, np.max(samples, axis=0))
                    else:
                        if offset_eta is not None:
                            samples = samples + offset_eta[:, np.newaxis]
                        samples = icpt_tx(samples)
                        samples_cond_reduced = extrema_func(samples)
                        icpt_min = min(icpt_min, np.min(samples_cond_reduced))
                        icpt_max = max(icpt_max, np.max(samples_cond_reduced))
                else:
                    if peaks:
                        samples = diff_tx(samples)
                        diff_min = np.minimum(diff_min, np.min(samples, axis=0))
                        diff_max = np.maximum(diff_max, np.max(samples, axis=0))
                    else:
                        samples = diff_tx(samples)
                        samples_cond_reduced = extrema_func(samples)
                        diff_min = min(diff_min, np.min(samples_cond_reduced))
                        diff_max = max(diff_max, np.max(samples_cond_reduced))

                # create axes, and add title
                ax = fig.add_subplot(gs[ri, ci])
                ax.set_title(cell_eqn if titles is None else titles[ri][ci])

                cells.append((ax, (cell_eqn, decs), is_icpt))

        if peaks:
            diff_extrema = np.max(np.abs(np.c_[diff_min, diff_max]), axis=1)
        else:
            diff_extrema = max(abs(diff_min), abs(diff_max))

        # second pass: plot
        for ax, (cell_eqn, decs), is_icpt in cells:

            if is_icpt:
                vmin, vmax = icpt_min, icpt_max
                value_label = icpt_value_label
            else:
                vmin, vmax = -diff_extrema, diff_extrema
                value_label = diff_value_label

            # Evaluate the equation under the variables given in decs.
            # Copy is required since a cell_eqn containing just a single decs key will simply return the decs
            #   value, which we may modify later.
            samples = eval(cell_eqn, {}, decs).copy()

            # recompute sum so that keeping previously calculated sum doesn't use up memory
            if is_icpt:
                if peaks:
                    if offset_eta is not None:
                        samples[:, 0] = samples[:, 0] + offset_eta
                    samples[:, 0] = icpt_tx(samples[:, 0])
                else:
                    if offset_eta is not None:
                        samples = samples + offset_eta[:, np.newaxis]
                    samples = icpt_tx(samples)
            else:
                samples = diff_tx(samples)

            # plot
            if peaks:
                self._plot_peaks_axes(ax, samples, freq_cpm, icpt=is_icpt, vmin=vmin, vmax=vmax, alpha=alpha,
                                      value_label=value_label)
            else:
                self._plot_axes(ax, samples, freq_cpm, icpt=is_icpt, vmin=vmin, vmax=vmax, alpha=alpha,
                                value_label=value_label)

        if suptitle is not None:
            fig.suptitle(suptitle, y=0.99)

        fig.tight_layout()
        return fig

    def plot_coeffs(self, samples, outpath, nchains=1):

        # model.freqs is on the log2 scale
        if self.freqs is None:
            raise RuntimeError('set_data must be called on model before calling plot_coeffs')
        freqs_cpm = 2 ** self.freqs

        # plot other parameters
        params = set(self.get_params_fe()) - set(['beta', 'gamma', 'noise'])
        for param in params:
            x = samples[param]
            if len(x.shape) > 2:
                if param == 'lambda_beta':
                    coeffs = self.fe_mu_coeffs
                elif param == 'lambda_gamma':
                    coeffs = self.fe_noise_coeffs
                else:
                    print(f'not yet implemented plotting parameter {param} of shape {x.shape}')
                    continue

                nr = int(np.ceil(np.sqrt(len(coeffs))))
                nc = int(np.ceil(len(coeffs) / nr))
                fig, ax = plt.subplots(nr, nc, figsize=(3 * nr + 1, 2.5 * nc + 1))
                for i, coeff in enumerate(coeffs):
                    ri = i % nr
                    ci = i // nr
                    ax[ri, ci].scatter(x[:, 0, ri], x[:, 1, ri], c='b', s=2, linewidths=0, alpha=0.01)
                    ax[ri, ci].set_title(coeff)
                fig.tight_layout()
                fig.savefig(outpath / f'{param}.png', dpi=150)
                plt.close(fig)

            else:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                if len(x.shape) == 1:
                    ax.hist(np.reshape(x, (x.shape[0] // nchains, nchains), order='F'),
                            bins=50, histtype='bar', stacked=True)
                else:
                    # TODO reshape and scatter with different color per chain, here and above.
                    ax.scatter(x[:, 0], x[:, 1], c='b', s=2, linewidths=0, alpha=0.01)
                ax.set_title(param)
                fig.tight_layout()
                fig.savefig(outpath / f'{param}.png', dpi=150)
                plt.close(fig)

        # plot betas
        fig = self.plot(freqs_cpm, samples['beta'], self.fe_mu_coeffs, offset_eta=samples['offset_eta'])
        fig.savefig(outpath / 'beta.png', dpi=150)
        plt.close(fig)

        # plot gamma
        fig = self.plot(freqs_cpm, samples['gamma'], self.fe_noise_coeffs)
        fig.savefig(outpath / 'gamma.png', dpi=150)
        plt.close(fig)

        # plot mu random effects
        if len(self.re_mu_formulas) > 0:
            print(f'Plotting random-effects mu')
        for ri, re_formula in enumerate(self.re_mu_formulas):
            b = samples[f'mu_b{ri + 1}']
            label = re_formula.split('|')[-1].strip()

            coeffs = simplify_column_names(self.re_mu_coeffs[re_formula])
            levels = simplify_column_names(self.re_mu_levels[re_formula])

            print(f'  {re_formula}: {coeffs}')

            for ci, coeff in enumerate(coeffs):
                if len(coeffs) > 1:
                    b_coeff = b[:, :, ci, :]
                else:
                    b_coeff = b
                fig = self.plot(freqs_cpm, b_coeff, levels)
                fig.savefig(outpath / f'mu_{label}_{coeff}.png', dpi=150)
                plt.close(fig)

        # plot noise random effects
        if len(self.re_noise_formulas) > 0:
            print(f'Plot random-effects noise:')
        for ri, re_formula in enumerate(self.re_noise_formulas):
            b = samples[f'noise_b{ri + 1}']
            label = re_formula.split('|')[-1].strip()

            coeffs = simplify_column_names(self.re_noise_coeffs[re_formula])
            levels = simplify_column_names(self.re_noise_levels[re_formula])

            print(f'  {re_formula}: {coeffs}')

            for ci, coeff in enumerate(coeffs):
                if len(coeffs) > 1:
                    b_coeff = b[:, :, ci, :]
                else:
                    b_coeff = b
                fig = self.plot(freqs_cpm, b_coeff, levels)
                fig.savefig(outpath / f'noise_{label}_{coeff}.png', dpi=150)
                plt.close(fig)


class GPFreqModel(GPModel):

    def __init__(self, df, fe_mu_formula=None, re_mu_formulas=None, fe_noise_formula=None, re_noise_formulas=None,
                 priors=None):
        super().__init__(df, fe_mu_formula, re_mu_formulas, fe_noise_formula, re_noise_formulas, priors)

    @classmethod
    def get_template(cls):
        this_module = sys.modules[cls.__module__]
        codepath = Path(inspect.getfile(this_module)).parent / 'gp1d_template.stan'
        with codepath.open('r') as f:
            return f.read().splitlines()

    @classmethod
    def get_params_fe(cls):
        return super().get_params_fe() + ['lambda_noise']

    @classmethod
    def get_params_re(cls):
        return ['sigma_{}', 'lambda_{}', 'new_{}']

    def set_data(self, y, **kwargs):
        freqs = kwargs['freqs']
        self.stan_input_data['y'] = y
        self.stan_input_data['F'] = len(freqs)
        self.stan_input_data['f'] = freqs

        self.freqs = freqs

    @classmethod
    def _interp(cls, x, x_star, y, per_sample_kernel_func_x, nugget_size=1e-6):

        nsamples, n = y.shape

        nugget_x = nugget_size * np.identity(len(x))

        y_star = np.empty((nsamples, len(x_star)))
        for i in range(nsamples):

            kern_x      = per_sample_kernel_func_x(i, x     [:, None], x[None, :])
            kern_x_star = per_sample_kernel_func_x(i, x_star[:, None], x[None, :])

            # note: no need to scale by variance here since it cancels out through the solve followed by multiply
            # i.e: chol_solve(K alpha1 == y/a)  and then   y_star = a (L* alpha1)   is the same as
            #      chol_solve(K alpha2 == y  )  and then   y_star =    L* alpha2    with alpha2 = a * alpha1

            kern_chol = np.linalg.cholesky(kern_x + nugget_x)
            alpha = scipy.linalg.cho_solve((kern_chol, True), y[i])

            y_star[i, ...] = (kern_x_star @ alpha).flatten()

        return y_star

    @classmethod
    def interp_samples(cls, samples, x, x_star):

        def sqr_exp_kernel(s, l, x1, x2):
            return s**2 * np.exp(-0.5 * ((x1 - x2) / l)**2)

        lengthscales_x = dict(beta=samples['lambda_beta'], gamma=samples['lambda_gamma'])
        nsamples = len(lengthscales_x['beta'])
        nx_star = len(x_star)

        samples_star = dict()

        def kern_func_x(prm, k, i, x1, x2):
            return sqr_exp_kernel(1, lengthscales_x[prm][i, k], x1, x2)

        # interpolate fixed-effects
        for param in ['beta', 'gamma']:

            v = samples[param]
            _, ncols, _ = v.shape

            # interpolate param
            v_star = np.empty((nsamples, ncols, nx_star))
            for j in range(ncols):
                kern_x = functools.partial(kern_func_x, param, j)
                v_star[:, j, :] = cls._interp(x, x_star, v[:, j, ...], kern_x)

            samples_star[param] = v_star

        # copy over non-x-dependent parameters
        for key in samples.keys():
            if key not in samples_star.keys():
                samples_star[key] = samples[key].copy()

        return samples_star

    def interp_samples_subdivide(self, samples, subdivision: int):
        log2_freqs_cpm = self.freqs
        nfreq = len(log2_freqs_cpm)
        log2_freqs_cpm_subdiv = np.linspace(log2_freqs_cpm[0], log2_freqs_cpm[-1], (nfreq - 1) * subdivision + 1)
        samples_subdiv = self.interp_samples(samples, log2_freqs_cpm, log2_freqs_cpm_subdiv)
        return samples_subdiv, log2_freqs_cpm_subdiv

    def _plot_peaks_axes(self, ax, samples, freq_cpm, icpt, vmin, vmax, alpha=0.05, value_label=None):
        log2_freqs_cpm = np.log2(freq_cpm)

        # create ticks, labels, and grid for freq and phase
        freq_order1 = int(round(log2_freqs_cpm[0]))
        freq_order2 = int(round(log2_freqs_cpm[-1]))
        freq_ticks = np.arange(freq_order1, freq_order2 + 1, dtype=np.int32)
        freq_labels = ['{}'.format(2**fe) if fe >= 0 else '1/{}'.format(2**(-fe)) for fe in freq_ticks]

        ax.get_figure().sca(ax)

        facecolor_value = mcolors.rgb_to_hsv(ax.get_facecolor()[:3])[-1]
        if facecolor_value > 0.5:
            linecolor = 'k'
        else:
            linecolor = 'w'

        #       power          frequency
        ax.plot(samples[:, 0], samples[:, 1],
                lw=0, marker='.', mec='none', mfc=linecolor, alpha=0.03, zorder=20)

        if not icpt:
            ax.axvline(0, c=(1, 0, 0), zorder=50)
            ax.axhline(0, c=(1, 0, 0), zorder=50)

        ax.set_xlabel(value_label)
        if icpt:
            ax.set_ylabel('Frequency (cpm)')
        else:
            ax.set_ylabel(value_label)

        if icpt:
            # Frequency always in log-space on icpt.
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels)
        else:
            ticks, ticklabels = log_ticks_for_ratios(vmax[0])
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ticks, ticklabels = log_ticks_for_ratios(vmax[1])
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)

        ax.set_xlim(vmin[0], vmax[0])
        # ax.set_ylim(vmin[1], vmax[1])

        # Show percentages in each quadrant.
        if not icpt:
            # Note: some equality with 0 can happen due to discretised frequencies.
            tl = np.mean((samples[:, 0] < 0) & (samples[:, 1] > 0))
            tr = np.mean((samples[:, 0] > 0) & (samples[:, 1] > 0))
            bl = np.mean((samples[:, 0] < 0) & (samples[:, 1] < 0))
            br = np.mean((samples[:, 0] > 0) & (samples[:, 1] < 0))
            m = 0.02
            ax.annotate(f'{tl * 100:.1f}%', (0+m, 1-m), xycoords='axes fraction', ha='left', va='top')
            ax.annotate(f'{tr * 100:.1f}%', (1-m, 1-m), xycoords='axes fraction', ha='right', va='top')
            ax.annotate(f'{bl * 100:.1f}%', (0+m, 0+m), xycoords='axes fraction', ha='left', va='bottom')
            ax.annotate(f'{br * 100:.1f}%', (1-m, 0+m), xycoords='axes fraction', ha='right', va='bottom')

    def _plot_axes(self, ax, samples, freq_cpm, icpt=False, vmin=None, vmax=None, alpha=0.05, value_label=None):

        log2_freqs_cpm = np.log2(freq_cpm)

        # create ticks, labels, and grid for freq and phase
        freq_order1 = int(round(log2_freqs_cpm[0]))
        freq_order2 = int(round(log2_freqs_cpm[-1]))
        freq_ticks = np.arange(freq_order1, freq_order2 + 1, dtype=np.int32)
        freq_labels = ['{}'.format(2**fe) if fe >= 0 else '1/{}'.format(2**(-fe)) for fe in freq_ticks]

        ax.get_figure().sca(ax)

        lower = 100 * (0.5 * alpha)
        upper = 100 * (1 - 0.5 * alpha)

        # if we have 3 dimensions then it is a freq-freq comparison
        if len(samples.shape) == 3:

            cdict = dict(
                blue =[(0, 0, 0.5), (0.25, 1  , 1  ), (0.5, 1, 1), (0.75, 0  , 0  ), (1, 0  , 0)],
                green=[(0, 0, 0  ), (0.25, 0.4, 0.4), (0.5, 1, 1), (0.75, 0.4, 0.4), (1, 0  , 0)],
                red  =[(0, 0, 0  ), (0.25, 0  , 0  ), (0.5, 1, 1), (0.75, 1  , 1  ), (1, 0.5, 0)],
            )
            cmap_diff = mcolors.LinearSegmentedColormap('RedBlue', cdict, N=501)

            xegrid, yegrid = edge_meshgrid(log2_freqs_cpm, log2_freqs_cpm)  # for pcolormesh
            xcgrid, ycgrid = np.meshgrid(log2_freqs_cpm, log2_freqs_cpm)    # for contour

            level = 1 - 0.5 * alpha
            samples_pos = np.mean(samples > 0, axis=0)
            samples_neg = np.mean(samples < 0, axis=0)
            samples = np.median(samples, axis=0)
            samples[(samples_pos < level) & (samples_neg < level)] = 0

            mappable = ax.pcolormesh(xegrid, yegrid, samples.T, vmin=vmin, vmax=vmax, cmap=cmap_diff)
            cbar = ax.get_figure().colorbar(mappable, use_gridspec=True, label=value_label)

            kwargs = dict(colors='k', linewidths=1, levels=[level])
            if np.min(samples_pos) < level < np.max(samples_pos):
                ax.contour(xcgrid, ycgrid, samples_pos.T, linestyles='-', **kwargs)
            if np.min(samples_neg) < level < np.max(samples_neg):
                ax.contour(xcgrid, ycgrid, samples_neg.T, linestyles='-', **kwargs)

            cbar_ticks, cbar_ticklabels = log_ticks_for_ratios(vmax)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)

            ax.set_xlabel('Frequency (cpm) [X]')
            ax.set_ylabel('Frequency (cpm) [Y]')
            ax.set_xticks(freq_ticks)
            ax.set_xticklabels(freq_labels)
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels)

        # otherwise just a normal amplitude over frequency plot
        else:

            mu_ci = np.array([np.percentile(samples, q, axis=0) for q in (lower, upper)])

            # Test for 0-crossing.
            if not icpt:
                ci_cross = np.diff(mu_ci > 0, axis=1)
                if np.any(ci_cross):
                    title = ax.get_title()
                    print(f'    "{title}" ci crosses at {2**log2_freqs_cpm[np.where(ci_cross)[1]]} cpm')
                # med_cross = np.diff(np.percentile(samples, 50, axis=0) > 0)
                # if np.any(med_cross):
                #     title = ax.get_title()
                #     print(f'    "{title}" median crosses at {2**log2_freqs_cpm[np.where(med_cross)[0]]} cpm')

            facecolor_value = mcolors.rgb_to_hsv(ax.get_facecolor()[:3])[-1]
            if facecolor_value > 0.5:
                linecolor = 'k'
            else:
                linecolor = 'w'
            ax.plot(mu_ci.T, log2_freqs_cpm, c=linecolor, ls=':', zorder=40)
            ax.plot(samples.T, log2_freqs_cpm, c=linecolor, lw=0.5, alpha=0.01, zorder=20)

            if not icpt:
                ax.axvline(0, c=(1, 0, 0), zorder=50)

            ax.set_xlabel(value_label)
            ax.set_ylabel('Frequency (cpm)')

            # ratio ticks
            if not icpt:
                ticks, ticklabels = log_ticks_for_ratios(vmax)
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticklabels)

            ax.set_xlim(vmin, vmax)

            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels)


class GPFreqPhaseModel(GPModel):

    def __init__(self, df, fe_mu_formula=None, re_mu_formulas=None, fe_noise_formula=None, re_noise_formulas=None,
                 priors=None, sep=None):
        super().__init__(df, fe_mu_formula, re_mu_formulas, fe_noise_formula, re_noise_formulas, priors)
        self.sep = sep

        # init variable that will hold phases once set_data is called
        self.phases = None

    @classmethod
    def get_template(cls):
        this_module = sys.modules[cls.__module__]
        codepath = Path(inspect.getfile(this_module)).parent / 'gp2d_template.stan'
        with codepath.open('r') as f:
            return f.read().splitlines()

    @classmethod
    def get_params_fe(cls):
        return super().get_params_fe() + ['lambda_noise', 'lambda_rho_beta', 'lambda_rho_gamma', 'lambda_rho_noise']

    @classmethod
    def get_params_re(cls):
        return ['sigma_{}', 'lambda_{}', 'lambda_rho_{}', 'new_{}']

    def set_data(self, y, **kwargs):
        freqs = kwargs['freqs']
        phases = kwargs['phases']

        self.freqs = freqs
        self.phases = phases

        # convert from N,F,H major->minor order to N,F*H reshape used in the Stan code (with F*H being F-minor)
        self.stan_input_data['y'] = y.transpose((2, 1, 0)).reshape(-1, y.shape[0]).T

        self.stan_input_data['F'] = len(freqs)
        self.stan_input_data['H'] = len(phases)
        self.stan_input_data['f'] = freqs
        self.stan_input_data['h'] = phases

    def _plot_axes(self, ax, samples, freq_cpm, icpt=False, vmin=None, vmax=None, alpha=0.05, value_label=None):
        log2_freqs_cpm = np.log2(freq_cpm)

        samples = np.reshape(samples, (-1, samples.shape[-1] // len(freq_cpm), len(freq_cpm)))

        nphase = samples.shape[1]
        phase_edges = np.linspace(-np.pi, np.pi, nphase + 1)
        phases = edges_to_centers(phase_edges)

        # create 1x3 periodic grid so that contours are drawn correctly
        phases_1x3 = periodic_coord_wings(phases)

        # pad freqs so that contours at top and bottom are treated properly
        delta_f = log2_freqs_cpm[1] - log2_freqs_cpm[0]
        log2_freqs_cpm = np.r_[log2_freqs_cpm[0] - delta_f, log2_freqs_cpm, log2_freqs_cpm[-1] + delta_f]
        samples = np.pad(samples, [(0, 0), (0, 0), (1, 1)], mode='edge')

        def tile_1x3(x):
            return np.r_[x, x, x]

        hegrid, fegrid = edge_meshgrid(phases_1x3, log2_freqs_cpm)
        hcgrid, fcgrid = np.meshgrid(phases_1x3, log2_freqs_cpm)

        # create ticks, labels, and grid for freq and phase
        freq_order1 = int(log2_freqs_cpm[0])
        freq_order2 = int(log2_freqs_cpm[-1])
        freq_ticks = np.arange(freq_order1, freq_order2 + 1, dtype=np.int32)
        freq_labels = ['{}'.format(2**fe) if fe >= 0 else '1/{}'.format(2**(-fe)) for fe in freq_ticks]
        phase_ticks = np.pi * np.array([-1, -0.5, 0, 0.5, 1])
        phase_labels = ['-π', '-π/2', '0', 'π/2', 'π']

        # cmap for difference-based samples
        # cdict = dict(
        #     blue =[(0, 0, 0.5), (0.25, 1  , 1  ), (0.5, 1, 1), (0.75, 0  , 0  ), (1, 0  , 0)],
        #     green=[(0, 0, 0  ), (0.25, 0.4, 0.4), (0.5, 1, 1), (0.75, 0.4, 0.4), (1, 0  , 0)],
        #     red  =[(0, 0, 0  ), (0.25, 0  , 0  ), (0.5, 1, 1), (0.75, 1  , 1  ), (1, 0.5, 0)],
        # )
        cdict = dict(
            blue =[(0, 0, 1.0), (0.3, 1  , 1  ), (0.5, 0, 0), (0.7, 0.2, 0.2), (1, 0.6, 0)],
            green=[(0, 0, 0.8), (0.3, 0.3, 0.3), (0.5, 0, 0), (0.7, 0.3, 0.3), (1, 0.8, 0)],
            red  =[(0, 0, 0.6), (0.3, 0.2, 0.2), (0.5, 0, 0), (0.7, 1  , 1  ), (1, 1.0, 0)],
        )
        cmap_diff = mcolors.LinearSegmentedColormap('RedBlue', cdict, N=501)

        ax.get_figure().sca(ax)

        mappable = ax.pcolormesh(hegrid, fegrid, tile_1x3(np.median(samples, axis=0)).T,
                                 vmin=vmin, vmax=vmax, cmap='viridis' if icpt else cmap_diff)
        cbar = ax.get_figure().colorbar(mappable, use_gridspec=True, label=value_label or '')

        # colorbar ticks
        if not icpt:
            cbar_ticks, cbar_ticklabels = log_ticks_for_ratios(vmax)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)

        # plot isovelocity lines
        if self.sep is not None:
            velocities = [1, 3, 10, 30, 100]  # cm/min when sep is in cm
            for vel in velocities:
                freq_cpm_edges = centers_to_edges(freq_cpm, log=True)
                vel_phases = 2 * np.pi * self.sep * freq_cpm_edges / vel
                vel_freqs = np.log2(freq_cpm_edges)
                ax.plot(vel_phases, vel_freqs, c='w', lw=1, ls=':', alpha=0.5)
                ax.plot(-vel_phases, vel_freqs, c='w', lw=1, ls=':', alpha=0.5)

        # plot contours at alpha
        levels = [0.5 * alpha, 1 - 0.5 * alpha]
        if not icpt:
            samples_pos = np.mean(samples > 0, axis=0)
            # make padded boundary below contours
            samples_pos[:,  0] = 0.5
            samples_pos[:, -1] = 0.5
            ax.contour(hcgrid, fcgrid, tile_1x3(samples_pos).T,
                       linestyles='-', levels=levels, linewidths=2, colors='w')
            cs = ax.contourf(hcgrid, fcgrid, tile_1x3(samples_pos).T, colors='none',
                             hatches=[None, '////'], levels=levels, extend='both')
            for c in cs.collections:
                c.set_edgecolor((1, 1, 1, 0.3))

        # plot contours at alpha for mirror diff
        if icpt:
            # if intercept, then only care about plotting which direction has higher values
            samples_diff_extreme = np.mean((samples - samples[:, ::-1, :]) > 0, axis=0)
        else:
            # if diff, then we care about plotting which direction has significantly larger absolute difference
            samples_diff_extreme = np.mean(((samples - samples[:, ::-1, :]) > 0) & (samples > 0) |
                                           ((samples - samples[:, ::-1, :]) < 0) & (samples < 0), axis=0)
        if np.min(samples_diff_extreme) < levels[1] < np.max(samples_diff_extreme):  # only plot if there exists contour
            ax.contour(hcgrid, fcgrid, tile_1x3(samples_diff_extreme).T,
                       linestyles=[(2, (2, 2))], colors='k', linewidths=2, levels=[levels[1]])
            ax.contour(hcgrid, fcgrid, tile_1x3(samples_diff_extreme).T,
                       linestyles=[(0, (2, 2))], colors='w', linewidths=2, levels=[levels[1]])

        ax.axvline(0, c='w', lw=1, alpha=0.5)

        ax.set_yticks(freq_ticks)
        ax.set_yticklabels(freq_labels)
        ax.set_xticks(phase_ticks)
        ax.set_xticklabels(phase_labels)

        ax.set_xlim(phase_edges[0], phase_edges[-1])  # must limit here since we're drawing at 1x3
        ax.set_ylim(fegrid[1, 0], fegrid[-2, -1])  # remove freq padding

        ax.set_ylabel('Frequency (cpm)')
        ax.set_xlabel('Phase (rad)')

    @classmethod
    def _interp(cls, x, x_star, w, w_star, y, per_sample_kernel_func_x, per_sample_kernel_func_w, nugget_size=1e-6):

        nsamples, n = y.shape

        nugget_x = nugget_size * np.identity(len(x))
        nugget_w = nugget_size * np.identity(len(w))

        y_star = np.empty((nsamples, len(x_star) * len(w_star)))
        for i in range(nsamples):

            kern_x      = per_sample_kernel_func_x(i, x     [:, None], x[None, :])
            kern_x_star = per_sample_kernel_func_x(i, x_star[:, None], x[None, :])

            kern_w      = per_sample_kernel_func_w(i, w     [:, None], w[None, :])
            kern_w_star = per_sample_kernel_func_w(i, w_star[:, None], w[None, :])

            inv_kern_w = np.linalg.inv(kern_w + nugget_w)
            inv_kern_x = np.linalg.inv(kern_x + nugget_x)
            alpha = kron_mvprod([inv_kern_x, inv_kern_w], y[i]).flatten()

            y_star[i, ...] = kron_mvprod([kern_x_star, kern_w_star], alpha).flatten()

        return y_star

    @classmethod
    def interp_samples(cls, samples, x, w, x_star, w_star):

        def sqr_exp_kernel(s, l, x1, x2):
            return s**2 * np.exp(-0.5 * ((x1 - x2) / l)**2)

        def periodic_sqr_exp_kernel(s, l, w1, w2):
            return s**2 * np.exp(-2 * np.sin(0.5 * np.abs(w1 - w2))**2 / l**2)

        lengthscales_x = dict(beta=samples['lambda_beta'][:, 0], gamma=samples['lambda_gamma'][:, 0])
        lengthscales_w = dict(beta=samples['lambda_beta'][:, 1], gamma=samples['lambda_gamma'][:, 1])
        # sigma_noise = samples['sigma_noise']
        nsamples = len(lengthscales_x['beta'])
        nx_star = len(x_star)
        nw_star = len(w_star)

        samples_star = dict()

        def kern_func_x(prm, k, i, x1, x2):
            return sqr_exp_kernel(1, lengthscales_x[prm][i, k], x1, x2)

        def kern_func_w(prm, k, i, w1, w2):
            return periodic_sqr_exp_kernel(1, lengthscales_w[prm][i, k], w1, w2)

        # interpolate fixed-effects
        for param in ['beta', 'gamma']:

            v = samples[param]
            _, ncols, _ = v.shape

            # interpolate param
            v_star = np.empty((nsamples, ncols, nx_star * nw_star))
            for j in range(ncols):
                kern_x = functools.partial(kern_func_x, param, j)
                kern_w = functools.partial(kern_func_w, param, j)
                v_star[:, j, :] = cls._interp(x, x_star, w, w_star, v[:, j, ...], kern_x, kern_w)

            samples_star[param] = v_star

        # copy over non-x-dependent parameters
        for key in samples.keys():
            if key not in samples_star.keys():
                samples_star[key] = samples[key].copy()

        return samples_star

    def interp_samples_subdivide(self, samples, subdivision):

        # subdivide frequencies
        log2_freqs_cpm = self.freqs
        nfreq = len(log2_freqs_cpm)
        log2_freqs_cpm_subdiv = np.linspace(log2_freqs_cpm[0], log2_freqs_cpm[-1], (nfreq - 1) * subdivision + 1)

        # subdivide phases
        phases = self.phases
        nphase = len(phases)
        phase_edges = centers_to_edges(phases)
        phase_edges_subdiv = np.linspace(phase_edges[0], phase_edges[-1], nphase * subdivision)
        phases_subdiv = edges_to_centers(phase_edges_subdiv)

        samples_subdiv = self.interp_samples(samples, log2_freqs_cpm, phases, log2_freqs_cpm_subdiv, phases_subdiv)
        return samples_subdiv, log2_freqs_cpm_subdiv


def log_ticks_for_ratios(log_xmax):

    xmax = np.exp(log_xmax)
    log2_xmax = np.log2(xmax)

    # logarithmically-spaced ticks (will be linear-spaced on a log-scale axis)
    if log2_xmax > 1:
        log2_ticks = np.arange(0, np.ceil(log2_xmax) + 1)

    # linearly-spaced ticks (will be log-spaced on a linear-scale axis)
    else:
        ticker = mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
        log2_ticks = np.log2(np.array(ticker.tick_values(1, xmax)))

    log2_ticks = np.r_[-log2_ticks[1:][::-1], log2_ticks]
    ticklabels = [f'{2 ** v:g}' if v >= 0 else f'{2 ** -v:g}⁻¹' for v in log2_ticks]
    log_ticks = np.log(2 ** log2_ticks)

    return log_ticks, ticklabels


def make_design_matrices(df, fe_formula=None, re_formulas=None):

    fe_formula = fe_formula or '1'
    re_formulas = re_formulas or []

    # TODO we want structural zeros: https://github.com/matthewwardrop/formulaic/issues/152
    #  This is very difficult due to lack of API documentation and methods to facilitate, so instead work around it
    #  by ignoring zeros in the Stan model.

    dmat_fe = formulaic.model_matrix(fe_formula, df, ensure_full_rank=True, na_action='raise')
    x = np.asarray(dmat_fe)

    dmats_re = OrderedDict()

    # TODO pre-process by breaking up factor expressions
    # "(x | g1 / g2)"  ->  "(x | g1) + (x | g1:g2)"
    # "(x || g)"       ->  "(1 | g)  + (0 + x | g)"

    zs = []
    for re_formula in re_formulas:
        re_expr, factor = (a.strip() for a in re_formula.split('|'))

        # each row's level
        factor_dmat = formulaic.model_matrix(f'0 + {factor}', df)
        factor_x = np.asarray(factor_dmat, dtype=int)
        if not np.all(np.sum(factor_x, axis=1) == 1):
            raise Exception(f'Incorrectly specified factor "{factor}" in formula: "{re_formula}"')

        # design matrix of the random effect expression
        re_dmat = formulaic.model_matrix(f'{re_expr}', df)
        re_x = np.asarray(re_dmat)
        dmats_re[re_formula] = (re_dmat, factor_dmat)

        _, l_idxs = np.nonzero(factor_x)
        zs.append((re_x, l_idxs + 1, re_formula))

    return x, zs, dmat_fe, dmats_re


def simplify_column_names(column_name, simplify_terms=True):

    if type(column_name) is list:
        return [simplify_column_names(n, simplify_terms) for n in column_name]

    if column_name != 'Intercept':
        column_name = column_name.replace('T.', '')
        if simplify_terms and '[' in column_name:
            labels = []
            for term in column_name.split(':'):
                if '[' in term:
                    m = re.search(r'.+\[(.+)\]', term)
                    labels.append(m.group(1))
                else:
                    labels.append(term)
            column_name = '_'.join(labels)
        else:
            column_name.replace(':', '\n')
    return column_name


def to_grid_list(x: List) -> List[List]:
    nrows = int(np.floor(np.sqrt(len(x))))
    ncols = int(np.ceil(len(x) / nrows))
    x = x + [None, ] * (nrows * ncols - len(x))  # expand x with Nones to make it factorise to nrows * ncols
    return [x[i:i+ncols] for i in range(0, len(x), ncols)]


def periodic_coord_wings(x):
    dx = x[1] - x[0]
    return np.r_[x - x[-1] + x[0] - dx, x, x - x[0] + x[-1] + dx]


def kron_mvprod(ms, v):
    u = v.copy()
    for m in ms[::-1]:
        u = m @ np.reshape(u.T, (m.shape[1], -1))
    return u.T


def edges_to_centers(edges, log=False):
    if log:
        edges = np.log2(edges)
    centers = edges[1:] - 0.5 * (edges[1] - edges[0])
    if log:
        centers = 2 ** centers
    return centers


def centers_to_edges(centers, log=False):
    if log:
        centers = np.log2(centers)
    if len(centers) == 1:
        dx = 1
    else:
        dx = centers[1] - centers[0]
    edges = np.r_[centers, centers[-1] + dx] - 0.5 * dx
    if log:
        edges = 2 ** edges
    return edges


def edge_meshgrid(centers_x, centers_y, logx=False, logy=False):
    return np.meshgrid(centers_to_edges(centers_x, logx), centers_to_edges(centers_y, logy))
