from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage

import utils


def load_hrm_txt(filename):
    """
    Load a High-Resolution Manometry text file saved by John's catheter recording software.
    :param filename: full-path filename of the text file saved with the "<TIME>\t<MARK>\t<P0>\t<P1>\t...\t<PM>" format
    :return: t, c, p: numpy arrays of n time-samples, with time t:(n,), marks c:(n,), pressures p:(m,n)
    """
    df = pd.read_csv(filename, header=None, sep='\t')
    x = df.values
    times = x[:, 0]
    marks = x[:, 1]
    pres = x[:, 2:].T

    # sometimes there are extra columns of NANs at the end, so remove them
    pres = pres[~np.all(np.isnan(pres), axis=1), :]

    return times, marks, pres


# TODO replace with ema like in wrinklescope
def baseline_gauss(x, sigma, iters):
    orig = x
    for i in range(iters):
        x = scipy.ndimage.gaussian_filter1d(x, sigma)
        if i < iters - 1:
            x = np.minimum(orig, x)
    return x


def clean_pressures(p, sigma_samples, iters, sync_rem):
    """
    Performs baseline and synchronous anomaly removal.
    @param p: (nchan, nsamp) shaped array of pressures
    @param sigma_samples: parameter for lsw.signal.baseline_gauss
    @param iters: parameter for lsw.signal.baseline_gauss
    @param sync_rem: whether to perform synchronous anomaly removal
    @return: cleaned p
    """

    # baseline removal
    def exec_func(chan):
        p[chan, :] -= baseline_gauss(p[chan, :], sigma_samples, iters)
    utils.parexec(exec_func, p.shape[0])

    # synchronous activity removal
    if sync_rem:
        p = np.maximum(0, p - np.maximum(0, np.median(p, axis=0, keepdims=True)))

    return p


class OnDemandHRM:
    """
    Loading and pre-processing (cleaning pressures) is time consuming, so only do it when we need the result
    by calling get_data(). The reason we need this at all is because it's easier to return this object and
    then later check to see if we need the data because we might not have the cached result of processing
    this data.
    """

    def __init__(self, filepath: Path, ensure_dt: float | None, syncrem: bool):
        self.filepath = filepath
        self.ensure_dt = ensure_dt
        self.t = None
        self.x = None
        self.syncrem = syncrem

    def get_data(self):
        if self.x is None:
            errors = []

            # load filename
            t, _, x = load_hrm_txt(self.filepath)

            diff_t = np.diff(t)
            dt = np.median(diff_t)

            # Ensure sample rate is consistent throughout.
            if (np.min(diff_t) / np.max(diff_t) - 1.) > 1e-6:
                errors.append('Inconsistent sampling rate')

            # Ensure expected sampling rate.
            if self.ensure_dt is not None and np.abs(dt / self.ensure_dt - 1.0) > 1e-6:
                errors.append(f'ensure_dt={self.ensure_dt}, actual dt={dt}')

            # Ensure no nans.
            if np.sum(np.isnan(x)) > 0:
                errors.append('There are NaNs in the pressures')

            if len(errors) > 0:
                raise RuntimeError(f'Errors in "{self.filepath}": {errors}')

            # pre-process with baseline and synchronous anomaly removal
            x = clean_pressures(x, sigma_samples=10/dt, iters=10, sync_rem=self.syncrem)

            self.t = t
            self.x = x

        return self.t, self.x


def pandas_split(df, column):
    """
    Splits df by groupby on column, yielding pairs of (value, dataframe) per unique column entry.
    """
    for _, df_sub in df.groupby(column):
        value = df_sub[column].iloc[0]
        yield value, df_sub.drop(columns=[column])


def get_data(df: pd.DataFrame,
             root_data_path: Path,
             syncrem: bool,
             channel_column='channels',
             time_column='seconds',
             ensure_dt: float | None = 0.1):

    for filename, df in pandas_split(df, 'filename'):

        filepath = root_data_path / filename

        on_demand_hrm = OnDemandHRM(filepath, ensure_dt, syncrem)

        # for each section of the recording
        for _, df_row in df.iterrows():

            chan_start, chan_end = df_row[channel_column]
            sec_start, sec_end = df_row[time_column]

            # If we don't need this df_row, the allow us to skip loading and preprocessng at the get_data call site.
            def lazy_loader() -> np.ndarray:
                t, x = on_demand_hrm.get_data()
                samp_start = np.searchsorted(t, sec_start, side='left')
                samp_end = np.searchsorted(t, sec_end, side='right')
                return x[chan_start:chan_end, samp_start:samp_end]

            yield df_row.copy(), lazy_loader
