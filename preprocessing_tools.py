import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import tsaug
from scipy.stats import iqr

from tqdm import tqdm


def read_data(ecg: str, desc: str = None):
    """Read data and remove some garbage (empty rows/columns).
    
    Parameters
    ----------
    ecg : str
        Path to ECG data for 7 leads.
    desc : str, optional
        Path to descriptional data for `ecg` table.
        Contains lables for a certain amount of rows from `ecg`.
        
    Returns
    -------
    pandas_ecg : pandas df
        Slightly preprocessed ecg data.
    pandas_desc : pandas df
        Slightly preprocessed descriptional data.
        
    Notes
    -----
    Description table gets new column with label: 1 - bigeminy, 0 - other problem or nothing, -1 - outlier.
    Outliers will be dropped.
    """
    pandas_ecg = pd.read_csv(ecg, sep=' ', header=None).iloc[:, :7]
    pandas_ecg.columns = ['i', 'ii', 'v5', 'iii', 'avl', 'avr', 'avf']
    
    if desc:
        pandas_desc = pd.read_excel(desc, header=[0, 1]).iloc[:, :6]
        pandas_desc.columns = ['time', 'ms', 'n_rows', 'complex', 'arr_code', 'arr']
        pandas_desc['label'] = np.select(
            condlist=[pandas_desc['complex'].isin(['X', 'Z']), pandas_desc['arr_code'] == 'ANZA'],
            choicelist=[-1, 1],
            default=0
        )
    else:
        pandas_desc = None

    return pandas_ecg, pandas_desc


def add_noise(arr: np.ndarray, scale: float = 0.1):
    """Add normal noise to data.
    
    Parameters
    ----------
    arr : array-like
        Array to noise.
    scale : float, optional
        Scale of noise.
        
    Returns
    -------
    noised_array : array-like
        Array with noised values.
    """
    noised_array = tsaug.AddNoise(scale=scale).augment(arr)
    return noised_array


def resample_ts(X: np.ndarray, y: np.ndarray, scale: float = 0.1):
    """Resample time-series to make classes balanced.
    
    Parameters
    ----------
    X : array-like
        Array with feature values.
    y : array-like
        Array with binary label (0 and 1), where there are less 0's than 1's.
    scale : float, optional
        Scale of noise to add to new resampled observations.
        
    Returns
    -------
    new_X : array-like
        Array with original and resampled noised features.
    new_y : 
        Array with original and new sample of imbalanced class.
    """
    label_a_idx = np.where(y == 0)[0]
    label_b_idx = np.where(y == 1)[0]
    n_to_fill = len(label_a_idx) - len(label_b_idx)
    
    random_idx = np.random.choice(label_b_idx, size=n_to_fill, replace=True)
    noised_samples = np.apply_along_axis(lambda x: add_noise(x, scale=scale), axis=0, arr=X[random_idx, :])
    more_labels = np.ones(shape=n_to_fill)
    
    new_X = np.vstack((X, noised_samples))
    new_y = np.hstack((y, more_labels))
    
    return new_X, new_y


def scale_array(arr: np.ndarray, a: float = -1, b: float = 1):
    """Apply min-max scaling between `a` and `b` for a given array.
    
    Parameters
    ----------
    arr : array-like
        Numpy array with values.
    a : float, optional
        The lowest threshold for scaled values.
    b : float, optional
        The largest threshold for scaled values.
        
    Returns
    -------
    ab_scaled : array-like
        Numpy array with scaled values.
    """
    base_scaled = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    ab_scaled = base_scaled * (b - a) + a
    
    return ab_scaled


def clip_array(arr: np.ndarray, min_percentile: float = 1, max_percentile: float = 99, use_iqr: bool = True):
    """Clip array values by its respective min and max percentiles.
    
    Parameters
    ----------
    arr : array-like
        Numpy array with values.
    min_percentile : float, optional
        Values lower than this percentile (taken from original array)
        will be clipped to this percentile.
    max_percentile : float, optional
        Values larger than this perentile (taken from original array)
        will be clipped to this percentile.
    use_iqr : bool, optional
        Whether to use IQR to clip outliers.
        If True, `min_percentile` and `max_percentile` are ignored.
    
    Returns
    -------
    clipped_arr : array-like
        Numpy array with clipped values.
    """
    if use_iqr:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        min_value, max_value = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    else:
        min_value, max_value = np.percentile(arr, [min_percentile, max_percentile])
        
    clipped_arr = np.clip(arr, a_min=min_value, a_max=max_value)
    
    return clipped_arr


def resize_array(arr: np.ndarray, resize_to: int = 1330, pad_smaller: bool = True):
    """Resize array length with interpolation.
    
    Parameters
    ----------
    arr : array-like
        1-D Numpy array with values.
    resize_to : int, optional
        Length which array will be resized to.
    pad_smaller : bool, optional
        Pad array with 0's on the right side if its length is less than `resize_to` value.
        
    Returns
    -------
    resized_arr : array-like
        1-D Numpy array of resized length.
    """
    if pad_smaller and len(arr) < resize_to:
        resized_arr = np.pad(arr, pad_width=(0, resize_to - len(arr)), constant_values=0)
    elif len(arr) == resize_to:
        resized_arr = arr
    else:
        resized_arr = tsaug.Resize(size=resize_to).augment(arr)
    
    return resized_arr


def preprocess_desc(desc: pd.DataFrame, row_window: int = 1330):
    """Preprocess descriptional data for ECG.
    
    Parameters
    ----------
    desc : pandas dataframe
        Description for ECG data.
    row_window : int, optional
        Aggregate rows data until number of rows reach this limit.
    
    Returns
    -------
    new_row_number : list
        New number of rows under a common label.
    new_labels : list
        New labels corresponding to row numbers.
    """
    prev_label = 99
    prev_row_number = 0
    n_rows = 0
    new_row_numbers, new_labels = [], []
    
    for row_number, label in zip(desc['n_rows'], desc['label']):
        if label == prev_label and n_rows < row_window:
            new_row_numbers[-1] = row_number
            n_rows += row_number - prev_row_number
            
            prev_row_number = row_number
            prev_label = label
        else:
            new_row_numbers.append(row_number)
            new_labels.append(label)
            
            n_rows = row_number - prev_row_number
            
            prev_row_number = row_number
            prev_label = label
            
    return new_row_numbers, new_labels


class Preprocessor:
    """Preprocess data with chained usage of functions."""
    
    def __init__(
        self,
        desc_params: dict = None,
        resize_params: dict = None,
        clip_params: dict = None,
        scale_params: dict = None
    ):
        """Initialize self.
        
        Parameters
        ----------
        desc_params : dict, optional
            Refer to `preprocess desc` function for more info.
        resize_params : dict, optional
            Refer to `resize_array` function for more info.
        clip_params : dict, optional
            Refer to `clip_array` function for more info.
        scale_params : dict, optional
            Refer to `scale_array` function for more info.
        """
        self.desc_params = desc_params if desc_params else {}
        self.resize_params = resize_params if resize_params else {}
        self.clip_params = clip_params if clip_params else {}
        self.scale_params = scale_params if scale_params else {}
        
    def preprocess(self, ecg: pd.DataFrame, desc: pd.DataFrame, drop_less_than: int = 1330):
        """Create a pipeline for data preprocessing.
    
        Parameters
        ----------
        ecg : pandas datafarme
            ECG data for 7 leads.
        desc : pandas dataframe
            Description for ECG data.
        drop_less_than : int, optional
            Delete those aggregated label which has less rows than a specified value.

        Returns
        -------
        lead_dct : dict
            Dict with keys: `i`, `ii`, `v5`, `iii`, `avl`, `avr`, `avf` for X values and `y` for label.
        """
        lead_dct = {}
        new_row_numbers, new_labels = preprocess_desc(desc, **self.desc_params)
        non_corrupt_samples = [
            idx for idx, label in enumerate(new_labels) if label in [0, 1]
        ]
        large_samples = [
            idx + 1 for idx, diff in enumerate(np.diff(new_row_numbers)) if diff > drop_less_than
        ]
        correct_rows = set(non_corrupt_samples).intersection(set(large_samples))

        for col in tqdm(ecg.columns):
            splitted_arr = np.split(ecg[col].values, indices_or_sections=new_row_numbers)
            resized_arr = [
                resize_array(arr, **self.resize_params)
                for idx, arr in enumerate(splitted_arr) if idx in correct_rows
            ]
            stacked_arr = np.vstack(resized_arr)
            clipped_arr = clip_array(stacked_arr, **self.clip_params)
            scaled_arr = scale_array(clipped_arr, **self.scale_params)

            lead_dct[col] = scaled_arr.astype(np.float32)
            
        lead_dct['y'] = np.array(
            [label for idx, label in enumerate(new_labels) if idx in correct_rows]
        ).astype(np.int8)

        return lead_dct
    
    def preprocess_unseen(self, ecg: pd.DataFrame, step: int = 266, size: int = 1330):
        """Create a pipeline for preprocessing unseen data.
        
        Parameters
        ----------
        ecg : pandas datafarme
            ECG data for 7 leads.
        step : int, optional
            Prediction window step.
        size : int, optional
            Window size.
        
        Returns
        -------
        lead_dct : dict
            Dict with keys: `i`, `ii`, `v5`, `iii`, `avl`, `avr`, `avf` for X values.
        idx : array-like
            Indices to divide array.
        """
        lead_dct = {}
        
        for lead in tqdm(['i', 'ii', 'v5', 'iii', 'avl', 'avr', 'avf']):
            clipped_array = clip_array(ecg[lead].values, **self.clip_params)
            scaled_array = scale_array(clipped_array, **self.scale_params)
            sliding_array = sliding_window_view(x=scaled_array, window_shape=size)[::step]
            if sliding_array[-1, -1] != scaled_array[-1]:
                idx = np.where(scaled_array == sliding_array[-1, -1])[0][-1] + 1
                sliding_array = np.vstack((
                    sliding_array,
                    resize_array(scaled_array[idx:], **self.resize_params)
                ))
            
            lead_dct[lead] = sliding_array
            
        return lead_dct
    