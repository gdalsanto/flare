from typing import Union

import torch
import pyfar as pf
import pandas as pd
import scipy
from scipy.io import savemat
import numpy as np
from numpy.typing import ArrayLike


def rt2slope(rt60: torch.Tensor, fs: int):
    """
    Convert RT60 (reverberation time) values to energy decay slopes.
    
    RT60 is the time required for the sound pressure level to decrease by 60 dB.
    This function converts RT60 values in seconds to the corresponding energy
    decay slope used in reverberation modeling.
    
    Parameters
    ----------
    rt60 : torch.Tensor
        Reverberation time values in seconds. Can be a scalar or tensor.
    fs : int
        Sampling frequency in Hz.
        
    Returns
    -------
    torch.Tensor
        Energy decay slopes. The slope is calculated as -60 / (rt60 * fs).
        For a 60 dB decay over rt60 seconds, this gives the per-sample decay rate.
    
    Examples
    --------
    >>> rt60 = torch.tensor([1.0, 2.0])  # 1s and 2s RT60
    >>> fs = 48000
    >>> slopes = rt2slope(rt60, fs)
    >>> print(slopes)  # [-0.00125, -0.000625]
    """
    return -60 / (rt60 * fs)


def df2mat(input_pickle: Union[str, pd.DataFrame], output_mat: str):
    """
    Convert a pandas DataFrame to a MATLAB .mat file.
    
    This function takes either a pandas DataFrame directly or a path to a pickle file
    containing a DataFrame, and converts it to a MATLAB .mat file format for
    cross-platform data sharing.
    
    Parameters
    ----------
    input_pickle : Union[str, pd.DataFrame]
        Either a path to a pickle file containing a pandas DataFrame, or a DataFrame
        object directly.
    output_mat : str
        Path where the output MATLAB .mat file will be saved.
        
    Returns
    -------
    None
        The function saves the .mat file to the specified path and prints a
        confirmation message.
        
    Notes
    -----
    - Each column in the DataFrame becomes a variable in the .mat file
    - Column names become variable names in MATLAB
            
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'time': [1, 2, 3], 'amplitude': [0.1, 0.2, 0.3]})
    >>> df2mat(df, 'data.mat')  # Save DataFrame directly
    """
    if isinstance(input_pickle, pd.DataFrame):
        # If input is already a DataFrame, use it directly
        df = input_pickle
    else:
        # Load DataFrame from pickle file
        df = pd.read_pickle(input_pickle)

    # Convert DataFrame to dict of lists for savemat
    # Each column becomes a key in the dictionary with its values as a list
    mat_dict = {col: df[col].to_list() for col in df.columns}

    savemat(output_mat, mat_dict)
    print(f"Successfully saved DataFrame to MATLAB .mat file: {output_mat}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def ms_to_samps(
    ms: Union[float, ArrayLike, torch.Tensor], fs: float
) -> Union[int, ArrayLike, torch.Tensor]:
    """
    Convert time values from milliseconds to samples.
    
    This function converts time durations from milliseconds to the corresponding
    number of samples based on the given sampling frequency. It supports both
    scalar and array inputs, and works with both NumPy arrays and PyTorch tensors.
    
    Parameters
    ----------
    ms : Union[float, ArrayLike, torch.Tensor]
        Time values in milliseconds. Can be a scalar, NumPy array, or PyTorch tensor.
    fs : float
        Sampling frequency in Hz.
        
    Returns
    -------
    Union[int, ArrayLike, torch.Tensor]
        Time values converted to samples. The return type matches the input type:
        - float -> int (for scalar inputs)
        - ArrayLike -> ArrayLike (for NumPy array inputs)
        - torch.Tensor -> torch.Tensor (for PyTorch tensor inputs)
        
    Notes
    -----
    The conversion is performed using the formula: samples = ms * 1e-3 * fs
    For PyTorch tensors, the result is converted to integer type.
    For NumPy arrays, scalars are converted to int, arrays to int32.
        
    Examples
    --------
    >>> fs = 48000
    >>> ms_to_samps(10.0, fs)  # 10ms at 48kHz
    480
    >>> ms_to_samps(np.array([10, 20, 30]), fs)  # NumPy array
    array([ 480,  960, 1440], dtype=int32)
    >>> ms_to_samps(torch.tensor([10.0, 20.0]), fs)  # PyTorch tensor
    tensor([480, 960], dtype=torch.int32)
    """
    if isinstance(ms, torch.Tensor):
        samp = ms * 1e-3 * torch.tensor(fs)
        return samp.int()
    else:
        samp = ms * 1e-3 * fs
        if np.isscalar(samp):
            return int(samp)
        else:
            return samp.astype(np.int32)


def tanh_neg_x(
    x: Union[float, ArrayLike, torch.Tensor],
    a: Union[float, ArrayLike, torch.Tensor] = 1,
    b: Union[float, ArrayLike, torch.Tensor] = 1,
    c: Union[float, ArrayLike, torch.Tensor] = 0,
    d: Union[float, ArrayLike, torch.Tensor] = 0,
):
    """
    Apply a modified hyperbolic tangent function with negative input scaling.
    
    This function implements the transformation: a * tanh(-b * x + c) + d
    where the input x is scaled by -b (negative scaling) before applying the tanh function.
    This is commonly used in audio processing and reverberation modeling for
    non-linear transformations with bounded output.
    
    Parameters
    ----------
    x : Union[float, ArrayLike, torch.Tensor]
        Input values to be transformed.
    a : Union[float, ArrayLike, torch.Tensor], optional
        Amplitude scaling parameter. Default is 1.
    b : Union[float, ArrayLike, torch.Tensor], optional
        Input scaling parameter (applied with negative sign). Default is 1.
    c : Union[float, ArrayLike, torch.Tensor], optional
        Horizontal shift parameter. Default is 0.
    d : Union[float, ArrayLike, torch.Tensor], optional
        Vertical shift parameter. Default is 0.
        
    Returns
    -------
    Union[float, ArrayLike, torch.Tensor]
        Transformed values: a * tanh(-b * x + c) + d
        The return type matches the input type of x.
        
    Notes
    -----
    - The negative scaling (-b * x) inverts the typical tanh behavior
    - Parameter a controls the output range: [-a + d, a + d]
    - Parameter b controls the steepness of the transition
    - Parameter c shifts the transition point horizontally
    - Parameter d shifts the output vertically
    - All parameters are automatically converted to the appropriate type (tensor or array)
        
    Examples
    --------
    >>> x = np.array([-2, -1, 0, 1, 2])
    >>> tanh_neg_x(x)  # Default parameters
    array([ 0.96402758,  0.76159416,  0.        , -0.76159416, -0.96402758])
    >>> tanh_neg_x(x, a=2, b=0.5, c=1, d=0.5)  # Custom parameters
    array([ 2.42805516,  2.31029651,  2.02318831,  1.42423431,  0.5       ])
    """
    if isinstance(x, torch.Tensor):
        # Convert parameters to tensors if they aren't already
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=x.dtype, device=x.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=x.dtype, device=x.device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=x.dtype, device=x.device)
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=x.dtype, device=x.device)

        return a * torch.tanh(-(b * x) + c) + d
    else:
        # Convert parameters to numpy arrays if they aren't already
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        d = np.asarray(d)

        return a * np.tanh(-(b * x) + c) + d

def filterbank(x: Union[np.ndarray, torch.Tensor],
               n_fractions: int = 1,
               f_min: int = 63,
               f_max: int = 16000,
               sample_rate: int = 48000,
               order: int = 14,
               compensate_energy: bool = True,
               filter_type: str = 'pyfar') -> Union[np.ndarray, torch.Tensor]:
    """
    Apply a fractional octave filterbank to a time-domain signal.
    
    This function applies a bank of bandpass filters to decompose a signal into
    frequency bands. It supports both PyTorch tensors and NumPy arrays, and can
    use either PyFAR-based filtering or Butterworth filters.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Input time-domain signal. Can be 1D or multi-dimensional.
        For multi-dimensional signals, filtering is applied along the second dimension.
    n_fractions : int, optional
        Number of fractions per octave. Default is 1 (full octave bands).
        Common values: 1 (full octave), 3 (1/3 octave), 6 (1/6 octave).
    f_min : int, optional
        Minimum frequency of the filterbank in Hz. Default is 63 Hz.
    f_max : int, optional
        Maximum frequency of the filterbank in Hz. Default is 16000 Hz.
    sample_rate : int, optional
        Sampling rate of the input signal in Hz. Default is 48000 Hz.
    order : int, optional
        Order of the Butterworth filters.
        Default is 14.
    compensate_energy : bool, optional
        If True, compensate for energy loss in filtering by normalizing each
        filter's frequency response. Default is True.
    filter_type : str, optional
        Type of filterbank to use:
        - 'pyfar': Use PyFAR library for filtering (default)
        - 'butter': Use Butterworth filters from SciPy
        
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Filtered signal with shape (n_bands, *x.shape) where n_bands is the
        number of frequency bands. The return type matches the input type.
        
    Notes
    -----
    - For PyTorch tensors, the function temporarily converts to NumPy for processing
      and then converts back to maintain compatibility
    - The filterbank covers the frequency range from f_min to f_max
    - Center frequencies are logarithmically spaced
    - Energy compensation helps maintain the overall signal energy across bands
    - The 'pyfar' option generally provides better performance for audio applications
        
    Examples
    --------
    >>> import numpy as np
    >>> # Create a test signal
    >>> t = np.linspace(0, 1, 48000)
    >>> x = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 5000 * t)
    >>> # Apply 1/3 octave filterbank
    >>> filtered, freqs = filterbank(x, n_fractions=3, f_min=125, f_max=8000)
    >>> print(f"Number of bands: {filtered.shape[0]}")
    >>> print(f"Center frequencies: {freqs}")
    """
    # Handle PyTorch tensor input
    is_torch_input = isinstance(x, torch.Tensor)
    original_device = None
    original_dtype = None
    
    if is_torch_input:
        original_device = x.device
        original_dtype = x.dtype
        # Convert to numpy for processing
        x_np = x.cpu().numpy()
    else:
        x_np = x

    if x_np.ndim == 1:
        # If input is 1D, reshape to (1, n_samples) for consistency
        x_np = x_np[np.newaxis, :]

    # Create impulse for filter design
    impulse = np.zeros(x_np.shape[1])
    impulse[0] = 1.0

    # Get center frequencies for fractional octave bands
    center_freqs = pf.dsp.filter.fractional_octave_frequencies(
        num_fractions=n_fractions,
        frequency_range=(f_min, f_max),
        return_cutoff=False)[0]

    if filter_type == 'butter':
        # Design SOS filters for each band
        sos_filters = []
        for band_idx, center_freq in enumerate(center_freqs):
            if abs(center_freq) < 1e-6:
                # Lowpass for lowest band
                f_cutoff = (1 / np.sqrt(2)) * center_freqs[band_idx + 1]
                sos = scipy.signal.butter(N=order,
                                          Wn=f_cutoff,
                                          fs=sample_rate,
                                          btype='lowpass',
                                          output='sos')
            elif abs(center_freq - sample_rate / 2) < 1e-6:
                # Highpass for highest band
                f_cutoff = np.sqrt(2) * center_freqs[band_idx - 1]
                sos = scipy.signal.butter(N=order,
                                          Wn=f_cutoff,
                                          fs=sample_rate,
                                          btype='highpass',
                                          output='sos')
            else:
                # Bandpass for intermediate bands
                f_cutoff = center_freq * np.array([1 / np.sqrt(2), np.sqrt(2)])
                sos = scipy.signal.butter(N=order,
                                          Wn=f_cutoff,
                                          fs=sample_rate,
                                          btype='bandpass',
                                          output='sos')
            sos_filters.append(sos)

        # Apply each filter to the signal
        filtered = [
            scipy.signal.sosfilt(sos, x_np, axis=1) for sos in sos_filters
        ]
        y = np.stack(filtered, axis=0)

    elif filter_type == 'pyfar':
        # Get frequency responses for each band
        f_bank = pf.dsp.filter.fractional_octave_bands(
            pf.Signal(impulse, sample_rate),
            num_fractions=n_fractions,
            frequency_range=(f_min, f_max),
            order=order,
        ).freq.T  # shape: (filter_length, n_bands)
        f_bank = np.squeeze(f_bank)
        y = np.zeros((f_bank.shape[1], *x_np.shape[1:]))

        # FFT of input signal
        X = np.fft.rfft(x_np, n=x_np.shape[1], axis=1)
        for i_band in range(f_bank.shape[1]):
            filt = np.pad(f_bank[:, i_band], (0, X.shape[1] - f_bank.shape[0]))
            if compensate_energy:
                norm = np.sqrt(np.sum(np.abs(filt)**2))
                Y_band = X * filt / norm
            else:
                Y_band = X * filt
            y[i_band, ...] = np.fft.irfft(Y_band, n=x_np.shape[1], axis=1)

    # Convert back to tensor if input was tensor
    if is_torch_input:
        y = torch.from_numpy(y).to(device=original_device, dtype=original_dtype)
        center_freqs = torch.from_numpy(center_freqs).to(device=original_device, dtype=original_dtype)

    return y, center_freqs


def discard_last_n_percent(x: Union[np.ndarray, torch.Tensor], percent: float) -> Union[np.ndarray, torch.Tensor]:
    """
    Remove the last n percent of samples from a signal.
    
    This function truncates a signal by removing the specified percentage of samples
    from the end. It's commonly used in audio processing to remove unwanted tail
    portions of signals or to standardize signal lengths.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Input signal to be truncated. Can be 1D or multi-dimensional.
        For multi-dimensional signals, truncation is applied along the second dimension.
    percent : float
        Percentage of samples to remove from the end of the signal.
        Must be between 0 and 100. A value of 0 means no samples are removed.
        
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Truncated signal with the last n percent of samples removed.
        The return type matches the input type.
        
    Notes
    -----
    - The function calculates the number of samples to remove as: n_samples = int(length * percent / 100)
    - If the calculated number of samples is 0, the original signal is returned unchanged
    - The truncation is applied along the last dimension for multi-dimensional signals
    - This is useful for removing reverb tails, noise, or standardizing signal lengths
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> discard_last_n_percent(x, 20)  # Remove last 20%
    array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> discard_last_n_percent(x, 0)   # Remove 0% (no change)
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    >>> # Multi-dimensional signal
    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> discard_last_n_percent(x_2d, 50)  # Remove last 50% of each row
    array([[1, 2],
           [5, 6]])
    """
    if x.ndim == 1:
        # If input is 1D, reshape to (1, n_samples) for consistency
        x = x[np.newaxis, :]

    if isinstance(x, torch.Tensor):
        n_samples = int(x.shape[1] * percent / 100)
        return x[:, :-n_samples] if n_samples > 0 else x
    else:
        n_samples = int(x.shape[1] * percent / 100)
        return x[:, :-n_samples] if n_samples > 0 else x


def rt_from_sabine(surface: torch.Tensor, surface_absorption: torch.Tensor, room_dim: torch.Tensor) -> torch.Tensor:
    """
    Calculate reverberation time (RT60) using Sabine's formula.
    
    This function computes the reverberation time based on the surface area of
    the room and the absorption coefficients of the materials. It uses Sabine's
    formula: RT60 = 0.161 * V / (S * A), where V is the room volume, S is the total
    surface area, and A is the average absorption.
    
    Parameters
    ----------
    surface : torch.Tensor
        Surface area of the room or the walls in square meters. Can be a scalar or tensor.
        If multiple surfaces are provided, the shape should be compatible with the surface_aboprtion tensor.
    surface_absorption : torch.Tensor
        Absorption coefficients of the materials in the room. Shape: [n_surfaces, n_frequencies].
    room_dim : torch.Tensor
        Dimensions of the room as a tensor with shape (3,) representing length,
        width, and height in meters.
        
    Returns
    -------
    torch.Tensor
        Reverberation time (RT60) in seconds. The return type matches the input type.

    """
    assert len(surface.shape) == (len(surface_absorption.shape) - 1), "Surface and surface absorption must have compatible shapes."
    # Calculate the average absorptivity in the room
    mean_absorption = torch.sum(torch.einsum('s, sf -> sf', surface , surface_absorption), dim=0) / surface.sum()
    volume = torch.prod(room_dim)
    rt60 = 0.161 * volume / (surface.sum() * mean_absorption)  # Scale by sampling frequency
    return rt60


def find_coprime_numbers(min_value, max_value, num_numbers, target_sum):
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def is_coprime(a, b):
        return gcd(a, b) == 1

    def get_logarithmically_distributed_values(min_value, max_value, num_numbers):
        # Generate logarithmically distributed values between min_value and max_value
        log_min = np.log(min_value)
        log_max = np.log(max_value)
        log_values = np.linspace(log_min, log_max, num_numbers)
        values = np.exp(log_values).astype(int)
        return np.unique(values)  # Remove duplicates and return

    def get_closest_numbers(values, reference_values):
        closest_numbers = []
        for i in values:
            closest_numbers.append(min(reference_values, key=lambda x: abs(x - i)))
        if len(np.unique(closest_numbers)) != len(closest_numbers):
            print("Warning: sampling duplicate values")
            
        return closest_numbers

    # Start with logarithmically distributed values
    prime_numbers = list(sp.primerange(min_value, max_value))
    log_values = get_logarithmically_distributed_values(min_value, max_value, num_numbers)
    coprime_numbers = get_closest_numbers(log_values, prime_numbers)

    current_sum = sum(coprime_numbers)
    while current_sum < 0.9 * target_sum or current_sum > target_sum * 1.1:  # allow 2% error
        if current_sum < target_sum:
            coprime_numbers = [sp.nextprime(num)  for num in coprime_numbers]
        else:
            coprime_numbers = [sp.prevprime(num)  for num in coprime_numbers]
        current_sum = sum(coprime_numbers)
        
    return coprime_numbers    