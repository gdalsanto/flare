import sys
import numpy as np
from typing import Union, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.fft import rfft, irfft
from scipy.signal import spectrogram
from scipy.stats import linregress

from flareverb.utils import (
    ms_to_samps, 
    filterbank, 
    discard_last_n_percent)

Tensor = torch.Tensor
NDArray = np.ndarray


def schroeder_backward_int(
    x: Union[Tensor, NDArray],
    energy_norm: bool = True,
    subtract_noise: bool = False,
    noise_level: float = 0.0,
) -> Tuple[Union[Tensor, NDArray], Union[Tensor, NDArray]]:
    """
    Compute the backward integration of the squared impulse response (Schroeder integration).

    Parameters
    ----------
    x : Union[Tensor, NDArray]
        Input signal (impulse response.
    energy_norm : bool, optional
        If True, normalize the output to its maximum value (default: True).
    subtract_noise : bool, optional
        If True, subtract the squared noise level from the squared signal (default: False).
    noise_level : float, optional
        The noise level to subtract if subtract_noise is True (default: 0.0).

    Returns
    -------
    tuple of Union[Tensor, NDArray]
        Tuple containing the backward integrated and normalized array, and the normalization value(s) used.
    """
    if isinstance(x, torch.Tensor):
        return _schroeder_backward_int_torch(x, energy_norm, subtract_noise, noise_level)
    else:
        return _schroeder_backward_int_numpy(x, energy_norm, subtract_noise, noise_level)


def _schroeder_backward_int_torch(
    x: Tensor,
    energy_norm: bool,
    subtract_noise: bool,
    noise_level: float,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation of Schroeder backward integration.
    
    This function computes the Schroeder backward integration for PyTorch tensors.
    The integration is performed by flipping the signal, computing the cumulative
    sum of squared values, and then flipping back.
    
    Parameters
    ----------
    x : Tensor
        Input signal tensor to be integrated.
    energy_norm : bool
        If True, normalize the output to its maximum value.
    subtract_noise : bool
        If True, subtract the squared noise level from the squared signal.
    noise_level : float
        The noise level to subtract if subtract_noise is True.
        
    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing:
        - out: The backward integrated and normalized signal
        - norm_vals: The normalization values used (maximum values per channel)
        
    Notes
    -----
    - If subtract_noise is True, noise_level^2 is subtracted from the squared signal
    - Normalization is useful for t60 estimation from the EDC
    """
    out = torch.flip(x, dims=[1])
    if subtract_noise:
        out_sqrd = out ** 2 - noise_level ** 2
    else:
        out_sqrd = out ** 2
    out = torch.cumsum(out_sqrd, dim=1)
    out = torch.flip(out, dims=[1])

    # Normalize to 1
    if energy_norm:
        norm_vals = torch.max(out, dim=1, keepdim=True)[0]  # per channel
    else:
        norm_vals = torch.ones_like(out)

    return out / norm_vals, norm_vals


def _schroeder_backward_int_numpy(
    x: NDArray,
    energy_norm: bool,
    subtract_noise: bool,
    noise_level: float,
) -> Tuple[NDArray, NDArray]:
    """
    NumPy implementation of Schroeder backward integration.
    
    This function computes the Schroeder backward integration for NumPy arrays.
    The integration is performed by flipping the signal, computing the cumulative
    sum of squared values, and then flipping back.
    
    Parameters
    ----------
    x : NDArray
        Input signal array to be integrated.
    energy_norm : bool
        If True, normalize the output to its maximum value.
    subtract_noise : bool
        If True, subtract the squared noise level from the squared signal.
    noise_level : float
        The noise level to subtract if subtract_noise is True.
        
    Returns
    -------
    Tuple[NDArray, NDArray]
        A tuple containing:
        - out: The backward integrated and normalized signal
        - norm_vals: The normalization values used (maximum values per channel)
        
    Notes
    -----
    - If subtract_noise is True, noise_level^2 is subtracted from the squared signal
    - Normalization is useful for t60 estimation from the EDC
    """
    out = np.flip(x, axis=1)
    if subtract_noise:
        out_sqrd = out ** 2 - noise_level ** 2
    else:
        out_sqrd = out ** 2
    out = np.cumsum(out_sqrd, axis=1)
    out = np.flip(out, axis=1)

    # Normalize to 1
    if energy_norm:
        norm_vals = np.max(out, keepdims=True, axis=1)  # per channel
    else:
        norm_vals = np.ones_like(out)

    return out / norm_vals, norm_vals


def compute_edc(
    x: Union[Tensor, NDArray],
    use_filterbank: bool = False,
    compensate_fbnk_energy: bool = True,
    n_fractions: int = 1,
    f_min: int = 63,
    f_max: int = 16000,
    fs: int = 48000,
    energy_norm: bool = True,
    subtract_noise: bool = False,
    noise_level: float = 0.0,
) -> Union[Tensor, NDArray]:
    """
    Compute the Energy Decay Curve (EDC) in dB from an input signal.
    
    The Energy Decay Curve shows how the energy of a room impulse response
    decays over time. It is computed using Schroeder backward integration
    and can optionally use frequency band filtering for multi-band analysis.
    
    Parameters
    ----------
    x : Union[Tensor, NDArray]
        Input signal (room impulse response) to analyze.
    use_filterbank : bool, optional
        If True, apply filterbank processing to compute EDCs for multiple
        frequency bands. Default is False.
    compensate_fbnk_energy : bool, optional
        If True, compensate for energy loss in filterbank processing.
        Only used when use_filterbank is True. Default is True.
    n_fractions : int, optional
        Number of fractions per octave for filterbank analysis.
        Only used when use_filterbank is True. Default is 1 (full octave).
    f_min : int, optional
        Minimum frequency for filterbank analysis in Hz.
        Only used when use_filterbank is True. Default is 63 Hz.
    f_max : int, optional
        Maximum frequency for filterbank analysis in Hz.
        Only used when use_filterbank is True. Default is 16000 Hz.
    fs : int, optional
        Sampling rate in Hz. Default is 48000 Hz.
    energy_norm : bool, optional
        If True, normalize the output to its maximum value. Default is True.
    subtract_noise : bool, optional
        If True, subtract the squared noise level from the squared signal.
        Default is False.
    noise_level : float, optional
        The noise level to subtract if subtract_noise is True. Default is 0.0.
        
    Returns
    -------
    Union[Tensor, NDArray]
        The energy decay curve in dB. If use_filterbank is True, returns
        EDCs for multiple frequency bands with shape (n_bands, time).
        Otherwise, returns a single EDC with shape (time,).
        
    Notes
    -----
    - The function removes the last 0.5 permille of samples to avoid filtering artifacts
    - Schroeder backward integration is used to compute the energy decay
    - The result is converted to dB using 10 * log10()
    """
    # Remove filtering artefacts (last 0.5 permille)
    out = discard_last_n_percent(x, 0.5)
    
    if use_filterbank:
        # Use filterbank to compute EDCs
        out, _ = filterbank(out, n_fractions=n_fractions, f_min=f_min, f_max=f_max, 
                           sample_rate=fs, compensate_energy=compensate_fbnk_energy)
    
    # compute EDCs
    out, _ = schroeder_backward_int(out, energy_norm, subtract_noise, noise_level)
    
    # get energy in dB
    if isinstance(out, torch.Tensor):
        out = 10 * torch.log10(out + 1e-32)
    else:
        out = 10 * np.log10(out + 1e-32)

    return out


def compute_edr(
    x: Union[Tensor, NDArray],
    energy_norm: bool = True,
    subtract_noise: bool = False,
    noise_level: float = 0.0,
) -> Union[Tensor, NDArray]:
    """
    Compute the Energy Decay Relief (EDR) in dB from an input signal using STFT.
    
    The Energy Decay Relief provides a time-frequency representation of how
    the energy decays over time and frequency. It is computed by applying
    Schroeder backward integration to the magnitude spectrogram.
    
    Parameters
    ----------
    x : Union[Tensor, NDArray]
        Input signal (room impulse response) to analyze.
    energy_norm : bool, optional
        If True, normalize the output to its maximum value. Default is True.
    subtract_noise : bool, optional
        If True, subtract the squared noise level from the squared signal.
        Default is False.
    noise_level : float, optional
        The noise level to subtract if subtract_noise is True. Default is 0.0.
        
    Returns
    -------
    Union[Tensor, NDArray]
        The energy decay relief in dB. The output has shape (frequency_bins, time_frames)
        representing the energy decay over time for each frequency bin.
        
    Notes
    -----
    - The function removes the last 0.5 permille of samples to avoid filtering artifacts
    - Short-time Fourier transform (STFT) is used to obtain the time-frequency representation
    - Schroeder backward integration is applied to the magnitude spectrogram
    - The result is converted to dB using 10 * log10()
    """
    # Remove filtering artefacts (last 0.5 permille)
    out = discard_last_n_percent(x, 0.5)
    
    if isinstance(out, torch.Tensor):
        # PyTorch STFT implementation
        stft_mag = _stft_torch(out)
    else:
        # NumPy STFT using scipy
        _, _, stft_mag = spectrogram(out, nperseg=1028, noverlap=int(1028 * 0.75), mode='magnitude', axis=1)
        stft_mag = torch.tensor(stft_mag)

    # compute EDRs
    out, _ = schroeder_backward_int(stft_mag, energy_norm, subtract_noise, noise_level)

    return 10*torch.log10(out)


def _stft_torch(x: Tensor, nperseg: int = 1028, noverlap: int = None) -> Tensor:
    """
    PyTorch implementation of STFT magnitude computation.
    
    This function computes the Short-time Fourier transform magnitude using PyTorch.
    It provides a time-frequency representation of the input signal using overlapping
    windows and FFT computation.
    
    Parameters
    ----------
    x : Tensor
        Input signal tensor to be analyzed.
    nperseg : int, optional
        Length of each segment (window length) in samples. Default is 1028.
    noverlap : int, optional
        Number of points to overlap between segments. If None, defaults to
        75% of nperseg. Default is None.
        
    Returns
    -------
    Tensor
        STFT magnitude tensor with shape (frequency_bins, time_frames).
    """
    if noverlap is None:
        noverlap = int(nperseg * 0.75)

    hop_length = nperseg - noverlap
    
    # Pad the signal
    pad_length = nperseg // 2
    x_padded = F.pad(x, (0, 0, pad_length, pad_length, 0, 0))
    
    # Create windows
    window = torch.hann_window(nperseg, dtype=x.dtype, device=x.device)
    
    # Compute STFT
    stft = torch.stft(x_padded.squeeze(), n_fft=nperseg, hop_length=hop_length, 
                     window=window, return_complex=True, center=False)
    
    return torch.abs(stft)


def estimate_rt60(
    edc_db: Union[Tensor, NDArray], 
    time: Union[Tensor, NDArray], 
    decay_start_db: float = -5, 
    decay_end_db: float = -35
) -> Tuple[float, float, float, Union[Tensor, NDArray]]:
    """
    Estimate the reverberation time (RT60) from an Energy Decay Curve (EDC) using linear regression.
    
    RT60 is the time required for the sound pressure level to decrease by 60 dB.
    This function estimates RT60 by fitting a linear regression to the decay portion
    of the energy decay curve.
    
    Parameters
    ----------
    edc_db : Union[Tensor, NDArray]
        Energy decay curve in dB. Should be a monotonically decreasing curve.
    time : Union[Tensor, NDArray]
        Time vector corresponding to the EDC samples in seconds.
    decay_start_db : float, optional
        Starting decay level in dB for the linear fit. The fit begins when the
        EDC drops below this level. Default is -5 dB.
    decay_end_db : float, optional
        Ending decay level in dB for the linear fit. The fit ends when the
        EDC drops below this level. Default is -35 dB.
        
    Returns
    -------
    Tuple[float, float, float, Union[Tensor, NDArray]]
        A tuple containing:
        - rt60 : float
            Estimated RT60 in seconds. Returns infinity if no valid decay range is found.
        - slope : float
            Slope of the linear fit in dB/s.
        - intercept : float
            Y-intercept of the linear fit.
        - valid_range : Union[Tensor, NDArray]
            Boolean array indicating the samples used for the fit.
            
    Notes
    -----
    - The function finds the range where the EDC is between decay_start_db and decay_end_db
    - Linear regression is performed on this range to estimate the decay rate
    - RT60 is calculated as -60 / slope (the time for 60 dB decay)
    - If no valid range is found, RT60 is set to infinity
    - The decay range should be chosen to avoid the initial build-up and noise floor
    - Typical values for T60 from a 30dB range are -5 dB to -35 dB, but may need
      adjustment for different signals
    """
    valid_range = (edc_db < decay_start_db) & (edc_db > decay_end_db)
    
    if not torch.any(valid_range):
        return float('inf'), 0.0, 0.0, valid_range
    
    if isinstance(edc_db, torch.Tensor):
        # Convert to numpy for linregress
        time_valid = time[valid_range.squeeze()].cpu().numpy()
        edc_valid = edc_db[valid_range].cpu().numpy()
    else: 
        time_valid = time[valid_range.squeeze()]
        edc_valid = edc_db[valid_range]

    slope, intercept, *_ = linregress(time_valid, edc_valid)
    rt60 = -60 / slope if slope != 0 else float('inf')
    
    return rt60, slope, intercept, valid_range


def normalized_echo_density(
    rir: Union[Tensor, NDArray],
    fs: float,
    window_length_ms: float = 30,
    use_local_avg: bool = True
) -> Union[Tensor, NDArray]:
    """
    Compute the normalized echo density profile as defined by Abel.
    
    Echo density measures how the density of reflections changes over time in a
    room impulse response. The normalized echo density provides a quantitative
    measure of the temporal evolution
    
    Parameters
    ----------
    rir : Union[Tensor, NDArray]
        Room impulse response to analyze.
    fs : float
        Sampling rate in Hz.
    window_length_ms : float, optional
        Length of the analysis window in milliseconds. Default is 30 ms.
    use_local_avg : bool, optional
        If True, use local average for weighted standard deviation calculation.
        This provides better estimates of the local signal characteristics.
        Default is True.
        
    Returns
    -------
    Union[Tensor, NDArray]
        Normalized echo density profile. The output has the same length as the
        input RIR and represents the echo density at each time point.
        
    Notes
    -----
    - The function uses a sliding window approach to analyze the RIR
    - For each window position, it computes the weighted standard deviation
    - Echo density is calculated as the ratio of samples above the threshold
    - The result is normalized by the complementary error function constant (0.3173)
    - This metric is useful for analyzing the temporal evolution of a RIR
    """
    if isinstance(rir, torch.Tensor):
        rir = rir.cpu().numpy()  # Convert to NumPy for processing

    def weighted_std(signal: NDArray, window_func: NDArray, use_local_avg: bool):
        """Return the weighted standard deviation of a signal."""
        if use_local_avg:
            average = np.average(signal, weights=window_func, axis=1)
            variance = np.average((signal - average)**2, weights=window_func, axis=1)
        else:
            variance = np.average((signal)**2, weights=window_func, axis=1)
        return np.sqrt(variance)
    
    # erfc(1/√2)
    ERFC = 0.3173
    window_length_samps = ms_to_samps(window_length_ms, fs)
    
    # Ensure window length is odd for symmetric windowing
    if not window_length_samps % 2:
        window_length_samps += 1
    
    half_window = int((window_length_samps - 1) / 2)

    # Pad the RIR to handle windowing at the edges
    padded_rir = np.pad(rir, ((0, 0), (half_window, half_window), (0, 0)), mode='constant')

    # Prepare output array and window function
    output = np.zeros(rir.shape[1] + 2 * half_window)
    window_func = np.hanning(window_length_samps)
    window_func = window_func / np.sum(window_func)
    
    # Slide window across RIR and compute normalized echo density
    for cursor in range(rir.shape[1]):
        frame = padded_rir[:, cursor:cursor + window_length_samps, :]
        std = weighted_std(frame, window_func, use_local_avg)
        # Count samples above weighted std, weighted by window
        count = np.sum((np.abs(frame) > std).squeeze() * window_func)
        # Normalize by ERFC constant
        output[cursor] = (1 / ERFC) * count
    
    ned = output[:-window_length_samps]
    return ned

def compute_mixing_time(ned, fs, mixing_thresh=0.9, pre_delay=0):
    """Estimate the mixing time from a normalized echo density (NED) profile.

    The mixing time is defined here as the first time (in milliseconds) when
    the normalized echo density exceeds the provided `mixing_thresh` value.

    Parameters
    ----------
    ned : ndarray
        Normalized Echo Density (NED) profile. Expected to be a 1-D array c
        computed with normalized_echo_density
    fs : float
        Sampling rate in Hz used to convert sample indices to time.
    mixing_thresh : float, optional
        Threshold in the NED above which the signal is considered "mixed".
        Default is 0.9.
    pre_delay : int, optional
        Number of samples to subtract from the detected index to account for
        any pre-delay; used when converting to time. Default is 0.

    Returns
    -------
    float
        Estimated mixing time in milliseconds. The value is computed as
        (index - pre_delay) / fs * 1000.

    Notes
    -----
    - The implementation uses ``np.argmax(ned > mixing_thresh)`` which
      returns the index of the first True value. If no element exceeds the
      threshold, ``np.argmax`` returns 0. The caller should ensure ``ned``
      covers a sufficient range such that this behaviour is acceptable. The
      function does not currently raise on a missing crossing.
    - ``pre_delay`` is expected in samples (not seconds).
    - Reference: Abel, Jonathan S., and Patty Huang. "A simple, robust measure 
    of reverberation echo density." Audio Engineering Society Convention 121. 
    Audio Engineering Society, 2006.
    """

    # determine mixing time
    d = np.argmax(ned >= mixing_thresh)
    t_mix = (d - pre_delay) / fs * 1000

    if t_mix is None:
        t_mix = 0
        print('Mixing time not found within given limits.')

    return t_mix

def compute_clarity_parameters(rir: Union[Tensor, NDArray], fs: float) -> tuple:
    """
    Compute clarity parameters (C50, C80) from a room impulse response.
    
    Clarity parameters measure the ratio of early to late arriving sound energy.
    C50 and C80 are calculated using 50ms and 80ms time boundaries respectively.
    Higher values indicate better speech intelligibility and music clarity.
    
    Parameters
    ----------
    rir : Union[Tensor, NDArray]
        Room impulse response to analyze.
    fs : float
        Sampling rate in Hz.
        
    Returns
    -------
    tuple
        A tuple containing:
        - c50 : float
            Clarity index at 50ms boundary in dB
        - c80 : float
            Clarity index at 80ms boundary in dB
    """
    # Time boundaries in samples
    t50_samples = int(50 * fs / 1000)
    t80_samples = int(80 * fs / 1000)

    # Early and late energy
    if isinstance(rir, torch.Tensor):
        early_energy_50 = torch.sum(rir[:, :t50_samples] ** 2)
        late_energy_50 = torch.sum(rir[:, t50_samples:] ** 2)

        early_energy_80 = torch.sum(rir[:, :t80_samples] ** 2)
        late_energy_80 = torch.sum(rir[:, t80_samples:] ** 2)

        # Clarity parameters
        c50 = 10 * torch.log10(early_energy_50 / (late_energy_50 + 1e-32))
        c80 = 10 * torch.log10(early_energy_80 / (late_energy_80 + 1e-32))
    else:
        early_energy_50 = np.sum(rir[:, :t50_samples] ** 2)
        late_energy_50 = np.sum(rir[:, t50_samples:] ** 2)

        early_energy_80 = np.sum(rir[:, :t80_samples] ** 2)
        late_energy_80 = np.sum(rir[:, t80_samples:] ** 2)

        # Clarity parameters
        c50 = 10 * np.log10(early_energy_50 / (late_energy_50 + 1e-32))
        c80 = 10 * np.log10(early_energy_80 / (late_energy_80 + 1e-32))

    return c50, c80

def compute_definition_parameters(rir: Union[Tensor, NDArray], fs: int, interval_ms = 50) -> tuple:
    """
    Compute definition parameters (D50, D80) from a room impulse response.
    
    Definition parameters measure the ratio of early to total sound energy.
    D50 and D80 are calculated using 50ms and 80ms time boundaries respectively.
    These parameters are related to clarity but use total energy as the denominator.
    
    Parameters
    ----------
    rir : Union[Tensor, NDArray]
        Room impulse response to analyze.
    fs : int
        Sampling rate in Hz.
    interval_ms : int, optional
        Time boundary in milliseconds for the definition calculation.
        Default is 50 ms (D50).
        
    Returns
    -------
    tuple
        A tuple containing:
        - D : float
            Definition parameter (ratio of early to total energy)
    """
    # Time boundaries in samples

    
    t_samples = int(interval_ms * fs / 1000)

    # Early and total energy
    if isinstance(rir, torch.Tensor):
        early_energy = torch.sum(rir[:, :t_samples] ** 2)
        total_energy = torch.sum(rir ** 2)
    else:
        early_energy = np.sum(rir[:, :t_samples] ** 2)
        total_energy = np.sum(rir ** 2)

    # Definition parameters
    D = early_energy / (total_energy + 1e-32)
    
    return D

# Analysis class for better organization
class AcousticAnalyzer:
    """
    A comprehensive acoustic analysis class for computing various acoustic parameters
    from room impulse responses.
    
    This class provides methods to analyze room impulse responses and compute
    standard acoustic parameters including RT60, clarity, definition, echo density,
    and energy decay curves.
    
    Attributes
    ----------
    fs : int
        Sampling rate in Hz used for all calculations.
    device : str
        Device ('cpu' or 'cuda') for PyTorch computations.
        
    Methods
    -------
    analyze_rir(rir)
        Perform comprehensive analysis of a room impulse response.
        
    Notes
    -----
    - The class automatically handles both PyTorch tensors and NumPy arrays
    """
    
    def __init__(self, fs: int = 48000, device: str = 'cpu'):
        """
        Initialize the acoustic analyzer.
        
        Parameters
        ----------
        fs : int
            Sampling rate in Hz
        device : str
            Device to use for PyTorch computations ('cpu' or 'cuda')
        """
        self.fs = fs
        self.device = device
    
    def analyze_rir(self, rir: Union[Tensor, NDArray]) -> dict:
        """
        Perform comprehensive analysis of a room impulse response.
        
        This method computes all standard acoustic parameters from a room impulse
        response, including energy decay curves, clarity, definition, echo density,
        and reverberation time.
        
        Parameters
        ----------
        rir : Union[Tensor, NDArray]
            Room impulse response to analyze. Can be 1D, 2D, or 3D.
            The method automatically reshapes to 3D format (batch, time, channels).
            
        Returns
        -------
        dict
            Dictionary containing all computed acoustic parameters:
            - 'edc': Energy Decay Curve in dB
            - 'edr': Energy Decay Relief in dB (time-frequency representation)
            - 'ned': Normalized Echo Density profile
            - 'c50': Clarity index at 50ms boundary in dB
            - 'c80': Clarity index at 80ms boundary in dB
            - 'd50': Definition parameter at 50ms boundary (ratio)
            - 'rt60': Reverberation time in seconds
        """
        # Ensure 3D shape (batch, time, channels)
        if rir.ndim == 1:
            rir = rir[None, :, None]  
        elif rir.ndim == 2:
            rir = rir[:, :, None] 
        results = {}
        
        # Convert to tensor if needed
        if isinstance(rir, NDArray):
            rir_tensor = torch.from_numpy(rir).to(self.device)
        else:
            rir_tensor = rir.to(self.device)
        
        # Compute EDC
        results['edc'] = compute_edc(rir_tensor, fs=self.fs)
        
        # Compute EDR
        results['edr'] = compute_edr(rir_tensor)
        
        # Compute normalized echo density
        results['ned'] = normalized_echo_density(rir_tensor, self.fs, use_local_avg=False)
        results['mixing_time'] = compute_mixing_time(results['ned'], self.fs)
        ## compute clarity index at 50ms and 80ms
        results['c50'], results['c80'] = compute_clarity_parameters(rir_tensor, self.fs)
        ## compute definition 
        results['d50'] = compute_definition_parameters(rir_tensor, self.fs)
        # Estimate RT60
        time_vector = torch.arange(results['edc'].shape[1], dtype=results['edc'].dtype, device=self.device) / self.fs
        rt60, *_ = estimate_rt60(results['edc'], time_vector)
        results['rt60'] = rt60
        
        return results
