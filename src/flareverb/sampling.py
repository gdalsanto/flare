from typing import Union
import json
from pathlib import Path

import torch
import sympy as sp
import numpy as np
from numpy.typing import ArrayLike

from flareverb.config.config import (
    FDNConfig, 
    GFDNConfig, 
    FDNAttenuation)
from flareverb.utils import ms_to_samps
from flareverb.reverb import BaseFDN
from flareverb.utils import rt_from_sabine
def load_material_coefficients(filename: str = "pyra_materials.json"):
    """Load material absorption coefficients from JSON file."""
    materials_file = Path(__file__).parent / "data" / filename

    try:
        with open(materials_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load materials file: {e}")


def fdn_params(config: Union[FDNConfig, GFDNConfig], device: str):
    """
    Generate parameters for a Feedback Delay Network (FDN).
    
    This function creates the essential parameters needed to initialize an FDN:
    delay line lengths and gain parameters (input gains, output gains, and feedback matrix).
    The delay lengths can be either randomly chosen prime numbers within a specified range
    or provided explicitly in the configuration.
    
    Parameters
    ----------
    config : FDNConfig
        Configuration object containing FDN parameters including:
        - N: Number of delay lines per group
        - n_groups: Number of groups (for grouped FDN)
        - delay_range_ms: Range for delay lengths in milliseconds
        - delay_log_spacing: Whether to use logarithmic spacing for delays
        - delay_lengths: Explicit delay lengths (if provided)
        - gain_init: Initialization method for gains ('randn' or 'uniform')
        - mixing_matrix_config: Configuration for the mixing matrix
    device : str
        Device to create tensors on ('cpu' or 'cuda').
        
    Returns
    -------
    tuple
        A tuple containing:
        - delay_lengths: List of delay lengths in samples
        - b: Input gains tensor of shape (N, 1)
        - c: Output gains tensor of shape (1, N)
        - U: Feedback matrix tensor of shape (N, N) or (n_stages, N, N) for scattering
        
    Notes
    -----
    - Delay lengths are chosen as prime numbers to avoid periodic artifacts
    - If delay_log_spacing is True, delays are logarithmically spaced within the range
    - The total number of delay lines is N * n_groups
    - Gain parameters are initialized randomly using the specified distribution
    - For scattering matrices, U has shape (n_stages, N, N) where n_stages is the number
      of scattering stages
        
    Examples
    --------
    >>> from flareverb.config.config import FDNConfig
    >>> config = FDNConfig(N=4, n_groups=1, delay_range_ms=[20, 50])
    >>> delays, b, c, U = fdn_params(config, 'cpu')
    >>> print(f"Delay lengths: {delays}")
    >>> print(f"Input gains shape: {b.shape}")
    >>> print(f"Output gains shape: {c.shape}")
    >>> print(f"Feedback matrix shape: {U.shape}")
    """
    if isinstance(config, GFDNConfig):
        N = config.N * config.n_groups
    else:
        N = config.N

    n_groups = N // config.N
    delay_lengths = []
    if config.delay_lengths is None:
        for i_group in range(n_groups):
            delay_range_samps = ms_to_samps(np.asarray(config.delay_range_ms), config.fs)
            # generate prime numbers in specified range - add some randomness
            prime_nums = np.array(
                list(sp.primerange(delay_range_samps[0]*(1 + np.random.rand()/10), delay_range_samps[1]*(1 + np.random.rand()/10))),
                dtype=np.int32,
            )

            if config.delay_log_spacing: 
                # find N prime numbers in the range which are logarithmically spaced
                log_samps = np.logspace(
                    np.log10(delay_range_samps[0]), np.log10(delay_range_samps[1] - 1), np.round(N/n_groups).astype(int), dtype=int
                )
                # find the prime numbers which are closest to the logarithmically spaced samples
                curr_delay_lengths = prime_nums[np.searchsorted(prime_nums, log_samps)].tolist()
                # check if there are repeated prime numbers
                if len(set(curr_delay_lengths)) < N/n_groups:
                    rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
                    # delay line lengths
                    curr_delay_lengths = np.array(
                        np.r_[rand_primes[:N/n_groups - 1], sp.nextprime(delay_range_samps[1])],
                        dtype=np.int32,
                    ).tolist() 
            else:
                rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
                # delay line lengths
                curr_delay_lengths = np.array(
                    np.r_[rand_primes[:int(N/n_groups - 1)], sp.nextprime(delay_range_samps[1])],
                    dtype=np.int32,
                ).tolist()

            delay_lengths.extend(curr_delay_lengths)    
    else:
        delay_lengths = config.delay_lengths
    # random sampling of the gain parameters
    if config.mixing_matrix_config.is_scattering or config.mixing_matrix_config.is_velvet_noise:
        U_dims = (config.mixing_matrix_config.n_stages, N, N)
    else:
        U_dims = (N, N)
    if config.gain_init == "randn":
        b = torch.randn(size=(N, 1), device=device)
        c = torch.randn(size=(1, N), device=device)
        U = torch.randn(size=U_dims, device=device)
    elif config.gain_init == "uniform":
        b = torch.rand(size=(N, 1), device=device)
        c = torch.rand(size=(1, N), device=device)
        U = torch.rand(size=U_dims, device=device)        
    else: 
        raise ValueError("Distribution not recognized")

    return delay_lengths, b, c, U



def normalize_fdn_energy(config: FDNConfig, fdn: BaseFDN, target_energy: Union[float, ArrayLike, torch.Tensor]):
    """
    Normalize the energy of a Feedback Delay Network to match a target energy level.

    This function adjusts the input and output gains of an FDN, and the early reflections
    to achieve a specific target energy level in the frequency response. The normalization is performed
    by scaling the gains proportionally to maintain the FDN's characteristics while
    achieving the desired energy level. The combination of the fdn and the direct path
    is assumed to be uncorrelated.

    Parameters
    ----------
    config : FDNConfig
        Configuration object containing FDN parameters, including the direct-to-reverberant
        ratio (drr) which is used in the normalization.
    fdn : BaseFDN
        The FDN object whose energy is to be normalized.
    target_energy : Union[float, ArrayLike, torch.Tensor]
        The target energy level to achieve. Can be a scalar or tensor.
        
    Returns
    -------
    BaseFDN
        The modified FDN object with normalized energy. The input and output gains
        have been adjusted to achieve the target energy level.
        
    Notes
    -----
    - The function calculates the current energy from the FDN's frequency response
    - Energy normalization is performed by scaling both input and output gains
    - The scaling factor is calculated as: (target_energy / current_energy)^(1/4)
    - The direct path gain is also adjusted using the drr parameter from config
    - This normalization preserves the FDN's reverberation characteristics while
      achieving the desired overall energy level
        
    Examples
    --------
    >>> from flareverb.config.config import FDNConfig
    >>> from flareverb.reverb import BaseFDN
    >>> config = FDNConfig(drr=0.25)
    >>> fdn = BaseFDN(config, nfft=8192, alias_decay_db=0.0, delay_lengths=[100, 200, 300])
    >>> # Normalize to target energy of 1.0
    >>> normalized_fdn = normalize_fdn_energy(config, fdn, target_energy=1.0)
    """
    if not isinstance(target_energy, torch.Tensor):
        target_energy = torch.tensor(target_energy, device=fdn.device)
    core = fdn.shell.get_core()
    H = fdn.shell.get_freq_response()
    energy = torch.sum(torch.pow(torch.abs(H), 2)) / torch.tensor(H.size(1), device=fdn.device) 
    energy_fdn = target_energy / (1 + config.drr)
    curr_energy_direct = torch.sum(torch.pow(core.branchB.early_reflections.map(core.branchB.early_reflections.param), 2)) 
    if config.drr > 0:
        energy_direct = target_energy * (config.drr / (1 + config.drr))
        er = core.branchB.early_reflections.param
        # change the energy of the direct path is it's nonzero
        core.branchB.early_reflections.assign_value(
            er / torch.pow((curr_energy_direct), 1 / 2) * torch.pow(energy_direct, 1 / 2)
        )
    else: 
        energy_direct = 0.0
    # energy normalization
    core = fdn.shell.get_core()
    # get input and output gains
    b = core.branchA.input_gain.param
    c = core.branchA.output_gain.param
    # assign new gains to the FDN
    core.branchA.input_gain.assign_value(
        b / torch.pow((energy - curr_energy_direct), 1 / 4) * torch.pow(energy_fdn, 1 / 4)
    )
    core.branchA.output_gain.assign_value(
        c / torch.pow((energy - curr_energy_direct), 1 / 4) * torch.pow(energy_fdn, 1 / 4)
    )
    fdn.shell.set_core(core)
    return fdn

# set the limits of the room dimensions
room_dimentions = {
    "small": {  "length": [1.5, 7], "width": [1.5, 5],  "height": [2, 3]},
    "medium": { "length": [5, 10],  "width": [3, 8],    "height": [2, 5]},
    "large": {  "length": [7, 15],  "width": [5, 12],   "height": [3, 7]}
}

def sample_reverb_time(config: FDNAttenuation, device: str ):
    """
    Generation of the reverberation time curve based on Sabine's formula.
    This function generates the parameters randomly and uses the frequency dependent attenuation by air.

    Returns
    -------
    Reference: Prawda, K., Schlecht, S. J., & Välimäki, V. (2022). Calibrating the Sabine and Eyring formulas. The Journal of the Acoustical Society of America, 152(2), 1158-1169.
    """
    size = np.random.choice(list(room_dimentions.keys()))
    room_l = np.random.uniform(room_dimentions[size]['length'][0], room_dimentions[size]['length'][1])
    room_w = np.random.uniform(room_dimentions[size]['width'][0], room_dimentions[size]['width'][1])
    room_h = np.random.uniform(room_dimentions[size]['height'][0], room_dimentions[size]['height'][1])
    room_dim = [room_l, room_w, room_h]


    # Load the materials when the module is imported
    pyra_materials = load_material_coefficients()

    # sample three materials from the pyra_materials.json
    materials = np.random.choice(
        list(pyra_materials['absorption'].keys()), size=3, replace=True
    )
    source_freqs = np.array(pyra_materials['center_freqs'])
    target_freqs = np.array(config.t60_center_freq)
    if config.attenuation_type == "geq":
        target_freqs = np.concatenate((np.array([10]), target_freqs))
    # interpolate the absorption coefficients to the frequency bands given in the config file
    if not np.array_equal(target_freqs, source_freqs):
        interpolated_coeffs = []
        for mat in materials:
            source_coeffs = np.array(pyra_materials['absorption'][mat]['coeffs'])
            # Perform linear interpolation
            interpolated_coeffs.append(torch.tensor(np.interp(target_freqs, source_freqs, source_coeffs), device=device))

        absorption_coeffs = torch.stack(interpolated_coeffs)
    else:
        absorption_coeffs = torch.tensor(
            [pyra_materials['absorption'][mat]['coeffs'] for mat in materials],
            device=device,
        )
    # convert the coefficients to a torch tensor
    surface = torch.empty((3), dtype=torch.float32)
    surface[0] = 2 * (room_l * room_h + room_w * room_h)  # walls
    surface[1] = (room_l * room_w)  # floor
    surface[2] = (room_l * room_w)   # ceiling
    # compute the reverberation time for each material
    return rt_from_sabine(surface, absorption_coeffs, torch.tensor(room_dim, device=device))

def sample_attenuation_params(config: Union[FDNConfig, GFDNConfig], device: str = "cpu"):

    if isinstance(config, GFDNConfig):
        att_params = []
        for i_group in range(config.n_groups):
            rt = sample_reverb_time(config.attenuation_config, device=device)
            if config.attenuation_config.attenuation_type == "geq":
                att_params.append(rt.tolist())
            elif config.attenuation_config.attenuation_type == "first_order_lp": 
                # generate random cutoff frequency 
                cutoff_freqs = np.random.uniform(np.pi / 10, np.pi)
                # use one random value for rt for the rt at low frequencies
                rt_low_freq = np.random.uniform(rt[0], rt[-1])
                # generate the first order lowpass filter coefficients
                att_params.append([rt_low_freq, cutoff_freqs])
            elif config.attenuation_config.attenuation_type == "homogeneous":
                att_params.append(config.attenuation_config.attenuation_param)
            else:
                raise ValueError(f"Unsupported attenuation type: {config.attenuation_config.attenuation_type}")
        config.attenuation_config.attenuation_param = att_params
    else:
        rt = sample_reverb_time(config.attenuation_config, device=device)
        # set the attenuation parameters to the reverberation time
        if config.attenuation_config.attenuation_type == "geq":
            config.attenuation_config.attenuation_param = [rt.tolist()]
        elif config.attenuation_config.attenuation_type == "first_order_lp": 
            # generate random cutoff frequency 
            cutoff_freqs = np.random.uniform(np.pi / 10, np.pi)
            # use one random value for rt for the rt at low frequencies
            rt_low_freq = np.random.uniform(rt[0], rt[-1])
            # generate the first order lowpass filter coefficients
            config.attenuation_config.attenuation_param = [[rt_low_freq, cutoff_freqs]]
        elif config.attenuation_config.attenuation_type == "homogeneous":
            return config.attenuation_config.attenuation_param
        else:
            raise ValueError(f"Unsupported attenuation type: {config.attenuation_config.attenuation_type}")

    return config.attenuation_config.attenuation_param

