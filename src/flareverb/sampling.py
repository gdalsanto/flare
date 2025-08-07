from typing import Union

import torch
import sympy as sp
import numpy as np
from numpy.typing import ArrayLike

from flareverb.config.config import FDNConfig, GFDNConfig
from flareverb.utils import ms_to_samps
from flareverb.reverb import BaseFDN


def fdn_params(config: FDNConfig, device: str):
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

    N = config.N * config.n_groups

    
    delay_lengths = []
    if config.delay_lengths is None:
        for i_group in range(config.n_groups):
            delay_range_samps = ms_to_samps(np.asarray(config.delay_range_ms), config.fs)
            # generate prime numbers in specified range - add some randomness
            prime_nums = np.array(
                list(sp.primerange(delay_range_samps[0]*(1 + np.random.rand()/10), delay_range_samps[1]*(1 + np.random.rand()/10))),
                dtype=np.int32,
            )

            if config.delay_log_spacing: 
                # find N prime numbers in the range which are logarithmically spaced
                log_samps = np.logspace(
                    np.log10(delay_range_samps[0]), np.log10(delay_range_samps[1] - 1), np.round(N/config.n_groups).astype(int), dtype=int
                )
                # find the prime numbers which are closest to the logarithmically spaced samples
                curr_delay_lengths = prime_nums[np.searchsorted(prime_nums, log_samps)].tolist()
                # check if there are repeated prime numbers
                if len(set(curr_delay_lengths)) < N/config.n_groups:
                    rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
                    # delay line lengths
                    curr_delay_lengths = np.array(
                        np.r_[rand_primes[:N/config.n_groups - 1], sp.nextprime(delay_range_samps[1])],
                        dtype=np.int32,
                    ).tolist() 
            else:
                rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
                # delay line lengths
                curr_delay_lengths = np.array(
                    np.r_[rand_primes[:int(N/config.n_groups - 1)], sp.nextprime(delay_range_samps[1])],
                    dtype=np.int32,
                ).tolist()

            delay_lengths.extend(curr_delay_lengths)    
    else:
        delay_lengths = config.delay_lengths
    # random sampling of the gain parameters
    if config.mixing_matrix_config.is_scattering:
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
    
    This function adjusts the input and output gains of an FDN to achieve a specific
    target energy level in the frequency response. The normalization is performed
    by scaling the gains proportionally to maintain the FDN's characteristics while
    achieving the desired energy level.
    
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
    H = fdn.shell.get_freq_response()
    energy = torch.sum(torch.pow(torch.abs(H), 2)) / torch.tensor(H.size(1), device=fdn.device)
    # energy normalization
    core = fdn.shell.get_core()
    # get input and output gains
    b = core.branchA.input_gain.param
    c = core.branchA.output_gain.param
    # assign new gains to the FDN
    core.branchA.input_gain.assign_value(
        b / torch.pow(energy, 1 / 4) * torch.pow(target_energy, 1 / 4)
    )
    core.branchA.output_gain.assign_value(
        c / torch.pow(energy, 1 / 4) * torch.pow(target_energy, 1 / 4)
    )
    core.branchB.direct.assign_value(
        config.drr * torch.pow(target_energy, 1 / 2).unsqueeze(0).unsqueeze(1)
    )
    fdn.shell.set_core(core)
    return fdn
