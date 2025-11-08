import torch
import logging
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import sympy as sp
from numpy import ndarray as NDArray

from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.loss import mse_loss, sparsity_loss
from flamo.optimize.trainer import Trainer

import flareverb.sampling as sampling
from flareverb.analysis import AcousticAnalyzer
from flareverb.reverb import BaseFDN, GroupedFDN
from flareverb.utils import df2mat
from flareverb.config.config import (
    FDNConfig, 
    GFDNConfig,
    BaseConfig)

Tensor = torch.Tensor
NDArray = np.ndarray

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def append_params(fdn_data: Dict[str, list], params: Dict[str, Any]) -> Dict[str, list]:
    """
    Append FDN parameters to the data collection dictionary.
    
    This helper function adds FDN parameters (delays, onset time, onset gain,
    input gains, output gains, and feedback matrix) to the data collection
    dictionary for later analysis or storage.
    
    Parameters
    ----------
    fdn_data : Dict[str, list]
        Dictionary containing lists of FDN data. Each key corresponds to a
        parameter type (e.g., 'delays', 'input_gains', etc.).
    params : Dict[str, Any]
        Dictionary containing the current FDN parameters to append.
        Expected keys: 'delays', 'onset_time', 'early_reflections', 'input_gains',
        'output_gains', 'feedback_matrix'.
        
    Returns
    -------
    Dict[str, list]
        Updated data collection dictionary with the new parameters appended
        to their respective lists.
    """
    fdn_data["delays"].append(params["delays"])
    fdn_data["onset_time"].append(params["onset_time"])
    fdn_data["early_reflections"].append(params["early_reflections"])
    fdn_data["input_gains"].append(params["input_gains"])
    fdn_data["output_gains"].append(params["output_gains"])
    fdn_data["feedback_matrix"].append(params["feedback_matrix"])
    return fdn_data

def append_acoustic_params(fdn_data: Dict[str, list], acoustic_params: Dict[str, Any]) -> Dict[str, list]:
    """
    Append acoustic parameters to the data collection dictionary.
    
    This helper function adds acoustic analysis results (EDC, EDR, clarity,
    definition, and RT60 values) to the data collection dictionary for
    later analysis or storage.
    
    Parameters
    ----------
    fdn_data : Dict[str, list]
        Dictionary containing lists of FDN data. Each key corresponds to a
        parameter type (e.g., 'edc', 'rt60', etc.).
    acoustic_params : Dict[str, Any]
        Dictionary containing the current acoustic parameters to append.
        Expected keys: 'edc', 'edr', 'c50', 'c80', 'd50', 'rt60'.
        
    Returns
    -------
    Dict[str, list]
        Updated data collection dictionary with the new acoustic parameters
        appended to their respective lists.
    """
    fdn_data["edc"].append(acoustic_params["edc"])
    fdn_data["edr"].append(acoustic_params["edr"])
    fdn_data["c50"].append(acoustic_params["c50"])
    fdn_data["c80"].append(acoustic_params["c80"])
    fdn_data["d50"].append(acoustic_params["d50"])
    fdn_data["rt60"].append(acoustic_params["rt60"])
    return fdn_data

def sample_fdn_parameters(fdn_config: Union[FDNConfig, GFDNConfig], device: str, is_delay_shift: bool = False) -> Tuple[list, Tensor, Tensor, Tensor]:
    """
    Sample FDN parameters with optional random delay shift.
    
    This function generates FDN parameters using the sampling function and optionally
    applies a random prime delay shift to add variation to the delay lengths.
    The random shift helps create more diverse FDN configurations.
    
    Parameters
    ----------
    fdn_config : Union[FDNConfig, GFDNConfig]
        FDN configuration object containing parameters for parameter generation.
    device : str
        Device to create tensors on ('cpu' or 'cuda').
    is_delay_shift : bool, optional
        Whether to apply a random prime delay shift. If True, a random prime number
        between 50 and 500 is added to all delay lengths. Default is False.
        
    Returns
    -------
    Tuple[list, Tensor, Tensor, Tensor]
        A tuple containing:
        - delay_lengths: List of delay lengths in samples (potentially shifted)
        - input_gains: Input gains tensor
        - output_gains: Output gains tensor
        - feedback_matrix: Feedback matrix tensor
        
    Notes
    -----
    - The function first calls sampling.fdn_params to generate basic parameters
    - If is_delay_shift is True, a random prime number is chosen from the range [50, 500]
    - This prime number is added to all delay lengths to create variation
    - The shift helps avoid potential periodic artifacts in the reverberation
    - The shift is applied to all delay lines uniformly
    """
    delay_lengths, input_gains, output_gains, feedback_matrix = sampling.fdn_params(fdn_config, device=device)
    if is_delay_shift:
        prime_nums = list(sp.primerange(50, 500))
        shift = np.random.choice(prime_nums).item()
        delay_lengths = [d + shift for d in delay_lengths]
    return delay_lengths, input_gains, output_gains, feedback_matrix

def create_fdn(fdn_config: Union[FDNConfig, GFDNConfig], delay_lengths: list, input_gains: Tensor, output_gains: Tensor, feedback_matrix: Tensor, config: BaseFDN) -> Union[GroupedFDN, BaseFDN]:
    """
    Create and configure an FDN object (GroupedFDN or BaseFDN).
    
    This function creates an FDN object based on the configuration and assigns
    the provided parameters (delay lengths, gains, and feedback matrix) to it.
    The function automatically chooses between GroupedFDN and BaseFDN based on
    the number of groups in the configuration.
    
    Parameters
    ----------
    fdn_config : Union[FDNConfig, GFDNConfig]
        FDN configuration object containing parameters for FDN creation.
    delay_lengths : list
        List of delay lengths in samples for the FDN.
    input_gains : Tensor
        Input gains tensor for the FDN.
    output_gains : Tensor
        Output gains tensor for the FDN.
    feedback_matrix : Tensor
        Feedback matrix tensor for the FDN.
    config : BaseFDN
        Overall configuration object containing device and optimization settings.
        
    Returns
    -------
    Union[GroupedFDN, BaseFDN]
        Configured FDN object (either GroupedFDN or BaseFDN) with all parameters
        assigned and ready for use.
        
    Notes
    -----
    - If fdn_config.n_groups > 1, a GroupedFDN is created
    - If fdn_config.n_groups == 1, a BaseFDN is created
    - The output layer is set to 'freq_mag' if optimization is enabled, otherwise 'time'
    - All gains and feedback matrix are assigned to the FDN's core
    """
    if isinstance(fdn_config, GFDNConfig):
        curr_fdn = GroupedFDN(
            fdn_config,
            nfft=config.nfft,
            alias_decay_db=fdn_config.alias_decay_db,
            delay_lengths=delay_lengths,
            device=config.device,
            dtype=config.dtype,
            requires_grad=config.optimize,
            output_layer="freq_mag" if config.optimize else "time",
        )
        core = curr_fdn.shell.get_core()
        core.branchA.input_gain.assign_value(input_gains)
        core.branchA.output_gain.assign_value(output_gains)
    else:
        curr_fdn = BaseFDN(
            fdn_config,
            nfft=config.nfft,
            alias_decay_db=fdn_config.alias_decay_db,
            delay_lengths=delay_lengths,
            device=config.device,
            dtype=config.dtype,
            requires_grad=config.optimize,
            output_layer="freq_mag" if config.optimize else "time",
        )
        core = curr_fdn.shell.get_core()
        core.branchA.input_gain.assign_value(input_gains)
        core.branchA.output_gain.assign_value(output_gains)
        core.branchA.feedback_loop.feedback.mixing_matrix.assign_value(feedback_matrix)
    curr_fdn.shell.set_core(core)
    return curr_fdn

def normalize_fdn_energy(fdn_config: Union[FDNConfig, GFDNConfig], curr_fdn: Union[GroupedFDN, BaseFDN]) -> Union[GroupedFDN, BaseFDN]:
    """
    Normalize the energy of the FDN if specified in the configuration.
    
    This function checks if energy normalization is required based on the FDN
    configuration and applies it if needed. Energy normalization ensures that
    the FDN has a specific target energy level in its frequency response.
    
    Parameters
    ----------
    fdn_config : Union[FDNConfig, GFDNConfig]
        FDN configuration object. If fdn_config.energy is not None, normalization
        will be applied to achieve that target energy level.
    curr_fdn : Union[GroupedFDN, BaseFDN]
        The FDN object to potentially normalize.
        
    Returns
    -------
    Union[GroupedFDN, BaseFDN]
        The FDN object, either unchanged (if no normalization needed) or with
        normalized energy (if normalization was applied).
        
    Notes
    -----
    - The function only applies normalization if fdn_config.energy is not None
    - If normalization is applied, it uses the sampling.normalize_fdn_energy function
    - Energy normalization adjusts input and output gains to achieve the target energy
    - This preserves the FDN's reverberation characteristics while setting the desired energy level
    """
    if fdn_config.energy is not None:
        curr_fdn = sampling.normalize_fdn_energy(
            fdn_config, curr_fdn, fdn_config.energy
        )
    return curr_fdn

def analyze_fdn(curr_fdn: Union[FDNConfig, GFDNConfig], acoustic_analyzer: AcousticAnalyzer) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Get the time response and acoustic parameters for the FDN.
    
    This function extracts the impulse response from an FDN and analyzes it to
    compute various acoustic parameters such as RT60, clarity, definition,
    and echo density.
    
    Parameters
    ----------
    curr_fdn : Union[FDNConfig, GFDNConfig]
        The FDN object to analyze.
    acoustic_analyzer : AcousticAnalyzer
        Analyzer object configured with the appropriate sampling frequency.
        
    Returns
    -------
    Tuple[Tensor, Dict[str, Any]]
        A tuple containing:
        - h: The impulse response (time domain) from the FDN
        - acoustic_params: Dictionary containing acoustic analysis results
          with keys: 'edc', 'edr', 'c50', 'c80', 'd50', 'rt60'
        
    Notes
    -----
    - The function first extracts the time-domain impulse response from the FDN
    - Then it uses the acoustic analyzer to compute various acoustic parameters
    """
    h = curr_fdn.shell.get_time_response()
    acoustic_params = acoustic_analyzer.analyze_rir(h)
    return h, acoustic_params

def optimize_fdn(curr_fdn: Union[BaseFDN, GroupedFDN], config: BaseConfig) -> None:
    """
    Run optimization on the FDN using the provided configuration.
    
    This function sets up and executes an optimization process to improve the
    FDN's coloration. The optimization uses a dataset and training process
    with specified loss functions.
    
    Parameters
    ----------
    curr_fdn : Union[BaseFDN, GroupedFDN]
        The FDN object to optimize. Must have a shell attribute that can be
        used as a trainable model.
    config : Any
        Configuration object containing optimization parameters including:
        - fdn_optim_config: Optimization configuration (epochs, learning rate, etc.)
        - nfft: FFT size for frequency domain processing
        - device: Device to run optimization on
        
    Returns
    -------
    None
        The function modifies the FDN object in-place through the optimization process.
        
    Notes
    -----
    - The function sets up a trainer with MSE loss and sparsity loss criteria
    - The optimization process modifies the FDN's parameters to improve its properties
    - The FDN must be configured with requires_grad=True for optimization to work
    - The optimization typically aims to make the FDN's frequency response more flat
    """
    dataset = DatasetColorless(
        input_shape=(1, config.nfft // 2 + 1, 1),
        target_shape=(1, config.nfft // 2 + 1, 1),
        expand=config.fdn_optim_config.dataset_length,
        device=config.device,
        dtype=config.dtype,
    )
    train_loader, valid_loader = load_dataset(
        dataset, batch_size=config.fdn_optim_config.batch_size
    )

    # Replace the attenuation with identity during optimization
    temp = curr_fdn.shell.get_core() 
    curr_attenuation = temp.branchA.feedback_loop.feedback.attenuation
    temp.branchA.feedback_loop.feedback.attenuation = torch.nn.Identity()
    if hasattr(temp, 'branchB'):
        curr_er = temp.branchB.early_reflections
        temp.branchB.early_reflections = torch.nn.Identity()
    curr_fdn.shell.set_core(temp)

    trainer = Trainer(
        curr_fdn.shell,
        max_epochs=config.fdn_optim_config.max_epochs,
        lr=config.fdn_optim_config.lr,
        train_dir=config.fdn_optim_config.train_dir,
        device=config.device,
    )
    trainer.register_criterion(
        mse_loss(nfft=config.nfft, device=config.device), 1
    )
    trainer.register_criterion(sparsity_loss(), 1, requires_model=True)
    trainer.train(train_loader, valid_loader)

    # restore the attenuation
    temp = curr_fdn.shell.get_core()
    temp.branchA.feedback_loop.feedback.attenuation = curr_attenuation
    if hasattr(temp, 'branchB'):
        temp.branchB.early_reflections = curr_er
    curr_fdn.shell.set_core(temp)

def save_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to pickle and .mat files.
    
    This function saves a pandas DataFrame to both pickle format (for Python)
    and MATLAB .mat format (for cross-platform compatibility). It provides
    feedback about the saving process.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save. Should contain FDN parameters and acoustic
        analysis results.
    output_path : str
        Base path for saving the files. The function will create:
        - A pickle file at the specified path
        - A .mat file with the same name but .mat extension
        
    Returns
    -------
    None
        The function saves files to disk and prints confirmation messages.
        
    Notes
    -----
    - The pickle file preserves the exact DataFrame structure for Python use
    - The .mat file allows the data to be loaded in MATLAB
    - The function automatically generates the .mat filename by replacing .pkl with .mat
    """
    df.to_pickle(output_path)
    mat_file_path = output_path.replace(".pkl", ".mat")
    df2mat(df, mat_file_path)
    logger.info(f"FDN RIRs saved to {output_path}")

def _init_fdn_data(keys: list) -> Dict[str, list]:
    """
    Initialize a dictionary of lists for FDN data collection.
    
    This helper function creates a dictionary where each key corresponds to a
    data field, and each value is an empty list ready to collect data from
    multiple FDN instances.
    """
    return {key: [] for key in keys}

# --- Main Orchestration Function ---
def fdn_dataset(
    config: Any,
    save: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Generate a dataset of FDN impulse responses and acoustic parameters.
    This is the main function for generating a dataset of FDN impulse responses
    along with their corresponding acoustic parameters. The function can
    optionally perform colorless optimization on the FDNs and save the results to disk.

    Parameters
    ----------
    config : Any
        Configuration object containing all necessary parameters for FDN generation,
        including:
        - fdn_config: FDN configuration (delays, gains, etc.)
        - num: Number of FDN instances to generate
        - optimize: Whether to perform optimization
        - device: Device to run computations on
        - output_data_path: Path for saving results
    save : bool, optional
        Whether to save the results to disk. If True, saves both pickle and .mat files.
        Default is True.
    logger : Optional[logging.Logger], optional
        Logger object for recording progress and messages. If None, uses the default
        logger for this module. Default is None.
        
    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.DataFrame]]
        A tuple containing:
        - data_fdn: DataFrame with FDN parameters and acoustic features for all generated instances
        - data_fdn_optim: DataFrame with optimized FDN data (if optimization is enabled), else None
        
    Notes
    -----
    - The function generates the specified number of FDN instances
    - For each FDN, it computes the impulse response and analyzes acoustic parameters
    - If optimization is enabled, it runs colorless optimization and analyzes the optimized FDN
    - The function handles both grouped and non-grouped FDN configurations
    - Memory is managed by deleting FDN objects after processing
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    df_keys = [
        "delays", "onset_time", "early_reflections", "input_gains", "output_gains", "feedback_matrix",
        "attenuation", "ir", "fs", "edc", "edr", "c50", "c80", "d50", "rt60"
    ]
    fdn_config = config.fdn_config
    acoustic_analyzer = AcousticAnalyzer(
        fs=fdn_config.fs,
        device=config.device
    )
    fdn_data = _init_fdn_data(df_keys)
    fdn_optim_data = _init_fdn_data(df_keys) if config.optimize else None
    if config.optimize:
        Path(config.fdn_optim_config.train_dir).mkdir(parents=True, exist_ok=True)

    att_flag = 0
    for _ in range(config.num):
        # Sample parameters
        delay_lengths, input_gains, output_gains, feedback_matrix = sample_fdn_parameters(
            fdn_config, config.device, config.is_delay_shift
        )
        # Sample attenuation parameters 
        if fdn_config.attenuation_config.attenuation_param is None or att_flag == 1:
            att_flag = 1
            fdn_config.attenuation_config.attenuation_param = sampling.sample_attenuation_params(fdn_config, device=config.device)
        # Create FDN
        curr_fdn = create_fdn(
            fdn_config, delay_lengths, input_gains, output_gains, feedback_matrix, config
        )
        # Normalize energy if needed
        curr_fdn = normalize_fdn_energy(fdn_config, curr_fdn)
        # Analyze
        h, acoustic_params = analyze_fdn(curr_fdn, acoustic_analyzer)
        params = curr_fdn.get_params()
        # Append results
        fdn_data = append_acoustic_params(fdn_data, acoustic_params)
        fdn_data = append_params(fdn_data, params)
        fdn_data["attenuation"].append(params["attenuation"])
        fdn_data["ir"].append(h.squeeze().cpu().numpy().tolist())
        fdn_data["fs"].append(fdn_config.fs)
        # Optimization (if enabled)
        if config.optimize:
            optimize_fdn(curr_fdn, config)
            h_opt, acoustic_params_opt = analyze_fdn(curr_fdn, acoustic_analyzer)
            params_opt = curr_fdn.get_params()
            fdn_optim_data = append_acoustic_params(fdn_optim_data, acoustic_params_opt)
            fdn_optim_data = append_params(fdn_optim_data, params_opt)
            fdn_optim_data["attenuation"].append(params_opt["attenuation"][0])
            fdn_optim_data["ir"].append(h_opt.squeeze().cpu().numpy().tolist())
            fdn_optim_data["fs"].append(fdn_config.fs)
        del curr_fdn
    # Create DataFrames
    data_fdn = pd.DataFrame(fdn_data)
    data_fdn_optim = pd.DataFrame(fdn_optim_data) if config.optimize else None
    # Save if requested
    if save:
        save_results(data_fdn, config.output_data_path)
        if config.optimize and data_fdn_optim is not None:
            optim_path = config.output_data_path.replace(".pkl", "_optim.pkl")
            save_results(data_fdn_optim, optim_path)
    return data_fdn, data_fdn_optim

