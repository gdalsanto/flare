# Standard library imports
from pathlib import Path
import warnings

# Third-party imports
from typing import Union, Optional, List
import numpy as np
import sympy as sps
import torch
from pydantic import BaseModel, model_validator, Field

class FDNAttenuation(BaseModel):
    """
    Configuration for attenuation filters in FDN.
    """
    attenuation_type: str = Field(
        default="homogeneous",
        description="Type of attenuation filter. Types can be 'homogeneous', 'geq', or 'first_order_lp'."
    )
    attenuation_range: List[float] = Field(
        default_factory=lambda: [0.5, 3.5],
        description="Attenuation range in seconds (used only when attenuation_param is not given)."
    )
    rt_nyquist: float = Field(
        default=0.2,
        description="RT at Nyquist (for first order filter)."
    )
    attenuation_param: Optional[List[List[float]]] = Field(
        default=None,
        description="T60 parameter. The size depends on the attenuation_type: " \
        "'homogeneous' -> [num, 1]; " \
        "'geq' -> [num, num_bands]; " \
        "'first_order_lp' -> [num, 2]."
    )
    t60_octave_interval: int = Field(
        default=1,
        description="Octave interval for T60."
    )
    t60_center_freq: List[float] = Field(
        default_factory=lambda: [63, 125, 250, 500, 1000, 2000, 4000, 8000],
        description="Center frequencies for T60."
    )
    # TODO: Add support for tone corrector in the reverb module
    tc_param: Optional[List[List]] = Field(
        default=None,
        description="Tone corrector gains as 1 octave GEQ."
    )
    tc_octave_interval: int = Field(
        default=1,
        description="Octave interval for tone corrector."
    )
    tc_center_freq: List[float] = Field(
        default_factory=lambda: [63, 125, 250, 500, 1000, 2000, 4000, 8000],
        description="Center frequencies for tone corrector."
    )

    @model_validator(mode="after")
    def check_geq_parameters(self) -> "FDNAttenuation":
        """
        Validate that for 'geq' attenuation type, t60_center_freq length matches 
        the second dimension of attenuation_param when provided.
        """
        if (self.attenuation_type == "geq" and 
            self.attenuation_param is not None and 
            len(self.attenuation_param) > 0):
            
            # Get the number of frequency bands from attenuation_param
            num_bands = len(self.attenuation_param[0])
            
            if len(self.t60_center_freq) != num_bands:
                raise ValueError(
                    f"For 'geq' attenuation type, length of t60_center_freq "
                    f"({len(self.t60_center_freq)}) must match the number of frequency bands "
                    f"in attenuation_param ({num_bands})"
                )
        
        return self

class FDNMixing(BaseModel):
    """
    Mixing matrix configuration for FDN.
    """
    mixing_type: str = Field(
        default="orthogonal",
        description="Type of mixing matrix: 'orthogonal', 'householder', 'hadamard', or 'rotation'."
    )
    is_scattering: bool = Field(
        default=False,
        description="If filter feedback matrix is used."
    )
    is_velvet_noise: bool = Field(
        default=False,
        description="If velvet noise is used."
    )
    sparsity: int = Field(
        default=1,
        description="Density for scattering mapping."
    )
    n_stages: int = Field(
        default=3,
        description="Number of stages in the scattering mapping."
    )

    @model_validator(mode="after")
    def check_mixing_exclusivity(self) -> "FDNMixing":
        """
        Validate that is_scattering and is_velvet_noise are not both True.
        """
        if self.is_scattering and self.is_velvet_noise:
            raise ValueError("is_scattering and is_velvet_noise cannot both be True")
        return self

class FDNConfig(BaseModel):
    """
    FDN Configuration class.
    """
    fdn_type: str = Field(
        default="fdn",
        description="FDN type: 'fdn' or 'gfdn'."
    )
    in_ch: int = Field(
        default=1,
        description="Input channels."
    )
    out_ch: int = Field(
        default=1,
        description="Output channels."
    )
    fs: int = Field(
        default=48000,
        description="Sampling frequency."
    )
    N: int = Field(
        default=6,
        description="Number of delay lines."
    )
    delay_lengths: Optional[List[int]] = Field(
        default=None,
        description="Delay lengths in samples."
    )
    delay_range_ms: List[float] = Field(
        default_factory=lambda: [20.0, 50.0],
        description="Delay lengths range in ms."
    )
    delay_log_spacing: bool = Field(
        default=False,
        description="If delay lengths should be logarithmically spaced."
    )
    onset_time: List[float] = Field(
        default_factory=lambda: [10],
        description="Onset time in ms."
    )
    early_reflections_type: Optional[str] = Field(
        default=None,
        description="Type of early reflections: 'gain', 'FIR', or None."
    )
    drr: float = Field(
        default=0.25,
        description="Direct to reverberant ratio."
    )
    energy: Optional[float] = Field(
        default=None,
        description="Energy of the FDN."
    )
    gain_init: str = Field(
        default="randn",
        description="Gain initialization distribution."
    )
    attenuation_config: FDNAttenuation = Field(
        default_factory=FDNAttenuation,
        description="Attenuation configuration."
    )
    mixing_matrix_config: FDNMixing = Field(
        default_factory=FDNMixing,
        description="Mixing matrix configuration."
    )
    alias_decay_db: float = Field(
        default=0.0,
        description="Alias decay in dB."
    )

    @model_validator(mode="after")
    def check_delay_lengths(self) -> "BaseConfig":
        """
        Validate that delay_lengths length matches N when provided, and check onset_time vs delay_range_ms.
        """
        if self.delay_lengths is not None:
            if len(self.delay_lengths) != self.N:
                raise ValueError(
                    f"Length of delay_lengths ({len(self.delay_lengths)}) must match N ({self.N})"
                )
        if max(self.onset_time) > self.delay_range_ms[0]:
            warnings.warn(
                f"Max onset_time ({self.onset_time} ms) is larger than first element of delay_range_ms ({self.delay_range_ms[0]} ms)"
            )
        return self

    @model_validator(mode="after")
    def check_early_reflections(self) -> "FDNConfig":
        """
        Set drr to 0 when early_reflections_type is None.
        """
        if self.early_reflections_type is None:
            self.drr = 0.0
            print("Setting drr to 0.0 since early_reflections_type is None")
        return self

class GFDNConfig(FDNConfig):
    """
    Group FDN Configuration class, inheriting from FDNConfig.
    """
    n_groups: int = Field(
        default=2,
        description="Number of groups."
    )
    coupling_angles: List[float] = Field(
        default_factory=lambda: [0.0],
        description="Coupling angles in radians (should be N(N-1)/2)."
    )
    mixing_angles: List[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="Mixing angles."
    )
    input_gains_mask: Optional[List[float]] = Field(
        default=None,
        description="Input gains mask."
    )
    output_gains_mask: Optional[List[float]] = Field(
        default=None,
        description="Output gains mask."
    )

    @model_validator(mode="after")
    def check_delay_lengths(self) -> "BaseConfig":
        """
        Validate that delay_lengths length matches N x n_groups when provided, and check onset_time vs delay_range_ms.
        """
        if self.delay_lengths is not None:
            if len(self.delay_lengths) != self.n_groups * self.N:
                raise ValueError(
                    f"Length of delay_lengths ({len(self.delay_lengths)}) must match N x n_groups ({self.N} x {self.n_groups})"
                )
        if max(self.onset_time) > self.delay_range_ms[0]:
            warnings.warn(
                f"Max onset_time ({self.onset_time} ms) is larger than first element of delay_range_ms ({self.delay_range_ms[0]} ms)"
            )
        return self

    @model_validator(mode="after")
    def check_angles(self) -> "GFDNConfig":
        """
        Validate that coupling and mixing angles are within [0, pi/4] and clip if necessary.
        """
        if not (np.all(np.abs(self.coupling_angles) <= np.pi/4)):
            raise ValueError("Coupling angles must be in the range [0, pi/4]")
        if not (np.all(np.abs(self.mixing_angles) <= np.pi/4)):
            raise ValueError("Mixing angles must be in the range [0, pi/4]")
        self.coupling_angles = np.clip(np.abs(self.coupling_angles), 0, np.pi/4).tolist()
        self.mixing_angles = np.clip(np.abs(self.mixing_angles), 0, np.pi/4).tolist()
        return self

    @model_validator(mode="after")
    def check_masks(self) -> "GFDNConfig":
        """
        Ensure input and output gains masks are provided and have correct length.
        """
        if self.input_gains_mask is None:
            self.input_gains_mask = [1.0] * (self.n_groups * self.N)
        if self.output_gains_mask is None:
            self.output_gains_mask = [1.0] * (self.n_groups * self.N)
        if len(self.input_gains_mask) != self.n_groups * self.N:
            raise ValueError(
                f"Length of input_gains_mask ({len(self.input_gains_mask)}) must match N x n_groups ({self.N} x {self.n_groups})"
            )
        if len(self.output_gains_mask) != self.n_groups * self.N:
            raise ValueError(
                f"Length of output_gains_mask ({len(self.output_gains_mask)}) must match N x n_groups ({self.N} x {self.n_groups})"
            )
        return self

class FDNOptimConfig(BaseModel):
    """
    FDN Optimization Configuration class.
    """
    max_epochs: int = Field(
        default=10,
        description="Number of optimization iterations."
    )
    lr: float = Field(
        default=1e-3,
        description="Learning rate."
    )
    batch_size: int = Field(
        default=1,
        description="Batch size."
    )
    device: str = Field(
        default="cuda",
        description="Device to use for optimization."
    )
    dataset_length: int = Field(
        default=100,
        description="Dataset length."
    )
    train_dir: str = Field(
        default="output/fdn_optim",
        description="Training directory."
    )

class BaseConfig(BaseModel):
    """
    Base Configuration class for the overall system.
    """
    fs: int = Field(
        default=48000,
        description="Sampling frequency."
    )
    nfft: int = Field(
        default=96000,
        description="Number of FFT points."
    )
    num: int = Field(
        default=1000,
        description="Number of examples to generate."
    )
    fdn_config: Union[FDNConfig, GFDNConfig] = Field(
        default_factory=FDNConfig,
        description="FDN configuration."
    )
    optimize: bool = Field(
        default=False,
        description="Whether to optimize for colorlessness."
    )
    fdn_optim_config: FDNOptimConfig = Field(
        default_factory=FDNOptimConfig,
        description="Optimization configuration."
    )
    output_data_path: str = Field(
        default="output/fdn_rirs.pkl",
        description="Output data path."
    )
    device: str = Field(
        default="cuda",
        description="Device to use."
    )
    dtype: str = Field(
        default="float32",
        description="Data type."
    )
    is_delay_shift: bool = Field(
        default=False,
        description="Whether to apply random delay shift."
    )

    @model_validator(mode="after")
    def validate_config(self) -> "BaseConfig":
        """
        Ensure the output directory exists, validate FDN config, and check device availability.
        """
        output_path = Path(self.output_data_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate FDN configuration
        if self.fdn_config.fs != self.fs:
            raise ValueError("Sampling frequency in fdn_config must match fs")

        # Validate device availability
        original_device = self.device
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                warnings.warn(f"CUDA not available, switching from '{original_device}' to 'cpu'")
                self.device = "cpu"
            elif self.device != "cuda":  # specific cuda device like "cuda:0"
                try:
                    device_id = int(self.device.split(":")[1])
                    if device_id >= torch.cuda.device_count():
                        warnings.warn(f"CUDA device {device_id} not available, switching to 'cuda:0'")
                        self.device = "cuda:0"
                except (IndexError, ValueError):
                    warnings.warn(f"Invalid device format '{original_device}', switching to 'cuda'")
                    self.device = "cuda"
        elif self.device == "mps":
            if not torch.backends.mps.is_available():
                warnings.warn(f"MPS not available, switching from '{original_device}' to 'cpu'")
                self.device = "cpu"

        # Sync device with optimization config
        self.fdn_optim_config.device = self.device

        # convert dtype string to torch dtype
        dtype_mapping = {
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if self.dtype in dtype_mapping:
            self.dtype = dtype_mapping[self.dtype]
        else:
            raise ValueError(f"Unsupported dtype '{self.dtype}'. Supported dtypes are: {list(dtype_mapping.keys())}")
        return self