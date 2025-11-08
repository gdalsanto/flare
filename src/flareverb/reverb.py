from collections import OrderedDict
from typing import List, Literal, Optional, Dict, Any, Union

import torch
from torch import nn
from flamo import dsp, system
from flamo.auxiliary.reverb import (
    parallelFDNAccurateGEQ, 
    parallelFirstOrderShelving, 
    parallelGFDNAccurateGEQ
)
from flamo.functional import signal_gallery

from flareverb.config.config import (
    BaseConfig,
    FDNAttenuation,
    FDNMixing,
    GFDNConfig,
    FDNConfig,
)

from flareverb.utils import ms_to_samps, rt2slope


class MapGamma(torch.nn.Module):
    """
    Gamma mapping module for attenuation calculations in FDN reverberation.

    This module maps input values to gamma coefficients used in attenuation
    calculations for Feedback Delay Networks. It supports both direct mapping
    and sigmoid-compressed mapping for bounded output ranges.

    Attributes
    ----------
    delays : torch.Tensor
        Delay lengths tensor used for gamma calculation.
    is_compressed : bool
        Whether to apply sigmoid compression to the output.
    g_min : float
        Minimum gamma value (0.99).
    g_max : float
        Maximum gamma value (1.0).

    Notes
    -----
    - Gamma coefficients control the decay rate of each delay line
    - The compressed mode uses sigmoid activation to ensure bounded output
    - The output is raised to the power of delay lengths for homogeneous control
      over all delay lines
    - This module is typically used as part of the attenuation system in FDNs
    """

    def __init__(self, delays: torch.Tensor, is_compressed: bool = False):
        """
        Initialize the gamma mapping module.

        Parameters
        ----------
        delays : torch.Tensor
            Delay lengths tensor. Used to compute gamma coefficients for each delay line.
            The shape should match the number of delay lines in the FDN.
        is_compressed : bool, optional
            Whether to apply sigmoid compression to the output. If True, the output
            is constrained to the range [g_min, g_max] using sigmoid activation.
            Default is False.
        """
        super().__init__()
        self.delays = delays
        self.is_compressed = is_compressed
        self.g_min = 0.99
        self.g_max = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the gamma mapping.

        Maps input values to gamma coefficients for attenuation calculations.
        The output is raised to the power of delay lengths to provide per-delay
        control over the attenuation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the raw gamma values. Expected to have shape
            compatible with the stored delays tensor.

        Returns
        -------
        torch.Tensor
            Gamma coefficients tensor. If is_compressed is True, the output is
            constrained to the range [g_min, g_max] and raised to the power of delays.
            Otherwise, the input is directly raised to the power of delays.
        """
        if self.is_compressed:
            return (
                torch.sigmoid(x[0]) * (self.g_max - self.g_min) + self.g_min
            ) ** self.delays
        else:
            return x[0] ** self.delays



class BaseFDN(nn.Module):
    """
    Base Feedback Delay Network (FDN) implementation for reverberation synthesis.

    A modular FDN system that supports various attenuation types and onset
    configurations for generating room impulse responses. The FDN
    consists of multiple delay lines connected through a mixing matrix with
    configurable attenuation and onset characteristics.

    Notes
    -----
    - The FDN uses a modular architecture with separate components for delays,
      mixing, attenuation, and onset
    - Supports both homogeneous and frequency-dependent attenuation
    - Can be configured for optimization with requires_grad=True
    - Provides multiple output formats: time domain, frequency magnitude, or complex frequency
    - The system is designed to work with the flamo framework for efficient processing
    """

    def __init__(
        self,
        config: FDNConfig,
        nfft: int,
        alias_decay_db: float,
        delay_lengths: List[int],
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = True,
        output_layer: Literal["freq_complex", "freq_mag", "time"] = "time",
    ) -> None:
        """
        Initialize the BaseFDN.

        Parameters
        ----------
        config : FDNConfig
            Configuration object containing FDN parameters including attenuation,
            mixing matrix, and onset settings.
        nfft : int
            FFT size for frequency domain processing. Must be large enough to
            accommodate the longest delay line without aliasing.
        alias_decay_db : float
            Alias decay in decibels. Controls the decay rate of aliased components
            in frequency domain processing.
        delay_lengths : List[int]
            List of delay lengths in samples for each delay line. Should be prime
            numbers to avoid periodic artifacts.
        device : Literal["cpu", "cuda"], optional
            Device to run computations on. Default is 'cuda'.
        dtype : torch.dtype, optional
            Data type for tensors. Default is torch.float32.
        requires_grad : bool, optional
            Whether parameters should be learnable for optimization. Default is True.
        output_layer : Literal["freq_complex", "freq_mag", "time"], optional
            Type of output layer to use:
            - 'time': Time domain output (default)
            - 'freq_mag': Frequency magnitude output
            - 'freq_complex': Complex frequency output
            Default is "time".

        Notes
        -----
        - The initialization process validates delays, sets up parameters, and
          configures the FDN system
        - Learnable parameters enable colorless optimization of the gains and
          feedback matrix
        """
        super().__init__()

        self._validate_delays(config, delay_lengths)
        self._initialize_parameters(
            config, nfft, alias_decay_db, delay_lengths, device, dtype, requires_grad
        )
        self._setup_fdn_system(config, output_layer)

    def forward(
        self,
        inputs: torch.Tensor,
        ext_params: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Forward pass through the FDN.

        Processes input signals through the Feedback Delay Network to generate
        reverberated output. Each input can have its own set of external parameters
        for dynamic control of the FDN characteristics.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, signal_length).
        ext_params : List[Dict[str, Any]]
            List of external parameters for each input signal. Each dictionary
            can contain parameters to modify the FDN behavior during processing.
            The length must match the batch size.

        Returns
        -------
        torch.Tensor
            Processed output tensor. Contains the reverberated signals.
        """
        outputs = []
        for x, ext_param in zip(inputs, ext_params):
            # Apply the FDN with the external parameters
            output = self.shell(x[..., None], ext_param)
            outputs.append(output)

        return torch.stack(outputs).squeeze(-1)

    def get_params(self) -> OrderedDict[str, Any]:
        """
        Get the current parameters of the FDN.

        Extracts all learnable and configurable parameters from the FDN system
        for analysis, storage, or parameter transfer. All parameters are converted
        to CPU NumPy arrays for compatibility.

        Returns
        -------
        OrderedDict[str, Any]
            Dictionary containing all FDN parameters:
            - 'delays': List of delay lengths in samples
            - 'onset_time': List of onset times in milliseconds
            - 'early_reflections': Direct path gain values
            - 'input_gains': Input gain coefficients for each delay line
            - 'output_gains': Output gain coefficients for each delay line
            - 'feedback_matrix': Mixing (feedback) matrix coefficients
            - 'attenuation': Attenuation coefficients for each delay line

        Notes
        -----
        - All parameters are detached from the computation graph and moved to CPU
        - The returned parameters can be used to recreate or modify the FDN
        """
        core = self.shell.get_core()

        params = OrderedDict()
        params["delays"] = self.delay_lengths.cpu().numpy().tolist()
        params["onset_time"] = self.onset
        params["early_reflections"] = (
            core.branchB.early_reflections.param.cpu().detach().numpy().tolist()
        )
        params["input_gains"] = (
            core.branchA.input_gain.param.cpu().detach().numpy().tolist()
        )
        params["output_gains"] = (
            core.branchA.output_gain.param[0].cpu().detach().numpy().tolist()
        )
        params["feedback_matrix"] = (
            core.branchA.feedback_loop.feedback.mixing_matrix.param.cpu()
            .detach()
            .numpy()
            .tolist()
        )
        params["attenuation"] = (
            core.branchA.feedback_loop.feedback.attenuation.param.cpu()
            .detach()
            .numpy()
            .tolist()
        )
        return params

    def _validate_delays(self, config: BaseConfig, delay_lengths: List[int]) -> None:
        """Validate delay lengths."""
        if config.N != len(delay_lengths):
            raise ValueError(
                f"N ({config.N}) must match the length of delay_lengths ({len(delay_lengths)})"
            )

    def _initialize_parameters(
        self,
        config: FDNConfig,
        nfft: int,
        alias_decay_db: float,
        delay_lengths: List[int],
        device: str,
        dtype: torch.dtype,
        requires_grad: bool,
    ) -> None:
        """Initialize FDN parameters."""
        self.device = torch.device(device)
        self.dtype = dtype
        # Core FDN parameters
        self.N = config.N
        self.fs = config.fs
        self.nfft = nfft
        self.alias_decay_db = alias_decay_db
        self.requires_grad = requires_grad

        # Onset configuration
        self.early_reflections_type = config.early_reflections_type
        self.onset = ms_to_samps(torch.tensor(config.onset_time), config.fs)

        # Channel configuration
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Delay configuration
        self.delay_lengths = torch.tensor(
            delay_lengths, device=self.device, dtype=torch.int64
        )

    def _setup_fdn_system(self, config: BaseConfig, output_layer: str) -> None:
        """Setup the complete FDN system."""
        # Create FDN branches
        branch_a = self._create_fdn_branch(
            config.attenuation_config, config.mixing_matrix_config
        )
        branch_b = self._create_direct_path(config)

        # Combine branches
        fdn_core = system.Parallel(brA=branch_a, brB=branch_b, sum_output=True)

        # Setup I/O layers
        input_layer = dsp.FFT(self.nfft, dtype=self.dtype)
        output_layer = self._create_output_layer(output_layer)

        # Create shell
        self.shell = system.Shell(
            core=fdn_core,
            input_layer=input_layer,
            output_layer=output_layer,
        )

    def _create_output_layer(self, output_type: str):
        """Create the appropriate output layer based on type."""
        if output_type == "time":
            return dsp.iFFTAntiAlias(self.nfft, self.alias_decay_db, dtype=self.dtype)
        elif output_type == "freq_complex":
            return dsp.Transform(transform=lambda x: x, dtype=self.dtype)
        elif output_type == "freq_mag":
            return dsp.Transform(transform=lambda x: torch.abs(x), dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported output layer type: {output_type}")

    def _create_fdn_branch(
        self, attenuation_config: FDNAttenuation, mixing_matrix_config: FDNMixing
    ):
        """Create the main FDN branch (branch A)."""
        # Input and output gains
        input_gain = dsp.Gain(
            size=(self.N, self.in_ch),
            nfft=self.nfft,
            requires_grad=self.requires_grad,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )

        output_gain = dsp.Gain(
            size=(self.out_ch, self.N),
            nfft=self.nfft,
            requires_grad=self.requires_grad,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )

        # Feedback loop components
        delays = self._create_delay_lines()
        mixing_matrix = self._create_mixing_matrix(mixing_matrix_config)
        attenuation = self._create_attenuation(attenuation_config)

        # Feedback path
        feedback = system.Series(
            OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
        )

        # Recursion
        feedback_loop = system.Recursion(fF=delays, fB=feedback)

        # Complete FDN branch
        return system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )

    def _create_delay_lines(self):
        """Create parallel delay lines."""
        delays = dsp.parallelDelay(
            size=(self.N,),
            max_len=self.delay_lengths.max(),
            nfft=self.nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )
        delays.assign_value(delays.sample2s(self.delay_lengths))
        return delays

    def _create_mixing_matrix(self, config: FDNMixing):
        """Create orthogonal mixing matrix."""
        if config.is_scattering or config.is_velvet_noise:
            m_L = torch.randint(
                low=1,
                high=int(torch.floor(min(self.delay_lengths) / 10)),
                size=[self.N],
            )
            m_R = torch.randint(
                low=1,
                high=int(torch.floor(min(self.delay_lengths) / 10)),
                size=[self.N],
            )
            if config.is_scattering:
                mixing = dsp.ScatteringMatrix(
                    size=(config.n_stages, self.N, self.N),
                    nfft=self.nfft,
                    sparsity=config.sparsity,
                    gain_per_sample=1.0,
                    m_L=m_L,
                    m_R=m_R,
                    requires_grad=self.requires_grad,
                    alias_decay_db=self.alias_decay_db,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                mixing = dsp.VelvetNoiseMatrix(
                    size=(config.n_stages, self.N, self.N),
                    nfft=self.nfft,
                    density=1 / config.sparsity,
                    gain_per_sample=1.0,
                    m_L=m_L,
                    m_R=m_R,
                    alias_decay_db=self.alias_decay_db,
                    device=self.device,
                    dtype=self.dtype,
                )
        elif config.mixing_type == "householder":
            mixing = dsp.HouseholderMatrix(
                size=(self.N, self.N),
                nfft=self.nfft,
                requires_grad=self.requires_grad,
                alias_decay_db=self.alias_decay_db,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            try:
                mixing = dsp.Matrix(
                    size=(self.N, self.N),
                    nfft=self.nfft,
                    matrix_type=config.mixing_type,
                    requires_grad=self.requires_grad,
                    alias_decay_db=self.alias_decay_db,
                    device=self.device,
                    dtype=self.dtype,
                )  # TODO add hadamard, tiny rotation
            except:
                raise ValueError(f"Unsupported mixing type: {config.mixing_type}")
        return mixing

    def _create_direct_path(self, config: BaseConfig):
        """Create the direct path branch (branch B)."""
        onset_delay = dsp.parallelDelay(
            size=(self.in_ch,),
            max_len=self.onset,
            nfft=self.nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )

        if config.early_reflections_type == "FIR":
            L = self.delay_lengths.min()
            early_reflections = dsp.parallelFilter(
                size=(L-self.onset, self.in_ch),
                nfft=self.nfft,
                requires_grad=False,
                map=lambda x: x,
                alias_decay_db=self.alias_decay_db,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            early_reflections = dsp.Gain(
                size=(self.in_ch, self.out_ch),
                nfft=self.nfft,
                requires_grad=False,
                map=lambda x: x,
                alias_decay_db=self.alias_decay_db,
                device=self.device,
                dtype=self.dtype,
            )

        self._configure_onset(onset_delay, early_reflections)

        return system.Series(
            OrderedDict(
                {
                    "onset_delay": onset_delay,
                    "early_reflections": early_reflections,
                }
            )
        )

    def _configure_onset(self, onset_delay, early_reflections):
        """Configure onset behavior based on early_reflections_type."""
        # Ensure onset has correct number of values
        if len(self.onset) != self.in_ch:
            self.onset = self.onset.repeat(self.in_ch)
        if self.early_reflections_type is None:
            onset_delay.assign_value(
                onset_delay.sample2s(torch.zeros((self.in_ch,), device=self.device))
            )
            early_reflections.assign_value(torch.zeros((self.in_ch, 1)))
            
        elif self.early_reflections_type == "gain":
            onset_delay.assign_value(onset_delay.sample2s(torch.tensor(self.onset)))
            early_reflections.assign_value(torch.randn((self.in_ch, 1)))

        elif self.early_reflections_type == "FIR":
            velvet_noise = signal_gallery(
                batch_size=1,
                n_samples=early_reflections.size[0],
                n=self.in_ch,
                signal_type="velvet",
                fs=self.fs,
                rate=max(int(torch.rand(1,) / 100 * self.fs), self.fs / early_reflections.size[0] + 1),
            ).squeeze(0)
            early_reflections.assign_value(velvet_noise)
        else:
            raise ValueError(f"Unsupported onset type: {self.early_reflections_type}")

    def _create_attenuation(self, config: FDNAttenuation):
        """Create attenuation based on configuration type."""
        if config.attenuation_type == "homogeneous":
            return self._create_homogeneous_attenuation(config)
        elif config.attenuation_type == "geq":
            return self._create_geq_attenuation(config)
        elif config.attenuation_type == "first_order_lp":
            return self._create_first_order_attenuation(config)
        else:
            raise ValueError(f"Unsupported attenuation type: {config.attenuation_type}")

    def _create_homogeneous_attenuation(self, config: FDNAttenuation):
        """Create homogeneous attenuation."""
        attenuation = dsp.parallelGain(
            size=(self.N,),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )
        attenuation.map = MapGamma(self.delay_lengths)

        if config.attenuation_param == None:
            # Random attenuation within range
            random_rt = (
                torch.rand((1,), device=self.device)
                * (config.attenuation_range[1] - config.attenuation_range[0])
                + config.attenuation_range[0]
            )
            attenuation_value = self._calculate_attenuation_value(random_rt)
        else:
            # Use specific attenuation parameter
            attenuation_value = self._calculate_attenuation_value(
                torch.tensor(config.attenuation_param, device=self.device)
            )

        attenuation.assign_value(attenuation_value)
        return attenuation

    def _calculate_attenuation_value(self, rt_value: torch.Tensor) -> torch.Tensor:
        """Calculate attenuation value from RT value."""
        return 10 ** (
            (rt2slope(rt_value, self.fs) * torch.ones((self.N,), device=self.device))
            / 20
        )

    def _create_geq_attenuation(self, config: FDNAttenuation):
        """Create GEQ-based attenuation."""

        attenuation = parallelFDNAccurateGEQ(
            octave_interval=config.t60_octave_interval,
            nfft=self.nfft,
            fs=self.fs,
            delays=self.delay_lengths,
            alias_decay_db=self.alias_decay_db,
            start_freq=config.t60_center_freq[0],
            end_freq=config.t60_center_freq[-1],
            device=None,
            dtype=self.dtype,
        )
        attenuation.assign_value(
            torch.tensor(config.attenuation_param[0], device=self.device)
        )
        return attenuation

    def _create_first_order_attenuation(self, config: FDNAttenuation):
        """Create first-order shelving attenuation."""

        attenuation = parallelFirstOrderShelving(
            nfft=self.nfft,
            fs=self.fs,
            rt_nyquist=config.rt_nyquist,
            delays=self.delay_lengths,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )
        attenuation.assign_value(
            torch.tensor(config.attenuation_param[0], device=self.device)
        )
        return attenuation


class GroupedFDN(BaseFDN):
    """
    Grouped Feedback Delay Network (FDN) implementation.

    This class extends BaseFDN to support grouped configurations for
    more complex reverberation synthesis scenarios. The GroupedFDN organizes
    delay lines into multiple groups with independent mixing matrices and
    controlled coupling between groups.

    The grouped architecture allows for:
    - Independent reverberation characteristics per group
    - Controllable inter-group coupling through coupling angles
    - Selective input/output routing via gain masks
    - Enhanced spatial and timbral diversity in reverberation

    Attributes
    ----------
    n_groups : int
        Number of delay line groups in the FDN.
    mixing_angles : List[float]
        Mixing angles for creating orthogonal matrices within each group.
    coupling_angles : List[float]
        Angles controlling the coupling strength between different groups.
    input_mask : torch.Tensor
        Mask tensor for selective input gain routing to delay lines.
    output_mask : torch.Tensor
        Mask tensor for selective output gain routing from delay lines.

    Notes
    -----
    - The total number of delay lines must equal N * n_groups
    - Each group has N delay lines with its own mixing matrix
    - Inter-group coupling is controlled by the coupling matrix R
    - The mixing matrix combines intra-group and inter-group interactions
    - Input and output masks enable flexible routing configurations

    References:
    Das, O., & Abel, J. S. (2021). Grouped feedback delay networks for modeling
    of coupled spaces. J. Audio Eng. Soc, 69(7/8), 486-496.
    """

    def __init__(
        self,
        config: GFDNConfig,
        nfft: int,
        alias_decay_db: float,
        delay_lengths: List[int],
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = True,
        output_layer: Literal["freq_complex", "freq_mag", "time"] = "time",
    ) -> None:
        # Additional initialization for grouped FDN
        self.n_groups = config.n_groups
        self.mixing_angles = config.mixing_angles
        self.coupling_angles = config.coupling_angles
        self.input_mask = torch.tensor(config.input_gains_mask, device=device)
        self.output_mask = torch.tensor(config.output_gains_mask, device=device)
        super().__init__(
            config=config,
            nfft=nfft,
            alias_decay_db=alias_decay_db,
            delay_lengths=delay_lengths,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            output_layer=output_layer,
        )

    def _validate_delays(self, config: BaseConfig, delay_lengths: List[int]) -> None:
        """Validate delay lengths."""
        if config.N * config.n_groups != len(delay_lengths):
            raise ValueError(
                f"N x n_groups ({config.N * config.n_groups}) must match the length of delay_lengths ({len(delay_lengths)})"
            )

    def _create_delay_lines(self):
        """Create parallel delay lines."""
        delays = dsp.parallelDelay(
            size=(self.N * self.n_groups,),
            max_len=self.delay_lengths.max(),
            nfft=self.nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )
        delays.assign_value(delays.sample2s(self.delay_lengths))
        return delays

    def _create_mixing_matrix(self):
        """Create orthogonal mixing matrix."""

        def create_submatrix(angles: List[float], iters: Optional[int] = None):
            """Create a submatrix for each group."""
            X = torch.zeros(2, 2, device=self.device)
            X.fill_diagonal_(torch.cos(torch.tensor(angles[0], device=self.device)))
            X[1, 0] = -torch.sin(torch.tensor(angles[0], device=self.device))
            X[0, 1] = torch.sin(torch.tensor(angles[0], device=self.device))

            if iters is None:
                iters = torch.log2(torch.tensor(self.N)).int().item() - 1
            for i in range(iters):
                if len(angles) > 1:
                    X = torch.kron(X, create_submatrix([angles[i]]))
                else:
                    X = torch.kron(X, X)
            return X

        M = torch.zeros(
            self.N * self.n_groups, self.N * self.n_groups, device=self.device
        )
        # creating the coupling matrix
        R = torch.ones(self.n_groups, self.n_groups, device=self.device)

        idx = 0
        coupling_factors = torch.sqrt(
            (1 / torch.cos(torch.tensor(self.coupling_angles, device=self.device))) ** 2
            - 1
        )
        for i in range(self.n_groups):
            for j in range(i + 1, self.n_groups):
                if i != j:
                    R[i, j] = -torch.tensor(coupling_factors[idx], device=self.device)
                    R[j, i] = -R[i, j]
                idx += 1
        row_sums = torch.sum(R**2, dim=1, keepdim=True)
        R = torch.div(R , row_sums)
        # create diagonal submatrices
        for i in range(self.n_groups):
            submatrix = create_submatrix([self.mixing_angles[i]])
            indx = [i * self.N, (i + 1) * self.N]
            M[indx[0] : indx[1], indx[0] : indx[1]] = submatrix * R[i, i]

        # fill the off-diagonal blocks
        for i in range(self.n_groups):
            submatrix_i = create_submatrix([self.mixing_angles[i] / 2])
            for j in range(self.n_groups): 
                if i != j:
                    submatrix_j = create_submatrix([self.mixing_angles[j] / 2])
                    M[i * self.N : (i + 1) * self.N, j * self.N : (j + 1) * self.N] = R[
                        i, j
                    ] * torch.matmul(submatrix_i, submatrix_j)

        mixing_matrix = dsp.Matrix(
            size=(self.N * self.n_groups, self.N * self.n_groups),
            nfft=self.nfft,
            matrix_type="random",
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )  # TODO implement a learnable mixing matrix, with learnable angles

        mixing_matrix.assign_value(M)
        return mixing_matrix

    def _create_fdn_branch(
        self, attenuation_config: FDNAttenuation, mixing_matrix_config: FDNMixing
    ):
        """Create the main FDN branch (branch A)."""
        # Input and output gains
        input_gain = dsp.Gain(
            size=(self.N * self.n_groups, self.in_ch),
            nfft=self.nfft,
            map=lambda x: x * self.input_mask,
            requires_grad=self.requires_grad,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype,
        )

        output_gain = dsp.Gain(
            size=(self.out_ch, self.N * self.n_groups),
            nfft=self.nfft,
            map=lambda x: x * self.output_mask,
            requires_grad=self.requires_grad,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype
        )

        # Feedback loop components
        delays = self._create_delay_lines()
        mixing_matrix = self._create_mixing_matrix()
        attenuation = self._create_attenuation(attenuation_config)
        # Feedback path
        feedback = system.Series(
            OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
        )

        # Recursion
        feedback_loop = system.Recursion(fF=delays, fB=feedback)

        # Complete FDN branch
        return system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )

    def _create_homogeneous_attenuation(self, config: FDNAttenuation):
        """Create homogeneous attenuation."""
        attenuation = dsp.parallelGain(
            size=(self.N * self.n_groups,),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype
        )
        attenuation.map = MapGamma(self.delay_lengths)

        if config.attenuation_param == None:
            # Random attenuation within range
            random_rt = (
                torch.rand((1,), device=self.device)
                * (config.attenuation_range[1] - config.attenuation_range[0])
                + config.attenuation_range[0]
            )
            attenuation_value = self._calculate_attenuation_value(random_rt)
        else:
            # Use specific attenuation parameter
            attenuation_value = self._calculate_attenuation_value(
                torch.tensor(config.attenuation_param, device=self.device)
            )

        attenuation.assign_value(attenuation_value)
        return attenuation

    def _calculate_attenuation_value(self, rt_value: torch.Tensor) -> torch.Tensor:
        """Calculate attenuation value from RT value."""
        assert rt_value.dim() == 2, "RT value must be a 2D tensor"
        att_value = 10 ** (
            (rt2slope(rt_value, self.fs) * torch.ones((self.N,), device=self.device))
            / 20
        )
        return att_value.flatten()

    def _create_geq_attenuation(self, config: FDNAttenuation):
        """Create GEQ-based attenuation."""

        attenuation = parallelGFDNAccurateGEQ(
            octave_interval=config.t60_octave_interval,
            n_groups=self.n_groups,
            nfft=self.nfft,
            fs=self.fs,
            delays=self.delay_lengths,
            alias_decay_db=self.alias_decay_db,
            start_freq=config.t60_center_freq[0],
            end_freq=config.t60_center_freq[-1],
            device=None,
            dtype=self.dtype
        )
        params = torch.tensor(config.attenuation_param, device=self.device).T.flatten()
        attenuation.assign_value(params)

        return attenuation

    def _create_first_order_attenuation(self, config: FDNAttenuation):
        """Create first-order shelving attenuation."""

        attenuation = parallelFirstOrderShelving(
            nfft=self.nfft,
            fs=self.fs,
            rt_nyquist=config.rt_nyquist,
            delays=self.delay_lengths,
            alias_decay_db=self.alias_decay_db,
            device=self.device,
            dtype=self.dtype
        )
        attenuation.assign_value(config.attenuation_param)
        return attenuation
