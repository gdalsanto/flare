# FLARE
[LBDP](https://github.com/gdalsanto/flare/blob/main/AES_AIMLA_abstract.pdf) | [poster](https://github.com/gdalsanto/flare/blob/main/AES_AIMLA_poster.pdf)  

An Open-Source Library for Room Impulse Response Synthesis and Analysis in PyTorch based on [FLAMO](https://github.com/gdalsanto/flamo).  

This library will be presented at the AES Conference on Artificial Intelligence and Machine Learning for Audio (AIMLA), Queen Mary University, London (UK), 8-10 September 2025

More information soon! 
## Installation

```bash
pip install flareverb
```

## Project Structure

```
src/flareverb/
├── reverb.py            # Core FDN implementations
├── generate.py          # RIR generation utilities
├── sampling.py          # Delays, gains, and filters sampling
├── analysis.py          # Acoustic analysis functions
├── utils.py             # Utility functions
├── config/              # Configuration modules
└── data/                # Data folder (contains absorption coefficients)

```

## Configuration

FLARE uses Pydantic models for configuration management. 
The suggested usage is via YAML files: you can write your configuration (for FDN, attenuation, mixing, etc.) in a YAML file, then load it in Python and passing it to the Pydantic config classes. The config classes (FDNConfig, GFDNConfig) are designed to accept dictionaries, so you can parse your YAML into a dict and instantiate the config objects:
```python
import yaml
from flareverb.config.config import BaseConfig

with open("my_config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

config = BaseConfig(**config_dict)

```
The main configuration classes are:

### FDNConfig
Core FDN configuration parameters:
- `N`: Number of delay lines (default: 6)
- `fs`: Sampling frequency in Hz (default: 48000)
- `in_ch` / `out_ch`: Input/output channels (default: 1)
- `delay_range_ms`: Delay lengths range in milliseconds (default: [20.0, 50.0])
- `delay_log_spacing`: Use logarithmic spacing for delays (default: False)
- `early_reflections_type`: Type of early reflections - 'gain', 'FIR', or None (default: None)
- `drr`: Direct-to-reverberant ratio (default: 0.25, auto-set to 0 if early_reflections_type is None)
- `gain_init`: Gain initialization - 'randn' or 'uniform' (default: 'randn')

### FDNAttenuation
Attenuation filter configuration:
- `attenuation_type`: Filter type - 'homogeneous', 'geq', or 'first_order_lp' (default: 'homogeneous')
- `attenuation_range`: RT range in seconds when attenuation_param not given (default: [0.5, 3.5])
- `t60_center_freq`: Center frequencies for T60 (default: [63, 125, 250, 500, 1000, 2000, 4000, 8000])
- `rt_nyquist`: RT at Nyquist frequency for first-order filters (default: 0.2)

### FDNMixing
Mixing matrix configuration:
- `mixing_type`: Matrix type - 'orthogonal', 'householder', 'hadamard', or 'rotation' (default: 'orthogonal')
- `is_scattering`: Use scattering matrix (default: False)
- `is_velvet_noise`: Use velvet noise (default: False)
- `n_stages`: Number of scattering stages (default: 3)

### GFDNConfig
Grouped FDN configuration (inherits from FDNConfig):
- `n_groups`: Number of groups (default: 2)
- `coupling_angles`: Inter-group coupling angles (default: [0.0])
- `mixing_angles`: Intra-group mixing angles (default: [0.0, 0.0])

The main orchestration function is [fdn_dataset](https://github.com/gdalsanto/flare/blob/103ed82c0546e7da7001c7029bfd067ad944d2cd/src/flareverb/generate.py#L385).  

## Requirements

- Python >= 3.10
- PyTorch
- FLAMO >= 0.1.13
- pydantic
- pyyaml
- pandas

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Links

- [Issues](https://github.com/gdalsanto/flare/issues)
