# FlareVerb

An Open-Source Library for Room Impulse Response (RIR) Synthesis and Analysis in PyTorch based on FLAMO.

## Installation

```bash
pip install flareverb
```

## Quick Start

```python
import torch
from flareverb import BaseFDN, FDNConfig

# Create a basic FDN configuration
config = FDNConfig(
    N=8,  # Number of delay lines
    fs=48000,  # Sample rate
    # ... other configuration parameters
)

# Initialize the FDN
fdn = BaseFDN(
    config=config,
    nfft=2048,
    alias_decay_db=-60,
    delay_lengths=[prime_delays],  # List of prime delay lengths
)

# Process audio
output = fdn(input_audio, ext_params)
```

## Project Structure

```
src/flareverb/
├── __init__.py          # Package initialization
├── reverb.py            # Core FDN implementations
├── generate.py          # RIR generation utilities
├── sampling.py          # Delays and gains sampling methods
├── analysis.py          # Acoustic analysis functions
├── utils.py             # Utility functions
└── config/              # Configuration modules
```

## Requirements

- Python >= 3.10
- PyTorch
- FLAMO == 0.1.4
- pydantic
- pyyaml
- pandas

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Links

- [GitHub Repository](https://github.com/gdalsanto/flare)
- [Issues](https://github.com/gdalsanto/flare/issues)
