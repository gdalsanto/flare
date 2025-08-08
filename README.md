# FLARE

An Open-Source Library for Room Impulse Response Synthesis and Analysis in PyTorch based on [FLAMO](https://github.com/gdalsanto/flamo).

## Installation

```bash
pip install flareverb
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
