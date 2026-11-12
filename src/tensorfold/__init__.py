"""PLI (fork of TensorFold): fast, exact LLM decoding on Apple Silicon behind an OpenAI-compatible endpoint."""

# First, before any module reads its settings: environment variables set under their old names reach the new names.
from tensorfold.pli import env_names as _env_names

_env_names.apply()

__version__ = "4.2.1"

# PLI is a fork of TensorFold: the Python module keeps the name tensorfold (simpler to take upstream changes),
# the installed package is called pli.
DISTRIBUTION = "pli"
NOM = "PLI"
AMONT = "TensorFold 0.6.6"
