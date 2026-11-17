"""PLI (fork of TensorFold): fast, exact LLM decoding on Apple Silicon behind an OpenAI-compatible endpoint."""

# First, before any module reads its settings: environment variables set under their old names reach the new names.
from pli.additions import env_names as _env_names

_env_names.apply()

__version__ = "5.6.5"

# PLI is a fork of TensorFold; since PLI 5 the Python package is named pli (the old name tensorfold is a
# compatibility package, src/tensorfold); the installed distribution is called pli too.
DISTRIBUTION = "pli"
NOM = "PLI"
AMONT = "TensorFold 0.6.6"
