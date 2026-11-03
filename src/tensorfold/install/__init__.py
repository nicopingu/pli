"""``pli setup`` and the commands of PLI's per-user service: status, start, stop, restart, logs, uninstall.

install.sh only creates the virtual environment and installs PLI in it, then runs ``pli setup``. Nothing here imports
MLX: the MTP head is written in 4 bits by ``python -m tensorfold.install.mtp_head`` in a process of its own.
"""
