"""PLI: the folded-paper mark printed once when ``pli serve`` starts (PLI 4.0.1).

Text only, written to standard output before the model loads; touches nothing else.
"""

from __future__ import annotations

MARK = r"""
        _______
       |\\   //
       |  \\/
       |   \\
       |_____\\
     //\\      |
   //  | \\    |
 //\\  | //\\  |
/____\\|/    \\|
"""


def text(version: str) -> str:
    """The mark, with the name and the version beside its third and fourth lines."""

    lines = MARK.strip("\n").split("\n")
    lines[2] = f"{lines[2]:<18}PLI {version}"
    lines[3] = f"{lines[3]:<18}Parallel Linked Inference"
    return "\n".join(lines) + "\n"
