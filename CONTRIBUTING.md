# Contributing to PLI

Issues and pull requests are welcome at [github.com/nicopingu/pli](https://github.com/nicopingu/pli).

- **Apple Silicon only.** PLI runs on macOS with MLX; say which Mac (chip, memory, macOS) you used.
- **Tests.** Run `python -m pytest` (install with `pip install -e ".[test]"`). Tests that need hardware your Mac lacks
  are skipped with the reason; say which ran and which were skipped.
- **Speed.** A change meant to make PLI faster comes with a measurement before and after, on the same Mac, model and
  settings (for example with `tools/pli_bench.py`).
- **Text.** A change must give the same text as before. If the text changes (other arithmetic, other rounding), the
  pull request says where and why.
- **Settings.** A new behaviour goes behind a `PLI_*` variable, off by default, and is listed in
  [docs/switches.md](docs/switches.md).
- **One change per pull request**, with a short description and the issue it closes, if any.

By contributing you agree that your contribution is licensed under the Apache License 2.0, as the rest of PLI.
