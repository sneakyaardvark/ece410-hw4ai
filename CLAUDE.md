# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECE 410 homework assignment repository combining AI/ML (PyTorch) with hardware design (Verilog).

## Development Environment

This project uses **Nix Flakes** for reproducible development environments with direnv integration.

```bash
# Enter development environment (automatically loads via direnv)
direnv allow

# Or manually enter the shell
nix develop
```

### Available Tools

- **Python 3.13** with torch, torchvision, pip
- **iverilog** for Verilog simulation

### Update Dependencies

```bash
nix flake update
```

## Build & Run Commands

No traditional build system is configured yet. Use tools directly:

```bash
# Python/PyTorch
python script.py

# Verilog simulation
iverilog -o output design.v testbench.v
vvp output
```
