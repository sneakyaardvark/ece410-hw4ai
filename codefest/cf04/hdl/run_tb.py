#!/usr/bin/env python
"""Run the MAC cocotb testbench with Icarus Verilog.

Usage: python run_tb.py [mac_llm_A.v | mac_llm_B.v]  (default: mac_llm_A.v)
"""
import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

HDL_DIR = Path(__file__).parent
source = HDL_DIR / (sys.argv[1] if len(sys.argv) > 1 else "mac_llm_A.v")

runner = get_runner("icarus")
runner.build(
    sources=[source],
    hdl_toplevel="mac",
    always=True,
    build_dir=HDL_DIR / "sim_build",
)
runner.test(
    hdl_toplevel="mac",
    test_module="mac_tb",
    hdl_toplevel_lang="verilog",
    build_dir=HDL_DIR / "sim_build",
    test_dir=HDL_DIR,
)
