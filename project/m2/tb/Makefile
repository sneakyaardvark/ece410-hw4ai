VENV_BIN      = $(realpath $(PWD)/../../.venv/bin)
export PATH  := $(VENV_BIN):$(PATH)

SIM           ?= icarus
TOPLEVEL_LANG  = verilog

RTL = $(PWD)/../rtl

COCOTB_MAKEFILES = $(shell cocotb-config --makefiles)

# ----------------------------------------------------------------------------
# DUT targets — run as: make mac | make mac_array | make all
# ----------------------------------------------------------------------------

.PHONY: mac mac_array lif_cell lif_bank compute_core interface all

mac:
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim \
		TOPLEVEL=tb_snn_mac \
		VERILOG_SOURCES="$(RTL)/snn_mac.sv $(PWD)/tb_snn_mac.sv" \
		COCOTB_TEST_MODULES=test_snn_mac \
		SIM_BUILD=sim_build_mac

mac_array:
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim \
		TOPLEVEL=tb_snn_mac_array \
		VERILOG_SOURCES="$(RTL)/snn_mac.sv $(RTL)/snn_mac_array.sv $(PWD)/tb_snn_mac_array.sv" \
		COCOTB_TEST_MODULES=test_snn_mac_array \
		SIM_BUILD=sim_build_mac_array

lif_cell:
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim \
		TOPLEVEL=tb_snn_lif_cell \
		VERILOG_SOURCES="$(RTL)/snn_lif_cell.sv $(PWD)/tb_snn_lif_cell.sv" \
		COCOTB_TEST_MODULES=test_snn_lif_cell \
		SIM_BUILD=sim_build_lif_cell

lif_bank:
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim \
		TOPLEVEL=tb_snn_lif_bank \
		VERILOG_SOURCES="$(RTL)/snn_lif_cell.sv $(RTL)/snn_lif_bank.sv $(PWD)/tb_snn_lif_bank.sv" \
		COCOTB_TEST_MODULES=test_snn_lif_bank \
		SIM_BUILD=sim_build_lif_bank

compute_core:
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim \
		TOPLEVEL=tb_compute_core \
		VERILOG_SOURCES="$(RTL)/snn_mac.sv $(RTL)/snn_mac_array.sv $(RTL)/snn_lif_cell.sv $(RTL)/snn_lif_bank.sv $(RTL)/compute_core.sv $(PWD)/tb_compute_core.sv" \
		COCOTB_TEST_MODULES=test_compute_core \
		SIM_BUILD=sim_build_compute_core

interface:
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim \
		TOPLEVEL=tb_interface \
		VERILOG_SOURCES="$(RTL)/interface.sv $(PWD)/tb_interface.sv" \
		COCOTB_TEST_MODULES=test_interface \
		SIM_BUILD=sim_build_interface

all: mac mac_array lif_cell lif_bank compute_core interface

# ----------------------------------------------------------------------------
# Include cocotb rules so `make sim` also works from this directory
# ----------------------------------------------------------------------------

include $(COCOTB_MAKEFILES)/Makefile.sim
