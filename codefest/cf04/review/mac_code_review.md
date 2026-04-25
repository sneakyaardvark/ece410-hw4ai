# CF04 CLLM HDL Code Review

LLM A: Claude, Sonnet 4.6

LLM B: Gemini, "Fast" mode

I did not discern any errors in either module. Both were followed up with a similar prompt to the example (check sensitivity list and sign extension).

## Compiling
No errors in either module when compiled with iverilog.

## Simulate

Testbench: `codefest/cf04/hdl/mac_tb.py`  
Runner: `codefest/cf04/hdl/run_tb.py`  
Simulator: Icarus Verilog 12.0 (stable), cocotb 2.0.1

Test sequence: reset for 1 cycle → `a=3, b=4` for 3 cycles (expect out = 12, 24, 36) → assert reset → `a=−5, b=2` for 2 cycles (expect out = −10, −20).

### LLM A (`mac_llm_A.v`)

```
** TEST                          STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
** mac_tb.test_mac                PASS   13000000000.00           0.00   21880398073836.28  **
** TESTS=1 PASS=1 FAIL=0 SKIP=0          13000000000.00           0.00   12656906220984.21  **
```

### LLM B (`mac_llm_B.v`)

```
** TEST                          STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
** mac_tb.test_mac                PASS   13000000000.00           0.00   22102128901499.80  **
** TESTS=1 PASS=1 FAIL=0 SKIP=0          13000000000.00           0.00   12838698375323.76  **
```

Both modules pass all 8 assertions. The two implementations are functionally identical; LLM B adds inline comments but the RTL is the same.

## Review
Both LLMs generated correct, simulatorable, and synthesizable HDL. From a programmer's perspective, Gemini is an *over commenter* where there can be almost as many comments as lines of code, but this does not effect the correctness of the code.

## Correct

`yosys -p 'read_verilog -sv mac_correct.v; synth; stat'` — Yosys 0.47, 0 problems reported. Adding `read_verilog -sv` is necessary as some of the features such as `always_ff` require SystemVerilog.

```
=== mac ===

   Number of wires:               1039
   Number of wire bits:           1301
   Number of public wires:           5
   Number of public wire bits:      50
   Number of ports:                  5
   Number of port bits:             50
   Number of memories:               0
   Number of memory bits:            0
   Number of processes:              0
   Number of cells:               1091
     $_ANDNOT_                     351
     $_AND_                         61
     $_NAND_                        46
     $_NOR_                         33
     $_NOT_                         47
     $_ORNOT_                       18
     $_OR_                         133
     $_SDFF_PP0_                    32
     $_XNOR_                        97
     $_XOR_                        273
```
