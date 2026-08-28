# ---------------------------------------------------------------------------
# Gram-Schmidt QR -- simulation makefile (Icarus Verilog)
#
#   make            run the default 8x4, P=2 case end to end
#   make M=16 N=8 P=4 CASE=illcond
#   make rsqrt      unit-test the inverse square root
#   make regress    full geometry / corner-case sweep
#   make wave       same as `make test` but dumps tb_gs_top.vcd
#   make clean
# ---------------------------------------------------------------------------

IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3

M     ?= 8
N     ?= 4
P     ?= 2
W     ?= 18
F     ?= 16
CASE  ?= random
SEED  ?= 1
RUNS  ?= 2

RTL   := $(wildcard ../rtl/*.v)
BUILD := build
MODEL := ../model

IVFLAGS := -g2005 -Wall
DEFS    := -DGS_M=$(M) -DGS_N=$(N) -DGS_P=$(P) -DGS_W=$(W) -DGS_F=$(F) -DGS_RUNS=$(RUNS)

.PHONY: all test stim rsqrt regress wave lint clean rom

all: test

$(BUILD):
	@mkdir -p $(BUILD)

# regenerate the Newton-Raphson seed table
rom:
	cd $(MODEL) && $(PYTHON) gen_seed_rom.py

stim: | $(BUILD)
	@cd $(MODEL) && $(PYTHON) gen_stimulus.py --m $(M) --n $(N) --p $(P) \
		--w $(W) --f $(F) --case $(CASE) --seed $(SEED) --out ../sim/$(BUILD)

$(BUILD)/tb.vvp: $(RTL) ../tb/tb_gs_top.v | $(BUILD)
	$(IVERILOG) $(IVFLAGS) $(DEFS) -o $@ $(RTL) ../tb/tb_gs_top.v

test: stim $(BUILD)/tb.vvp
	@cd $(BUILD) && $(VVP) tb.vvp
	@cd $(BUILD) && $(PYTHON) ../$(MODEL)/check_qr.py --m $(M) --n $(N) --w $(W) --f $(F) --dir .

wave: stim $(BUILD)/tb.vvp
	@cd $(BUILD) && $(VVP) tb.vvp +vcd
	@echo "wrote $(BUILD)/tb_gs_top.vcd"

$(BUILD)/tb_rsqrt.vvp: ../rtl/rsqrt_unit.v ../rtl/rsqrt_seed_rom.v ../tb/tb_rsqrt.v | $(BUILD)
	$(IVERILOG) $(IVFLAGS) -o $@ ../rtl/rsqrt_unit.v ../rtl/rsqrt_seed_rom.v ../tb/tb_rsqrt.v

rsqrt: $(BUILD)/tb_rsqrt.vvp
	@cd $(BUILD) && $(VVP) tb_rsqrt.vvp

regress:
	@$(PYTHON) regress.py

# elaboration-only pass: catches width/port mistakes without running anything
lint: | $(BUILD)
	$(IVERILOG) $(IVFLAGS) $(DEFS) -o /dev/null $(RTL) ../tb/tb_gs_top.v
	@echo "lint clean"

clean:
	rm -rf $(BUILD)
