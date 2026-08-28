# Out-of-context constraints for gs_top.
# Default 6.0 ns (166 MHz); the critical path is the 34x34 multiplier inside
# rsqrt_unit feeding the (3 - m*y^2) subtraction. See docs/architecture.md for
# how to pipeline it if you need to go faster.

create_clock -name clk -period 6.000 [get_ports clk]

# Asynchronous, synchronously-deasserted reset: do not time it.
set_false_path -from [get_ports rst_n]

# Streaming interfaces: budget 40% of the period for board-level routing.
set_input_delay  -clock clk 2.400 [get_ports {s_a_tdata[*] s_a_tvalid m_q_tready m_r_tready start}]
set_output_delay -clock clk 2.400 [get_ports {m_q_tdata[*] m_q_tvalid m_r_tdata[*] m_r_tvalid s_a_tready busy done rank_def ovf_flag}]
