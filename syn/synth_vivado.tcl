# Out-of-context synthesis of gs_top.
#   vivado -mode batch -source synth_vivado.tcl [-tclargs <part> <period_ns>]

set part   [expr {$argc > 0 ? [lindex $argv 0] : "xc7a100tcsg324-1"}]
set period [expr {$argc > 1 ? [lindex $argv 1] : 6.0}]

set rtl  [file normalize [file join [file dirname [info script]] .. rtl]]
set cons [file normalize [file join [file dirname [info script]] constraints]]

file mkdir build
cd build

read_verilog [glob $rtl/*.v]
read_xdc $cons/gs.xdc

synth_design -top gs_top -part $part -mode out_of_context \
             -directive AreaOptimized_high

# Force DSP inference for the multiplier arrays; without this Vivado may
# implement the narrow rsqrt partial products in fabric.
set_property USE_DSP yes [get_cells -hier -filter {REF_NAME =~ *vec_*}]

opt_design
report_utilization      -file utilization.rpt
report_timing_summary   -file timing.rpt -max_paths 10
report_clock_networks   -file clocks.rpt

puts "----------------------------------------------------------"
puts [format "part          : %s" $part]
puts [format "target period : %.2f ns (%.1f MHz)" $period [expr {1000.0/$period}]]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts [format "worst slack   : %s ns" $wns]
puts "reports in [pwd]"
puts "----------------------------------------------------------"

write_checkpoint -force gs_top_synth.dcp
