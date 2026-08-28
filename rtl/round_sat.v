// ---------------------------------------------------------------------------
// round_sat.v : arithmetic right shift with round-half-up and symmetric
//               two's-complement saturation.
//
//   dout = sat( (din + 2^(SHIFT-1)) >>> SHIFT )
//
// Purely combinational.  Used everywhere a wide product/accumulator is
// re-quantised back to the Q(W-F).F datapath format.
// ---------------------------------------------------------------------------
`default_nettype none

module round_sat #(
    parameter integer IN_W  = 50,   // input  width  (signed)
    parameter integer OUT_W = 18,   // output width  (signed)
    parameter integer SHIFT = 24    // number of fractional bits to drop
)(
    input  wire signed [IN_W-1:0]  din,
    output wire signed [OUT_W-1:0] dout,
    output wire                    ovf
);

    // ---- rounding constant : 2^(SHIFT-1) -----------------------------------
    wire signed [IN_W-1:0] one    = {{(IN_W-1){1'b0}}, 1'b1};
    wire signed [IN_W-1:0] half   = (SHIFT == 0) ? {IN_W{1'b0}} : (one <<< (SHIFT-1));

    wire signed [IN_W-1:0] rnd    = din + half;
    wire signed [IN_W-1:0] shifted= rnd >>> SHIFT;

    // ---- saturation limits -------------------------------------------------
    wire signed [IN_W-1:0] max_v  = (one <<< (OUT_W-1)) - one;   //  2^(OUT_W-1)-1
    wire signed [IN_W-1:0] min_v  = -(one <<< (OUT_W-1));        // -2^(OUT_W-1)

    wire hi = (shifted > max_v);
    wire lo = (shifted < min_v);

    assign ovf  = hi | lo;
    assign dout = hi ? max_v[OUT_W-1:0]
                : lo ? min_v[OUT_W-1:0]
                     : shifted[OUT_W-1:0];

endmodule

`default_nettype wire
