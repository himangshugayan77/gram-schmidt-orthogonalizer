// ---------------------------------------------------------------------------
// vec_axpy.v : P-lane scale-and-subtract unit.  Two modes, one datapath:
//
//   use_dst = 0 :  dst_out =            sat_round( src * scalar )   (NORMALISE)
//   use_dst = 1 :  dst_out = dst_in  -  sat_round( src * scalar )   (PROJECT)
//
// `src` and `dst_in` are P packed Q(W-F).F samples.  `scalar` is a signed
// Q(SC_W-SC_F).SC_F value, wide enough to hold 1/||v|| when the norm is small.
// Sharing one multiplier array between normalisation and projection removal
// halves the DSP count versus separate units.
//
// Pipeline
//   stage 1 : P parallel W x SC_W multipliers, dst_in aligned by one register
//   stage 2 : re-quantise, subtract, saturate
// Latency: 2 cycles from `en` to `vld`.
// ---------------------------------------------------------------------------
`default_nettype none

module vec_axpy #(
    parameter integer W    = 18,
    parameter integer P    = 2,
    parameter integer SC_W = 32,
    parameter integer SC_F = 24
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   en,
    input  wire                   use_dst,
    input  wire signed [SC_W-1:0] scalar,
    input  wire [P*W-1:0]         src,
    input  wire [P*W-1:0]         dst_in,
    output wire [P*W-1:0]         dst_out,
    output reg                    vld,
    output reg                    ovf
);

    localparam integer PW = W + SC_W;

    reg  signed [PW-1:0] prod [0:P-1];
    reg  [P*W-1:0]       dst_r;
    reg                  use_r;
    reg                  v1;
    reg  [P*W-1:0]       dst_o;

    wire [P*W-1:0]       t_flat;    // sat_round(prod)
    wire [P*W-1:0]       s_flat;    // sat(dst_r - t)
    wire [P-1:0]         ovf_m;     // multiplier requantise overflow
    wire [P-1:0]         ovf_s;     // subtract overflow

    integer i;

    // ---- stage 1 -----------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            v1    <= 1'b0;
            use_r <= 1'b0;
            dst_r <= {P*W{1'b0}};
            for (i = 0; i < P; i = i + 1) prod[i] <= {PW{1'b0}};
        end else begin
            v1    <= en;
            use_r <= use_dst;
            dst_r <= dst_in;
            for (i = 0; i < P; i = i + 1)
                prod[i] <= $signed(src[i*W +: W]) * scalar;
        end
    end

    // ---- stage 2 combinational : requantise + subtract + saturate ----------
    genvar g;
    generate
        for (g = 0; g < P; g = g + 1) begin : lane
            wire signed [W-1:0] t;
            round_sat #(.IN_W(PW), .OUT_W(W), .SHIFT(SC_F)) u_rs (
                .din (prod[g]),
                .dout(t),
                .ovf (ovf_m[g])
            );
            wire signed [W:0] diff = $signed(dst_r[g*W +: W]) - $signed({t[W-1], t});
            assign ovf_s[g]        = (diff[W] != diff[W-1]);
            wire signed [W-1:0] sat_d = ovf_s[g]
                                      ? (diff[W] ? {1'b1, {(W-1){1'b0}}}    // -max
                                                 : {1'b0, {(W-1){1'b1}}})   // +max
                                      : diff[W-1:0];
            assign t_flat[g*W +: W] = t;
            assign s_flat[g*W +: W] = sat_d;
        end
    endgenerate

    // ---- stage 2 register --------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            vld   <= 1'b0;
            ovf   <= 1'b0;
            dst_o <= {P*W{1'b0}};
        end else begin
            vld <= v1;
            ovf <= v1 & (|ovf_m | (use_r & |ovf_s));
            for (i = 0; i < P; i = i + 1)
                dst_o[i*W +: W] <= use_r ? s_flat[i*W +: W] : t_flat[i*W +: W];
        end
    end

    assign dst_out = dst_o;

endmodule

`default_nettype wire
