// ---------------------------------------------------------------------------
// vec_mac.v : P-lane multiply / adder-tree / accumulate unit.
//
// Consumes P signed Q(W-F).F samples of each operand per cycle and accumulates
// their inner product into a full-precision ACC_W-bit accumulator held in
// Q(ACC_W-2F).2F format (no rounding happens inside the accumulation, so the
// dot product is exact up to the guard-bit budget).
//
// Pipeline
//   stage 1 : P parallel W x W multipliers          (registered)
//   stage 2 : sign-extended adder tree              (registered)
//   stage 3 : accumulator                           (registered)
//
// Latency: `acc` reflects a sample presented with `en` 3 cycles later.
// `clr` zeroes the accumulator synchronously and must be asserted no later
// than the first `en` of a new dot product.
// ---------------------------------------------------------------------------
`default_nettype none

module vec_mac #(
    parameter integer W     = 18,
    parameter integer P     = 2,
    parameter integer ACC_W = 48
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clr,
    input  wire                    en,
    input  wire [P*W-1:0]          a,
    input  wire [P*W-1:0]          b,
    output reg  signed [ACC_W-1:0] acc
);

    localparam integer PW = 2*W;

    reg  signed [PW-1:0]    prod [0:P-1];
    reg                     v1, v2;
    reg  signed [ACC_W-1:0] sum2;
    reg  signed [ACC_W-1:0] sum_c;

    integer i;

    // ---- stage 1 : multipliers --------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            for (i = 0; i < P; i = i + 1) prod[i] <= {PW{1'b0}};
        end else begin
            v1 <= en;
            for (i = 0; i < P; i = i + 1)
                prod[i] <= $signed(a[i*W +: W]) * $signed(b[i*W +: W]);
        end
    end

    // ---- stage 2 : adder tree (synthesis balances this automatically) ------
    always @* begin
        sum_c = {ACC_W{1'b0}};
        for (i = 0; i < P; i = i + 1)
            sum_c = sum_c + {{(ACC_W-PW){prod[i][PW-1]}}, prod[i]};
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            v2   <= 1'b0;
            sum2 <= {ACC_W{1'b0}};
        end else begin
            v2   <= v1;
            sum2 <= sum_c;
        end
    end

    // ---- stage 3 : accumulator --------------------------------------------
    always @(posedge clk) begin
        if (!rst_n)     acc <= {ACC_W{1'b0}};
        else if (clr)   acc <= {ACC_W{1'b0}};
        else if (v2)    acc <= acc + sum2;
    end

endmodule

`default_nettype wire
