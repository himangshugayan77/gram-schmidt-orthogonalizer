// ---------------------------------------------------------------------------
// ram_2r1w.v : 2-read / 1-write synchronous RAM.
//
// Implemented as two mirrored banks that are written identically; each bank
// serves one read port.  This is the standard "duplicate for extra read port"
// trick and infers 2x block RAM on Xilinx/Intel without any vendor primitive.
//
// Read latency = 1 cycle.  Read-during-write to the same address returns the
// OLD data (read-first); the Gram-Schmidt controller never creates such a
// collision, the behaviour is defined here only for determinism.
// ---------------------------------------------------------------------------
`default_nettype none

module ram_2r1w #(
    parameter integer DW    = 36,
    parameter integer DEPTH = 16,
    parameter integer AW    = 4
)(
    input  wire           clk,
    // write port
    input  wire           we,
    input  wire [AW-1:0]  waddr,
    input  wire [DW-1:0]  wdata,
    // read port A
    input  wire [AW-1:0]  raddr_a,
    output reg  [DW-1:0]  rdata_a,
    // read port B
    input  wire [AW-1:0]  raddr_b,
    output reg  [DW-1:0]  rdata_b
);

    reg [DW-1:0] bank_a [0:DEPTH-1];
    reg [DW-1:0] bank_b [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) begin
            bank_a[waddr] <= wdata;
            bank_b[waddr] <= wdata;
        end
        rdata_a <= bank_a[raddr_a];
        rdata_b <= bank_b[raddr_b];
    end

`ifdef GS_SIM_INIT
    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            bank_a[i] = {DW{1'b0}};
            bank_b[i] = {DW{1'b0}};
        end
    end
`endif

endmodule

`default_nettype wire
