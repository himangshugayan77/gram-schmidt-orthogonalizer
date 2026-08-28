// ---------------------------------------------------------------------------
// ram_1r1w.v : 1-read / 1-write synchronous RAM, 1-cycle read latency.
// Used for the (small) R matrix, which never needs two concurrent reads.
// ---------------------------------------------------------------------------
`default_nettype none

module ram_1r1w #(
    parameter integer DW    = 18,
    parameter integer DEPTH = 16,
    parameter integer AW    = 4
)(
    input  wire           clk,
    input  wire           we,
    input  wire [AW-1:0]  waddr,
    input  wire [DW-1:0]  wdata,
    input  wire [AW-1:0]  raddr,
    output reg  [DW-1:0]  rdata
);

    reg [DW-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end

endmodule

`default_nettype wire
