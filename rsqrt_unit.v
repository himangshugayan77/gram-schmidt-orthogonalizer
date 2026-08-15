// ---------------------------------------------------------------------------
// rsqrt_unit.v : fixed-point 1/sqrt(x) AND sqrt(x) from one shared multiplier.
//
// Gram-Schmidt needs both  1/||v||  (to normalise) and  ||v||  (the R diagonal).
// Computing the reciprocal square root first and recovering the square root as
// sqrt(x) = m * (1/sqrt(m)) * 2^h avoids a divider entirely -- one Newton-
// Raphson kernel and one extra multiply give both results.
//
// Algorithm
//   1. NORMALISE  find p = msb(x); write x = m * 2^e with e even, m in [1,4).
//                 Internally  Mm = m * 2^MF  (MF = 30) in MW = 32 bits.
//   2. SEED       y0 = ROM[Mm[31:26]] ~ 1/sqrt(m), ~6 correct bits.
//   3. ITERATE    y <- y * (3 - m*y^2) / 2      (NR_ITER times, quadratic)
//   4. DENORMALISE
//                 1/sqrt(x) = y * 2^(-e/2)   -> Y_W bit result, Y_F frac bits
//                 sqrt(x)   = (Mm*y) * 2^(e/2) -> S_W bit result, S_F frac bits
//
// The 3 multiplies per iteration are sequenced onto one registered
// MULW x MULW multiplier (2 cycles each), so the whole unit costs ~1 DSP slice
// and ~2*(3*NR_ITER+1)+4 cycles.  It runs once per column of A, so its latency
// is amortised over M*(N-j) datapath cycles and never limits throughput.
//
// Degenerate input: x = 0 asserts `zero_flag`, returns y = max, s = 0.  The
// controller uses that (together with its own epsilon test) to flag rank
// deficiency instead of producing garbage.
// ---------------------------------------------------------------------------
`default_nettype none

module rsqrt_unit #(
    parameter integer X_W     = 48,   // input width  (unsigned)
    parameter integer X_F     = 32,   // input fractional bits
    parameter integer Y_W     = 32,   // 1/sqrt output width (positive, signed-safe)
    parameter integer Y_F     = 24,   // 1/sqrt output fractional bits
    parameter integer S_W     = 18,   // sqrt output width (positive, signed-safe)
    parameter integer S_F     = 16,   // sqrt output fractional bits
    parameter integer NR_ITER = 3     // Newton-Raphson iterations
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [X_W-1:0]    x,
    output reg  [Y_W-1:0]    y,          // ~ 1/sqrt(x)
    output reg  [S_W-1:0]    s,          // ~   sqrt(x)
    output reg               done,       // 1-cycle pulse
    output reg               zero_flag,
    output reg               sat_flag
);

    // ---- internal working precision ---------------------------------------
    localparam integer MW       = 32;   // mantissa width
    localparam integer MF       = 30;   // mantissa fractional bits (m in [1,4))
    localparam integer LUT_BITS = 6;
    localparam integer MULW     = 34;   // shared multiplier operand width
    localparam integer PRW      = 2*MULW;
    localparam integer TW       = 96;   // denormalisation shifter width

    localparam [3:0] S_IDLE = 4'd0,
                     S_NORM = 4'd1,
                     S_SEED = 4'd2,
                     S_MUL1 = 4'd3,   // t1 = y*y      >> MF
                     S_MUL2 = 4'd4,   // t3 = 3 - m*t1
                     S_MUL3 = 4'd5,   // y  = y*t3     >> (MF+1)
                     S_SQRT = 4'd6,   // pr = Mm*y
                     S_FIN  = 4'd7,
                     S_DONE = 4'd8;

    reg [3:0]        st;
    reg              phase;                 // 0 = latch operands, 1 = latch product
    reg [3:0]        iter;

    reg [X_W-1:0]    x_r;
    reg [MW-1:0]     Mm;
    reg [MW-1:0]     Yv;
    reg [MULW-1:0]   t1, t3;
    reg [PRW-1:0]    pr;
    reg signed [8:0] h;                     // e/2
    reg              zero_r;

    // ---- shared multiplier -------------------------------------------------
    reg  [MULW-1:0]  ma, mb;
    wire [PRW-1:0]   pmul = ma * mb;

    // ---- msb detection -----------------------------------------------------
    integer i;
    reg signed [8:0] p_c;
    always @* begin
        p_c = 9'sd0;
        for (i = 0; i < X_W; i = i + 1)
            if (x_r[i]) p_c = i[8:0];
    end

    wire             par_c = p_c[0] ^ X_F[0];              // parity of (p - X_F)
    wire signed [8:0] e_c  = p_c - X_F - {8'd0, par_c};    // even exponent
    wire signed [8:0] lsh  = (MF + {8'd0, par_c}) - p_c;   // mantissa alignment

    wire [TW-1:0]    x_ext = {{(TW-X_W){1'b0}}, x_r};
    wire [7:0]       lsh_a = lsh[8] ? (-lsh) : lsh;
    wire [TW-1:0]    x_al  = lsh[8] ? (x_ext >> lsh_a) : (x_ext << lsh_a);

    // ---- seed ROM ----------------------------------------------------------
    wire [15:0]      seed;
    rsqrt_seed_rom #(.LUT_BITS(LUT_BITS)) u_rom (
        .addr(Mm[MW-1 -: LUT_BITS]),
        .data(seed)
    );

    // ---- Newton-Raphson intermediates (combinational, from pmul) -----------
    wire [MULW-1:0]  t1_c   = pmul[MF +: MULW];                       // y^2
    wire [MULW-1:0]  t2_c   = pmul[MF +: MULW];                       // m*y^2
    wire [MULW-1:0]  three  = {{(MULW-MF-2){1'b0}}, 2'b11, {MF{1'b0}}};
    wire [MULW-1:0]  t3_c   = three - t2_c;
    wire [PRW-1:0]   ysh_c  = pmul >> (MF+1);
    wire [PRW-1:0]   one_mf = {{(PRW-MF-1){1'b0}}, 1'b1} << MF;      // 1.0 in Q.MF
    wire [MW-1:0]    y_next = (ysh_c > one_mf) ? one_mf[MW-1:0]      // clamp y <= 1
                                               : ysh_c[MW-1:0];

    // ---- denormalisation ---------------------------------------------------
    wire signed [8:0] y_sh   = (MF - Y_F) + h;
    wire [7:0]        y_sha  = y_sh[8] ? (-y_sh) : y_sh;
    wire [TW-1:0]     y_wide = {{(TW-MW){1'b0}}, Yv};
    wire [TW-1:0]     y_half = (y_sh[8] || y_sha == 8'd0)
                               ? {TW{1'b0}}
                               : ({{(TW-1){1'b0}}, 1'b1} << (y_sha - 8'd1));
    wire [TW-1:0]     y_res  = y_sh[8] ? (y_wide << y_sha)
                                       : ((y_wide + y_half) >> y_sha);
    wire              y_ovf  = |y_res[TW-1:Y_W-1];

    wire signed [8:0] s_sh   = (2*MF - S_F) - h;            // always > 0
    wire [7:0]        s_sha  = s_sh[7:0];
    wire [TW-1:0]     s_wide = {{(TW-PRW){1'b0}}, pr};
    wire [TW-1:0]     s_half = {{(TW-1){1'b0}}, 1'b1} << (s_sha - 8'd1);
    wire [TW-1:0]     s_res  = (s_wide + s_half) >> s_sha;
    wire              s_ovf  = |s_res[TW-1:S_W-1];

    // ---- control -----------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            st        <= S_IDLE;
            phase     <= 1'b0;
            iter      <= 4'd0;
            done      <= 1'b0;
            zero_flag <= 1'b0;
            sat_flag  <= 1'b0;
            y         <= {Y_W{1'b0}};
            s         <= {S_W{1'b0}};
        end else begin
            done <= 1'b0;
            case (st)
                // -------------------------------------------------------------
                S_IDLE: begin
                    phase <= 1'b0;
                    iter  <= 4'd0;
                    if (start) begin
                        x_r    <= x;
                        zero_r <= ~(|x);
                        st     <= S_NORM;
                    end
                end
                // -------------------------------------------------------------
                S_NORM: begin
                    Mm <= x_al[MW-1:0];
                    h  <= e_c >>> 1;
                    st <= S_SEED;
                end
                // -------------------------------------------------------------
                S_SEED: begin
                    Yv <= {seed, {(MF-15){1'b0}}};   // Q1.15 -> Q?.MF
                    st <= S_MUL1;
                end
                // -------------------------------------------------------------
                S_MUL1: begin
                    if (!phase) begin
                        ma <= {{(MULW-MW){1'b0}}, Yv};
                        mb <= {{(MULW-MW){1'b0}}, Yv};
                        phase <= 1'b1;
                    end else begin
                        t1    <= t1_c;
                        phase <= 1'b0;
                        st    <= S_MUL2;
                    end
                end
                // -------------------------------------------------------------
                S_MUL2: begin
                    if (!phase) begin
                        ma <= {{(MULW-MW){1'b0}}, Mm};
                        mb <= t1;
                        phase <= 1'b1;
                    end else begin
                        t3    <= t3_c;
                        phase <= 1'b0;
                        st    <= S_MUL3;
                    end
                end
                // -------------------------------------------------------------
                S_MUL3: begin
                    if (!phase) begin
                        ma <= {{(MULW-MW){1'b0}}, Yv};
                        mb <= t3;
                        phase <= 1'b1;
                    end else begin
                        Yv    <= y_next;
                        phase <= 1'b0;
                        iter  <= iter + 4'd1;
                        st    <= (iter == NR_ITER-1) ? S_SQRT : S_MUL1;
                    end
                end
                // -------------------------------------------------------------
                S_SQRT: begin
                    if (!phase) begin
                        ma <= {{(MULW-MW){1'b0}}, Mm};
                        mb <= {{(MULW-MW){1'b0}}, Yv};
                        phase <= 1'b1;
                    end else begin
                        pr    <= pmul;
                        phase <= 1'b0;
                        st    <= S_FIN;
                    end
                end
                // -------------------------------------------------------------
                S_FIN: begin
                    if (zero_r) begin
                        y <= {1'b0, {(Y_W-1){1'b1}}};
                        s <= {S_W{1'b0}};
                    end else begin
                        y <= y_ovf ? {1'b0, {(Y_W-1){1'b1}}} : y_res[Y_W-1:0];
                        s <= s_ovf ? {1'b0, {(S_W-1){1'b1}}} : s_res[S_W-1:0];
                    end
                    zero_flag <= zero_r;
                    sat_flag  <= zero_r | y_ovf | s_ovf;
                    st        <= S_DONE;
                end
                // -------------------------------------------------------------
                S_DONE: begin
                    done <= 1'b1;
                    st   <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
