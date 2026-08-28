// ===========================================================================
// gs_top.v -- Modified Gram-Schmidt orthogonaliser / thin QR decomposition
//
//   A (M x N, M >= N)  ->  Q (M x N, orthonormal columns) , R (N x N, upper)
//   with  A = Q * R
//
// ---------------------------------------------------------------------------
// ALGORITHM
//   Classical Gram-Schmidt subtracts all projections of the ORIGINAL a_k, so
//   rounding errors in q_0..q_{j-1} are never seen by later steps and the
//   computed Q loses orthogonality proportionally to kappa(A)^2.  This core
//   implements the MODIFIED recurrence, which subtracts each projection from
//   the RUNNING residual, giving a loss of orthogonality only ~ u*kappa(A):
//
//     v_k <- a_k                                for all k
//     for j = 0 .. N-1:
//         R[j][j] <- ||v_j||                    (NORM  + RSQRT)
//         q_j     <- v_j * (1/||v_j||)          (SCALE)
//         for k = j+1 .. N-1:
//             R[j][k] <- q_j . v_k              (PROJ)
//             v_k     <- v_k - R[j][k] * q_j    (UPD)
//
//   The update is done in place: column j of the vector memory holds a_j, then
//   the residual v_j, then finally q_j.  No second matrix buffer is needed.
//
// ---------------------------------------------------------------------------
// ARCHITECTURE
//
//                     +--------------------------------------+
//   s_a_* (A in) ---->| pack P:1 |                           |
//                     +----------+                           |
//                          |                                 |
//                    +-----v---------------------+           |
//                    |  V/Q memory  ram_2r1w     |           |
//                    |  DW = P*W, VDEPTH = CHUNKS*N          |
//                    +--+----------------+-------+           |
//                       | port A         | port B            |
//                  q_j /v_j              v_k                 |
//                       |                |                   |
//              +--------v----+    +------v------+            |
//              |  vec_mac    |    |  vec_axpy   |---> writeback
//              |  P lanes    |    |  P lanes    |
//              +------+------+    +------^------+
//                     | acc(ACC_W)       | scalar
//              +------v------+           |
//              | round_sat   |--> R[j][k]+
//              +------+------+           |
//                     | norm^2           |
//              +------v------+           |
//              | rsqrt_unit  |-----------+  1/||v||  (SC_W.SC_F)
//              |  NR, 1 mult |-----------+    ||v||  -> R[j][j]
//              +-------------+
//
//   P (lanes) trades area for time: each phase streams M/P memory words.
//   The MAC keeps a full ACC_W-bit accumulator so a dot product is exact up to
//   its guard bits; only the final write-back is rounded.
//
// ---------------------------------------------------------------------------
// CYCLE COST (CHUNKS = M/P, D = DRAIN)
//   load        : M*N
//   per column j: (CHUNKS+D)          norm
//               + L_rsqrt (~24)       1/||v||, ||v||
//               + (CHUNKS+D)          scale
//               + (N-1-j)*2*(CHUNKS+D)  project + update
//   unload      : M*N + N*N
//   total       ~ (M/P)*(N^2 + N) + N*(2D + 24) + 2*M*N + N^2
//
// ---------------------------------------------------------------------------
// FIXED-POINT CONTRACT
//   * data path is Q(W-F).F two's complement, saturating on every write-back
//   * scale A so that every column satisfies ||a_j|| < 2^(W-F-1); with the
//     defaults (Q2.16) that means ||a_j|| < 2, and |a_ij| <= 1/sqrt(M) is a
//     comfortable choice.  Q columns are unit norm and always representable.
//   * ACC_W must satisfy ACC_W >= 2*W + ceil(log2(M)) to keep dot products exact
//   * a column whose squared norm falls below EPS_NORM2 is declared dependent:
//     R[j][j] and q_j are forced to zero and rank_def is raised (sticky)
// ===========================================================================
`default_nettype none

module gs_top #(
    parameter integer M       = 8,    // rows      (vector length)
    parameter integer N       = 4,    // columns   (number of vectors), N <= M
    parameter integer P       = 2,    // lanes; M must be a multiple of P
    parameter integer W       = 18,   // datapath word width
    parameter integer F       = 16,   // datapath fractional bits
    parameter integer ACC_W   = 48,   // dot-product accumulator width
    parameter integer SC_W    = 32,   // scalar (1/norm) width
    parameter integer SC_F    = 20,   // scalar fractional bits
    parameter integer NR_ITER = 3,    // Newton-Raphson iterations in rsqrt
    // A column is declared linearly dependent when ||v||^2 < 2^EPS_SHIFT (the
    // accumulator carries 2*F fractional bits).  The default is the exact point
    // at which 1/||v|| stops fitting in the Q(SC_W-SC_F).SC_F scalar: beyond it
    // the normalisation would saturate and silently return a non-unit q.
    // Raise it to reject weaker columns, never lower it without widening SC_W.
    parameter integer EPS_SHIFT = 2*F - 2*(SC_W-1-SC_F)
)(
    input  wire             clk,
    input  wire             rst_n,

    // control ------------------------------------------------------------
    input  wire             start,      // pulse to begin a decomposition
    output reg              busy,
    output reg              done,       // 1-cycle pulse when R has drained
    output reg              rank_def,   // sticky: a column was ~dependent
    output reg              ovf_flag,   // sticky: a write-back saturated

    // A in, column-major, M*N beats ---------------------------------------
    input  wire [W-1:0]     s_a_tdata,
    input  wire             s_a_tvalid,
    output wire             s_a_tready,

    // Q out, column-major, M*N beats --------------------------------------
    output wire [W-1:0]     m_q_tdata,
    output wire             m_q_tvalid,
    input  wire             m_q_tready,

    // R out, row-major, N*N beats (zeros below the diagonal) ---------------
    output wire [W-1:0]     m_r_tdata,
    output wire             m_r_tvalid,
    input  wire             m_r_tready
);

    // ---------------- derived sizes ----------------------------------------
    function integer clog2;
        input integer v;
        integer i;
        begin
            clog2 = 0;
            for (i = v - 1; i > 0; i = i >> 1) clog2 = clog2 + 1;
        end
    endfunction

    localparam integer CHUNKS = M / P;
    localparam integer VDEPTH = CHUNKS * N;
    localparam integer VAW    = clog2(VDEPTH) < 1 ? 1 : clog2(VDEPTH);
    localparam integer RDEPTH = N * N;
    localparam integer RAW    = clog2(RDEPTH) < 1 ? 1 : clog2(RDEPTH);
    localparam integer CNTW   = clog2(CHUNKS + 1);   // counts chunks 0..CHUNKS-1
    localparam integer LCW    = clog2(P + 1);        // counts lanes  0..P-1
    localparam integer IDXW   = clog2(N + 1);
    localparam integer DW     = P * W;
    localparam integer DRAIN  = 6;    // >= RAM(1) + MAC(3) + 1 margin

    // squared-norm epsilon (see EPS_SHIFT above)
    localparam [ACC_W-1:0] EPS_NORM2 = {{(ACC_W-1){1'b0}}, 1'b1} << EPS_SHIFT;

    // ---------------- states -----------------------------------------------
    localparam [3:0] ST_IDLE    = 4'd0,
                     ST_RCLR    = 4'd1,
                     ST_LOAD    = 4'd2,
                     ST_NORM    = 4'd3,
                     ST_NORM_D  = 4'd4,
                     ST_RSQ     = 4'd5,
                     ST_SCALE   = 4'd6,
                     ST_SCALE_D = 4'd7,
                     ST_PROJ    = 4'd8,
                     ST_PROJ_D  = 4'd9,
                     ST_UPD     = 4'd10,
                     ST_UPD_D   = 4'd11,
                     ST_OUTQ    = 4'd12,
                     ST_OUTR    = 4'd13,
                     ST_FIN     = 4'd14;

    reg [3:0]        st;
    reg [CNTW-1:0]   cnt;
    reg [3:0]        dly;
    reg [IDXW-1:0]   j, k;
    reg [VAW-1:0]    base_j, base_k;
    reg [RAW-1:0]    rbase;

    // ---------------- vector memory ----------------------------------------
    reg              v_we;
    reg  [VAW-1:0]   v_waddr;
    reg  [DW-1:0]    v_wdata;
    reg  [VAW-1:0]   ra, rb;
    wire [DW-1:0]    v_da, v_db;

    ram_2r1w #(.DW(DW), .DEPTH(VDEPTH), .AW(VAW)) u_vmem (
        .clk(clk),
        .we(v_we), .waddr(v_waddr), .wdata(v_wdata),
        .raddr_a(ra), .rdata_a(v_da),
        .raddr_b(rb), .rdata_b(v_db)
    );

    // ---------------- R memory ---------------------------------------------
    reg              r_we;
    reg  [RAW-1:0]   r_waddr, r_raddr;
    reg  [W-1:0]     r_wdata;
    wire [W-1:0]     r_rdata;

    ram_1r1w #(.DW(W), .DEPTH(RDEPTH), .AW(RAW)) u_rmem (
        .clk(clk),
        .we(r_we), .waddr(r_waddr), .wdata(r_wdata),
        .raddr(r_raddr), .rdata(r_rdata)
    );

    // ---------------- datapath ---------------------------------------------
    reg                     iss_dot, iss_axpy;
    reg                     iss_dot_d, iss_axpy_d;
    reg                     mac_clr;
    reg                     use_dst_r;
    reg  signed [SC_W-1:0]  scalar;

    wire signed [ACC_W-1:0] acc;
    wire [DW-1:0]           axpy_q;
    wire                    axpy_vld, axpy_ovf;

    vec_mac #(.W(W), .P(P), .ACC_W(ACC_W)) u_mac (
        .clk(clk), .rst_n(rst_n),
        .clr(mac_clr), .en(iss_dot_d),
        .a(v_da), .b(v_db), .acc(acc)
    );

    vec_axpy #(.W(W), .P(P), .SC_W(SC_W), .SC_F(SC_F)) u_axpy (
        .clk(clk), .rst_n(rst_n),
        .en(iss_axpy_d), .use_dst(use_dst_r), .scalar(scalar),
        .src(v_da), .dst_in(v_db),
        .dst_out(axpy_q), .vld(axpy_vld), .ovf(axpy_ovf)
    );

    // accumulator -> Q(W-F).F  (accumulator carries 2F fractional bits)
    wire signed [W-1:0] acc_q;
    wire                acc_ovf;
    round_sat #(.IN_W(ACC_W), .OUT_W(W), .SHIFT(F)) u_accq (
        .din(acc), .dout(acc_q), .ovf(acc_ovf)
    );

    // ---------------- reciprocal square root --------------------------------
    reg  [ACC_W-1:0] norm2;
    reg              rsq_start;
    wire [SC_W-1:0]  rsq_y;
    wire [W-1:0]     rsq_s;
    wire             rsq_done, rsq_zero, rsq_sat;

    rsqrt_unit #(
        .X_W(ACC_W), .X_F(2*F),
        .Y_W(SC_W),  .Y_F(SC_F),
        .S_W(W),     .S_F(F),
        .NR_ITER(NR_ITER)
    ) u_rsqrt (
        .clk(clk), .rst_n(rst_n),
        .start(rsq_start), .x(norm2),
        .y(rsq_y), .s(rsq_s),
        .done(rsq_done), .zero_flag(rsq_zero), .sat_flag(rsq_sat)
    );

    // ---------------- write-address delay line ------------------------------
    // read addr -> RAM(1) -> axpy stage1(1) -> axpy stage2(1) == 3 registers
    reg           wsel_upd;
    reg [VAW-1:0] wa1, wa2, wa3;
    always @(posedge clk) begin
        wa1 <= wsel_upd ? rb : ra;
        wa2 <= wa1;
        wa3 <= wa2;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            iss_dot_d  <= 1'b0;
            iss_axpy_d <= 1'b0;
        end else begin
            iss_dot_d  <= iss_dot;
            iss_axpy_d <= iss_axpy;
        end
    end

    // ---------------- input packing (P beats -> 1 memory word) --------------
    reg [DW-1:0]   pack;
    reg [LCW-1:0]  pack_cnt;
    reg [VAW-1:0]  load_addr;
    reg [RAW-1:0]  rclr_addr;

    wire [DW-1:0] pack_next = (DW == W)
                            ? {{(DW-W){1'b0}}, s_a_tdata}
                            : ((pack >> W) | ({{(DW-W){1'b0}}, s_a_tdata} << (DW-W)));

    wire load_beat = (st == ST_LOAD) && s_a_tvalid;
    wire load_we   = load_beat && (pack_cnt == P-1);

    assign s_a_tready = (st == ST_LOAD);

    // ---------------- memory write mux --------------------------------------
    always @* begin
        if (st == ST_LOAD) begin
            v_we    = load_we;
            v_waddr = load_addr;
            v_wdata = pack_next;
        end else begin
            v_we    = axpy_vld;
            v_waddr = wa3;
            v_wdata = axpy_q;
        end
    end

    // ---------------- output streaming --------------------------------------
    reg [1:0]     oq_sub, or_sub;
    reg [VAW-1:0] out_word;
    reg [LCW-1:0] out_el;
    reg [DW-1:0]  outq_reg;
    reg [RAW-1:0] r_out;
    reg [W-1:0]   r_out_reg;

    assign m_q_tvalid = (st == ST_OUTQ) && (oq_sub == 2'd3);
    assign m_q_tdata  = outq_reg[W-1:0];
    assign m_r_tvalid = (st == ST_OUTR) && (or_sub == 2'd3);
    assign m_r_tdata  = r_out_reg;

    // =======================================================================
    // main sequencer
    // =======================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            st        <= ST_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            rank_def  <= 1'b0;
            ovf_flag  <= 1'b0;
            iss_dot   <= 1'b0;
            iss_axpy  <= 1'b0;
            mac_clr   <= 1'b0;
            r_we      <= 1'b0;
            rsq_start <= 1'b0;
            wsel_upd  <= 1'b0;
            use_dst_r <= 1'b0;
            scalar    <= {SC_W{1'b0}};
            cnt       <= {CNTW{1'b0}};
            dly       <= 4'd0;
            j         <= {IDXW{1'b0}};
            k         <= {IDXW{1'b0}};
            base_j    <= {VAW{1'b0}};
            base_k    <= {VAW{1'b0}};
            rbase     <= {RAW{1'b0}};
            pack_cnt  <= {LCW{1'b0}};
            load_addr <= {VAW{1'b0}};
            rclr_addr <= {RAW{1'b0}};
            oq_sub    <= 2'd0;
            or_sub    <= 2'd0;
        end else begin
            // one-shot defaults
            done      <= 1'b0;
            r_we      <= 1'b0;
            rsq_start <= 1'b0;
            if (axpy_ovf) ovf_flag <= 1'b1;

            case (st)
            // ---------------------------------------------------------------
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy      <= 1'b1;
                    rank_def  <= 1'b0;
                    ovf_flag  <= 1'b0;
                    rclr_addr <= {RAW{1'b0}};
                    st        <= ST_RCLR;
                end
            end

            // ---- zero the R store so the strict lower triangle reads 0 -----
            ST_RCLR: begin
                r_we    <= 1'b1;
                r_waddr <= rclr_addr;
                r_wdata <= {W{1'b0}};
                if (rclr_addr == RDEPTH-1) begin
                    load_addr <= {VAW{1'b0}};
                    pack_cnt  <= {LCW{1'b0}};
                    st        <= ST_LOAD;
                end else begin
                    rclr_addr <= rclr_addr + 1'b1;
                end
            end

            // ---- stream A in, column-major ---------------------------------
            ST_LOAD: begin
                if (load_beat) begin
                    pack <= pack_next;
                    if (pack_cnt == P-1) begin
                        pack_cnt <= {LCW{1'b0}};
                        if (load_addr == VDEPTH-1) begin
                            j       <= {IDXW{1'b0}};
                            base_j  <= {VAW{1'b0}};
                            rbase   <= {RAW{1'b0}};
                            cnt     <= {CNTW{1'b0}};
                            mac_clr <= 1'b1;
                            st      <= ST_NORM;
                        end else begin
                            load_addr <= load_addr + 1'b1;
                        end
                    end else begin
                        pack_cnt <= pack_cnt + 1'b1;
                    end
                end
            end

            // ---- ||v_j||^2 -------------------------------------------------
            ST_NORM: begin
                mac_clr <= 1'b0;
                ra      <= base_j + cnt;
                rb      <= base_j + cnt;
                iss_dot <= 1'b1;
                if (cnt == CHUNKS-1) begin
                    cnt <= {CNTW{1'b0}};
                    dly <= 4'd0;
                    st  <= ST_NORM_D;
                end else begin
                    cnt <= cnt + 1'b1;
                end
            end

            ST_NORM_D: begin
                iss_dot <= 1'b0;
                if (dly == DRAIN-1) begin
                    norm2     <= acc;
                    rsq_start <= 1'b1;
                    st        <= ST_RSQ;
                end else begin
                    dly <= dly + 1'b1;
                end
            end

            // ---- 1/||v_j|| and ||v_j|| -------------------------------------
            ST_RSQ: begin
                if (rsq_done) begin
                    r_we      <= 1'b1;
                    r_waddr   <= rbase + j;
                    use_dst_r <= 1'b0;
                    wsel_upd  <= 1'b0;
                    cnt       <= {CNTW{1'b0}};
                    st        <= ST_SCALE;
                    if (norm2 < EPS_NORM2) begin       // rank deficient column
                        rank_def <= 1'b1;
                        scalar   <= {SC_W{1'b0}};
                        r_wdata  <= {W{1'b0}};
                    end else begin
                        scalar   <= rsq_y;
                        r_wdata  <= rsq_s;
                        if (rsq_sat) ovf_flag <= 1'b1;
                    end
                end
            end

            // ---- q_j = v_j * (1/||v_j||) -----------------------------------
            ST_SCALE: begin
                ra       <= base_j + cnt;
                rb       <= base_j + cnt;
                iss_axpy <= 1'b1;
                if (cnt == CHUNKS-1) begin
                    cnt <= {CNTW{1'b0}};
                    dly <= 4'd0;
                    st  <= ST_SCALE_D;
                end else begin
                    cnt <= cnt + 1'b1;
                end
            end

            ST_SCALE_D: begin
                iss_axpy <= 1'b0;
                if (dly == DRAIN-1) begin
                    if (j == N-1) begin
                        out_word <= {VAW{1'b0}};
                        out_el   <= {LCW{1'b0}};
                        oq_sub   <= 2'd0;
                        st       <= ST_OUTQ;
                    end else begin
                        k       <= j + 1'b1;
                        base_k  <= base_j + CHUNKS;
                        cnt     <= {CNTW{1'b0}};
                        mac_clr <= 1'b1;
                        st      <= ST_PROJ;
                    end
                end else begin
                    dly <= dly + 1'b1;
                end
            end

            // ---- R[j][k] = q_j . v_k ---------------------------------------
            ST_PROJ: begin
                mac_clr <= 1'b0;
                ra      <= base_j + cnt;
                rb      <= base_k + cnt;
                iss_dot <= 1'b1;
                if (cnt == CHUNKS-1) begin
                    cnt <= {CNTW{1'b0}};
                    dly <= 4'd0;
                    st  <= ST_PROJ_D;
                end else begin
                    cnt <= cnt + 1'b1;
                end
            end

            ST_PROJ_D: begin
                iss_dot <= 1'b0;
                if (dly == DRAIN-1) begin
                    r_we      <= 1'b1;
                    r_waddr   <= rbase + k;
                    r_wdata   <= acc_q;
                    // Q(W-F).F  ->  Q(SC_W-SC_F).SC_F
                    scalar    <= {{(SC_W-W-(SC_F-F)){acc_q[W-1]}}, acc_q, {(SC_F-F){1'b0}}};
                    if (acc_ovf) ovf_flag <= 1'b1;
                    use_dst_r <= 1'b1;
                    wsel_upd  <= 1'b1;
                    cnt       <= {CNTW{1'b0}};
                    st        <= ST_UPD;
                end else begin
                    dly <= dly + 1'b1;
                end
            end

            // ---- v_k <- v_k - R[j][k]*q_j ----------------------------------
            ST_UPD: begin
                ra       <= base_j + cnt;
                rb       <= base_k + cnt;
                iss_axpy <= 1'b1;
                if (cnt == CHUNKS-1) begin
                    cnt <= {CNTW{1'b0}};
                    dly <= 4'd0;
                    st  <= ST_UPD_D;
                end else begin
                    cnt <= cnt + 1'b1;
                end
            end

            ST_UPD_D: begin
                iss_axpy <= 1'b0;
                if (dly == DRAIN-1) begin
                    mac_clr <= 1'b1;
                    cnt     <= {CNTW{1'b0}};
                    if (k == N-1) begin
                        j      <= j + 1'b1;
                        base_j <= base_j + CHUNKS;
                        rbase  <= rbase + N;
                        st     <= ST_NORM;
                    end else begin
                        k      <= k + 1'b1;
                        base_k <= base_k + CHUNKS;
                        st     <= ST_PROJ;
                    end
                end else begin
                    dly <= dly + 1'b1;
                end
            end

            // ---- stream Q out ----------------------------------------------
            ST_OUTQ: begin
                case (oq_sub)
                2'd0: begin ra <= out_word; oq_sub <= 2'd1; end
                2'd1: begin oq_sub <= 2'd2; end
                2'd2: begin
                    outq_reg <= v_da;
                    out_el   <= {LCW{1'b0}};
                    oq_sub   <= 2'd3;
                end
                default: begin
                    if (m_q_tready) begin
                        outq_reg <= outq_reg >> W;
                        if (out_el == P-1) begin
                            if (out_word == VDEPTH-1) begin
                                r_out  <= {RAW{1'b0}};
                                or_sub <= 2'd0;
                                st     <= ST_OUTR;
                            end else begin
                                out_word <= out_word + 1'b1;
                                oq_sub   <= 2'd0;
                            end
                        end else begin
                            out_el <= out_el + 1'b1;
                        end
                    end
                end
                endcase
            end

            // ---- stream R out ----------------------------------------------
            ST_OUTR: begin
                case (or_sub)
                2'd0: begin r_raddr <= r_out; or_sub <= 2'd1; end
                2'd1: begin or_sub <= 2'd2; end
                2'd2: begin r_out_reg <= r_rdata; or_sub <= 2'd3; end
                default: begin
                    if (m_r_tready) begin
                        if (r_out == RDEPTH-1) begin
                            st <= ST_FIN;
                        end else begin
                            r_out  <= r_out + 1'b1;
                            or_sub <= 2'd0;
                        end
                    end
                end
                endcase
            end

            ST_FIN: begin
                done <= 1'b1;
                busy <= 1'b0;
                st   <= ST_IDLE;
            end

            default: st <= ST_IDLE;
            endcase
        end
    end

    // ---------------- elaboration-time checks --------------------------------
    initial begin
        if (M % P != 0)
            $display("gs_top ERROR: M (%0d) must be a multiple of P (%0d)", M, P);
        if (N > M)
            $display("gs_top ERROR: N (%0d) must be <= M (%0d)", N, M);
        if (ACC_W < 2*W + clog2(M))
            $display("gs_top ERROR: ACC_W (%0d) < 2*W + log2(M) (%0d)", ACC_W, 2*W + clog2(M));
        if (SC_F < F)
            $display("gs_top ERROR: SC_F (%0d) must be >= F (%0d)", SC_F, F);
    end

endmodule

`default_nettype wire
