// ---------------------------------------------------------------------------
// tb_gs_top.v : self-checking testbench for the Gram-Schmidt QR engine.
//
// Reads a.hex (A, column-major) plus exp_q.hex / exp_r.hex produced by the
// bit-accurate Python model, drives the input stream with random bubbles and
// the output streams with random back-pressure, and checks
//
//   1. every Q and R beat matches the model EXACTLY (bit for bit)
//   2. the correct number of beats is produced
//   3. max |Q'Q - I| and max |A - QR| are within numerical limits
//   4. R is upper triangular with a non-negative diagonal
//   5. a second back-to-back run gives identical results (no sticky state)
//
// Run from the directory holding the .hex files.  Override the geometry with
// -DGS_M= -DGS_N= -DGS_P=.
// ---------------------------------------------------------------------------
`timescale 1ns/1ps

`ifndef GS_M
 `define GS_M 8
`endif
`ifndef GS_N
 `define GS_N 4
`endif
`ifndef GS_P
 `define GS_P 2
`endif
`ifndef GS_W
 `define GS_W 18
`endif
`ifndef GS_F
 `define GS_F 16
`endif
`ifndef GS_RUNS
 `define GS_RUNS 2
`endif
`ifndef GS_TOL_LSB
 `define GS_TOL_LSB 64
`endif

module tb_gs_top;

    localparam integer M = `GS_M;
    localparam integer N = `GS_N;
    localparam integer P = `GS_P;
    localparam integer W = `GS_W;
    localparam integer F = `GS_F;

    localparam integer NA = M*N;
    localparam integer NR = N*N;

    // tolerances on the real-valued metrics (LSBs of the Q(W-F).F format)
    real TOL_ORTHO, TOL_RECON;
    initial begin
        // MGS loses orthogonality proportionally to u*cond(A), so the caller
        // passes a condition-number-aware budget via -DGS_TOL_LSB.
        TOL_ORTHO = `GS_TOL_LSB / (2.0**F);
        TOL_RECON = `GS_TOL_LSB / (2.0**F);
    end

    reg clk = 1'b0, rst_n = 1'b0, start = 1'b0;
    always #5 clk = ~clk;

    wire              busy, done, rank_def, ovf_flag;
    wire [W-1:0]      s_a_tdata;
    reg               s_a_tvalid;
    wire              s_a_tready;
    wire [W-1:0]      m_q_tdata, m_r_tdata;
    wire              m_q_tvalid, m_r_tvalid;
    reg               m_q_tready, m_r_tready;

    // ---- storage -----------------------------------------------------------
    reg [W-1:0] a_mem [0:NA-1];
    reg [W-1:0] expq  [0:NA-1];
    reg [W-1:0] expr  [0:NR-1];
    reg [W-1:0] gotq  [0:NA-1];
    reg [W-1:0] gotr  [0:NR-1];

    integer in_idx, q_idx, r_idx;
    reg     idx_clr;
    integer errors = 0;
    integer run;

    // ---- DUT ---------------------------------------------------------------
    gs_top #(.M(M), .N(N), .P(P), .W(W), .F(F)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .busy(busy), .done(done),
        .rank_def(rank_def), .ovf_flag(ovf_flag),
        .s_a_tdata(s_a_tdata), .s_a_tvalid(s_a_tvalid), .s_a_tready(s_a_tready),
        .m_q_tdata(m_q_tdata), .m_q_tvalid(m_q_tvalid), .m_q_tready(m_q_tready),
        .m_r_tdata(m_r_tdata), .m_r_tvalid(m_r_tvalid), .m_r_tready(m_r_tready)
    );

    // ---- input driver: hold tvalid until accepted, random bubbles ----------
    assign s_a_tdata = (in_idx < NA) ? a_mem[in_idx] : {W{1'b0}};

    // NOTE: both stream drivers are clocked on the SAME edge the DUT samples.
    // Evaluating tvalid/tready on the opposite edge silently miscounts a beat
    // whenever tready changes on a clock edge (e.g. when the DUT enters LOAD).
    integer nxt_idx;
    always @(posedge clk) begin
        if (!rst_n || idx_clr) begin
            in_idx     <= 0;
            s_a_tvalid <= 1'b0;
        end else begin
            nxt_idx = (s_a_tvalid && s_a_tready) ? in_idx + 1 : in_idx;
            in_idx  <= nxt_idx;
            if (s_a_tvalid && !s_a_tready)
                s_a_tvalid <= 1'b1;                       // must hold
            else
`ifdef GS_NOSTALL
                s_a_tvalid <= (nxt_idx < NA);
`else
                s_a_tvalid <= (nxt_idx < NA) && (($random & 7) != 0);
`endif
        end
    end

    // ---- output collectors with random back-pressure ----------------------
    always @(posedge clk) begin
        if (!rst_n || idx_clr) begin
            q_idx      <= 0;
            r_idx      <= 0;
            m_q_tready <= 1'b0;
            m_r_tready <= 1'b0;
        end else begin
            if (m_q_tvalid && m_q_tready) begin
                if (q_idx < NA) gotq[q_idx] <= m_q_tdata;
                q_idx <= q_idx + 1;
            end
            if (m_r_tvalid && m_r_tready) begin
                if (r_idx < NR) gotr[r_idx] <= m_r_tdata;
                r_idx <= r_idx + 1;
            end
`ifdef GS_NOSTALL
            m_q_tready <= 1'b1;
            m_r_tready <= 1'b1;
`else
            m_q_tready <= (($random & 3) != 0);
            m_r_tready <= (($random & 3) != 0);
`endif
        end
    end

    // ---- helpers -----------------------------------------------------------
    function real q2r;
        input [W-1:0] v;
        reg signed [31:0] sv;
        begin
            sv  = {{(32-W){v[W-1]}}, v};
            q2r = $itor(sv) / (2.0 ** F);
        end
    endfunction

    function real fabs;
        input real v;
        begin fabs = (v < 0.0) ? -v : v; end
    endfunction

    // ---- numeric quality checks --------------------------------------------
    task check_numeric;
        integer i, jj, kk;
        real acc, worst_o, worst_r, diag;
        begin
            worst_o = 0.0;
            worst_r = 0.0;
            // Q'Q - I
            for (jj = 0; jj < N; jj = jj + 1)
                for (kk = 0; kk < N; kk = kk + 1) begin
                    acc = 0.0;
                    for (i = 0; i < M; i = i + 1)
                        acc = acc + q2r(gotq[jj*M+i]) * q2r(gotq[kk*M+i]);
                    if (jj == kk) acc = acc - 1.0;
                    if (fabs(acc) > worst_o) worst_o = fabs(acc);
                end
            // A - QR   (A[i][jj] = a_mem[jj*M+i], R[kk][jj] = gotr[kk*N+jj])
            for (i = 0; i < M; i = i + 1)
                for (jj = 0; jj < N; jj = jj + 1) begin
                    acc = 0.0;
                    for (kk = 0; kk < N; kk = kk + 1)
                        acc = acc + q2r(gotq[kk*M+i]) * q2r(gotr[kk*N+jj]);
                    acc = acc - q2r(a_mem[jj*M+i]);
                    if (fabs(acc) > worst_r) worst_r = fabs(acc);
                end

            $display("    max |Q'Q - I| = %0.3e  (%0.1f LSB)", worst_o, worst_o*(2.0**F));
            $display("    max |A - QR|  = %0.3e  (%0.1f LSB)", worst_r, worst_r*(2.0**F));

            if (!rank_def) begin
                if (worst_o > TOL_ORTHO) begin
                    $display("    FAIL: orthogonality out of tolerance");
                    errors = errors + 1;
                end
                if (worst_r > TOL_RECON) begin
                    $display("    FAIL: reconstruction out of tolerance");
                    errors = errors + 1;
                end
            end

            // R strictly-lower triangle must be zero, diagonal non-negative
            for (jj = 0; jj < N; jj = jj + 1) begin
                for (kk = 0; kk < jj; kk = kk + 1)
                    if (gotr[jj*N+kk] !== {W{1'b0}}) begin
                        $display("    FAIL: R[%0d][%0d] = %h, expected 0", jj, kk, gotr[jj*N+kk]);
                        errors = errors + 1;
                    end
                diag = q2r(gotr[jj*N+jj]);
                if (diag < 0.0) begin
                    $display("    FAIL: R[%0d][%0d] negative", jj, jj);
                    errors = errors + 1;
                end
            end
        end
    endtask

    // ---- exact comparison against the bit-accurate model -------------------
    task check_exact;
        integer i, bad;
        begin
            bad = 0;
            if (q_idx !== NA) begin
                $display("    FAIL: got %0d Q beats, expected %0d", q_idx, NA);
                errors = errors + 1;
            end
            if (r_idx !== NR) begin
                $display("    FAIL: got %0d R beats, expected %0d", r_idx, NR);
                errors = errors + 1;
            end
            for (i = 0; i < NA; i = i + 1)
                if (gotq[i] !== expq[i]) begin
                    if (bad < 8)
                        $display("    FAIL: Q[%0d] = %h, model says %h", i, gotq[i], expq[i]);
                    bad = bad + 1;
                end
            for (i = 0; i < NR; i = i + 1)
                if (gotr[i] !== expr[i]) begin
                    if (bad < 8)
                        $display("    FAIL: R[%0d] = %h, model says %h", i, gotr[i], expr[i]);
                    bad = bad + 1;
                end
            if (bad != 0) begin
                $display("    FAIL: %0d words differ from the bit-accurate model", bad);
                errors = errors + 1;
            end else begin
                $display("    exact match against bit-accurate model (%0d words)", NA+NR);
            end
        end
    endtask

    // ---- dump results for the Python checker -------------------------------
    task dump_files;
        integer fq, fr, i;
        begin
            fq = $fopen("q.hex", "w");
            for (i = 0; i < NA; i = i + 1) $fdisplay(fq, "%h", gotq[i]);
            $fclose(fq);
            fr = $fopen("r.hex", "w");
            for (i = 0; i < NR; i = i + 1) $fdisplay(fr, "%h", gotr[i]);
            $fclose(fr);
        end
    endtask

    // ---- main --------------------------------------------------------------
    integer cyc;
    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("tb_gs_top.vcd");
            $dumpvars(0, tb_gs_top);
        end
        $readmemh("a.hex",     a_mem);
        $readmemh("exp_q.hex", expq);
        $readmemh("exp_r.hex", expr);

        in_idx = 0; q_idx = 0; r_idx = 0; idx_clr = 1'b0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (run = 1; run <= `GS_RUNS; run = run + 1) begin
            $display("--- run %0d : M=%0d N=%0d P=%0d W=%0d F=%0d", run, M, N, P, W, F);
            idx_clr = 1'b1;
            @(negedge clk);
            idx_clr = 1'b0;
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            cyc = 0;
            while (!done && cyc < 200000) begin
                @(negedge clk);
                cyc = cyc + 1;
            end
            if (cyc >= 200000) begin
                $display("    FAIL: timeout (busy=%b)", busy);
                errors = errors + 1;
                run = `GS_RUNS;
            end else begin
                $display("    completed in %0d cycles  (rank_def=%b ovf=%b)", cyc, rank_def, ovf_flag);
                check_exact;
                check_numeric;
            end
            @(negedge clk);
        end

        dump_files;

        $display("==================================================");
        if (errors == 0) $display("TEST PASSED");
        else             $display("TEST FAILED (%0d errors)", errors);
        $display("==================================================");
        $finish;
    end

endmodule
