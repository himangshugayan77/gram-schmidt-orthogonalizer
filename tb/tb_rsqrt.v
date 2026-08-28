// ---------------------------------------------------------------------------
// tb_rsqrt.v : self-checking testbench for rsqrt_unit.
//
// Sweeps x over ~13 octaves plus targeted corner cases and checks BOTH outputs
// against a real-valued reference.  The pass criterion is
//
//     |got - exact|  <=  max( TOL_REL * exact , TOL_LSB * 2^-frac )
//
// i.e. an error is only counted when it exceeds both the algorithmic tolerance
// and the unavoidable quantisation of the output format.  Reporting relative
// error alone would flag every small sqrt result, since sqrt(x) is delivered in
// Q2.16 where 0.01 is only ~650 LSBs.
//
// NOTE: $itor() truncates its argument to 32 bits, so wide fixed-point values
// are converted with the explicit 16-bit-limbed to_real() function below.
// ---------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_rsqrt;

    localparam integer X_W = 48, X_F = 32;
    localparam integer Y_W = 32, Y_F = 24;
    localparam integer S_W = 18, S_F = 16;

    real TOL_REL, TOL_LSB;
    initial begin TOL_REL = 1.0e-5; TOL_LSB = 1.5; end

    reg               clk = 1'b0;
    reg               rst_n = 1'b0;
    reg               start = 1'b0;
    reg  [X_W-1:0]    x = 0;
    wire [Y_W-1:0]    y;
    wire [S_W-1:0]    s;
    wire              done, zero_flag, sat_flag;

    integer errors = 0;
    integer tests  = 0;
    integer cycles_max = 0;
    real    worst_y_ulp, worst_s_ulp;

    always #5 clk = ~clk;

    rsqrt_unit #(
        .X_W(X_W), .X_F(X_F), .Y_W(Y_W), .Y_F(Y_F),
        .S_W(S_W), .S_F(S_F), .NR_ITER(3)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .x(x),
        .y(y), .s(s), .done(done), .zero_flag(zero_flag), .sat_flag(sat_flag)
    );

    // ---- wide unsigned -> real (16-bit limbs; $itor truncates to 32b) -------
    function real to_real;
        input [63:0] v;
        integer i;
        real acc;
        begin
            acc = 0.0;
            for (i = 3; i >= 0; i = i - 1)
                acc = acc * 65536.0 + $itor(v[i*16 +: 16]);
            to_real = acc;
        end
    endfunction

    function real fabs;
        input real v;
        begin fabs = (v < 0.0) ? -v : v; end
    endfunction

    // -----------------------------------------------------------------------
    task run_one;
        input [X_W-1:0] xv;
        real xr, yr, sr, yexp, sexp, dy, ds, toly, tols;
        integer cyc;
        begin
            @(negedge clk);
            x     = xv;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            cyc = 0;
            while (!done && cyc < 500) begin
                @(negedge clk);
                cyc = cyc + 1;
            end
            if (cyc >= 500) begin
                $display("TIMEOUT on x=%h", xv);
                errors = errors + 1;
            end else begin
                tests = tests + 1;
                if (cyc > cycles_max) cycles_max = cyc;
                xr = to_real({16'd0, xv}) / (2.0 ** X_F);
                yr = to_real({32'd0, y})  / (2.0 ** Y_F);
                sr = to_real({46'd0, s})  / (2.0 ** S_F);
                if (xv == 0) begin
                    if (!zero_flag) begin
                        $display("FAIL: x=0 did not raise zero_flag");
                        errors = errors + 1;
                    end
                end else begin
                    yexp = 1.0 / $sqrt(xr);
                    sexp = $sqrt(xr);
                    // score only points representable in the output formats
                    if (yexp < 100.0 && sexp < 1.9) begin
                        dy   = fabs(yr - yexp);
                        ds   = fabs(sr - sexp);
                        toly = TOL_REL * yexp;
                        if (toly < TOL_LSB / (2.0**Y_F)) toly = TOL_LSB / (2.0**Y_F);
                        tols = TOL_REL * sexp;
                        if (tols < TOL_LSB / (2.0**S_F)) tols = TOL_LSB / (2.0**S_F);
                        if (dy * (2.0**Y_F) > worst_y_ulp) worst_y_ulp = dy * (2.0**Y_F);
                        if (ds * (2.0**S_F) > worst_s_ulp) worst_s_ulp = ds * (2.0**S_F);
                        if (dy > toly || ds > tols) begin
                            $display("FAIL x=%h (%e)  y=%e exp %e (%0.2f ulp)  s=%e exp %e (%0.2f ulp)",
                                     xv, xr, yr, yexp, dy*(2.0**Y_F), sr, sexp, ds*(2.0**S_F));
                            errors = errors + 1;
                        end
                        if (sat_flag) begin
                            $display("FAIL: unexpected sat_flag at x=%h", xv);
                            errors = errors + 1;
                        end
                    end
                end
            end
        end
    endtask

    // -----------------------------------------------------------------------
    integer k;
    reg [X_W-1:0] xv;
    initial begin
        worst_y_ulp = 0.0; worst_s_ulp = 0.0;
        if ($test$plusargs("vcd")) begin
            $dumpfile("tb_rsqrt.vcd");
            $dumpvars(0, tb_rsqrt);
        end
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // corner cases -------------------------------------------------------
        run_one(48'd0);                       // zero -> zero_flag
        run_one(48'd1 << X_F);                // 1.0
        run_one(48'd1 << (X_F+1));            // 2.0
        run_one(48'd1 << (X_F+2));            // 4.0  (sqrt saturates, unscored)
        run_one(48'd1 << (X_F-1));            // 0.5
        run_one(48'd1 << (X_F-2));            // 0.25
        run_one(48'd1);                       // 2^-32 (unscored, saturating)
        run_one({X_W{1'b1}});                 // max   (unscored)

        // octave sweep, 3 mantissas per octave --------------------------------
        for (k = 0; k < 36; k = k + 1) begin
            xv = 48'd1 << (k + 6);
            run_one(xv);
            run_one(xv + (xv >> 2));                    // x * 1.25
            run_one(xv + (xv >> 1) + (xv >> 3));        // x * 1.625
        end

        // pseudo-random sweep over the useful ||v||^2 range --------------------
        for (k = 0; k < 500; k = k + 1) begin
            xv = ({$random} % (48'd1 << 30)) + (48'd1 << 16);
            run_one(xv);
        end

        $display("--------------------------------------------------");
        $display("tb_rsqrt : %0d tests, %0d errors, max latency %0d cycles",
                 tests, errors, cycles_max);
        $display("           worst error : 1/sqrt %0.3f ulp   sqrt %0.3f ulp",
                 worst_y_ulp, worst_s_ulp);
        if (errors == 0) $display("TEST PASSED");
        else             $display("TEST FAILED");
        $display("--------------------------------------------------");
        $finish;
    end

endmodule
