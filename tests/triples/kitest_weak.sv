module kitest (
    input wire i_clk,
    input wire i_reset,
    input wire i_ce,
    input wire i_in,
    output wire o_bit
);
    parameter LN = 32;

    reg [LN-1:0] sa, sb;
    initial sa = 0;
    initial sb = 0;

    always @(posedge i_clk)
        if (i_reset) begin
            sa <= 0;
            sb <= 0;
        end else if (i_ce) begin
            sa <= { sa[LN-2:0], i_in };
            sb <= { sb[LN-2:0], i_in };
        end

    assign o_bit = sa[LN-1] ^ sb[LN-1];

    // Weak claim: the two MSBs agree. Obviously true -- the registers
    // are byte-identical and shift the same input -- but not inductive
    // alone: it constrains only the top bits, so the solver may start
    // from a state where sa and sb differ in a low bit (legal per this
    // assertion, unreachable in reality) and shift the difference up to
    // the MSB. The hidden fact that closes it: sa == sb, supplied
    // separately as the strengthening invariant.
    // LN=32 against depth 20 keeps the weak version NOT_INDUCTIVE at
    // any k <= 20 (a window of 20 cannot push a low differing bit out),
    // avoiding the known depth-15-vs-16 sensitivity of the LN=16
    // original from ZipCPU's induction exercise.
    always @(*)
        assert (!o_bit);
endmodule
