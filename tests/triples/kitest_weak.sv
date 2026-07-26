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

    always @(*)
        assert (!o_bit);
endmodule
