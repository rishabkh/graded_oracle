module slow_factor (
    input wire clk,
    input wire [31:0] a,
    input wire [31:0] b
);
    // Exact 32x32->64 product (zero-extended, no mod-2^64 truncation:
    // with 64-bit operands any odd constant is "factorable" via the
    // modular inverse and the solver refutes this instantly).
    // Refuting now requires genuinely factoring (2^32-5)*(2^32-17),
    // a 64-bit semiprime; a,b > 1 excludes the trivial factors.
    // Intractable for bit-blasting => the sby timeout fires.
    wire [63:0] product = {32'd0, a} * {32'd0, b};

    always @(posedge clk)
        if (a > 32'd1 && b > 32'd1)
            assert (product != 64'd18446649584429071189);
endmodule
