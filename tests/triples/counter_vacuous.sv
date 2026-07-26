module counter_vac (
    input wire clk,
    input wire rst,
    output reg [3:0] count
);
    initial count = 0;
    always @(posedge clk) begin
        if (rst)
            count <= 4'd0;
        else
            count <= count + 1;
    end

    // Antecedent can never fire: a 4-bit count never equals 20.
    // Passes k-induction while proving nothing.
    always @(posedge clk) begin
        if (!rst) begin
            if (count == 5'd20)
                assert (count == 7'd99);
        end
    end
endmodule
