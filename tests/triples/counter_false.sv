module counter_false (
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

    always @(posedge clk)
        assert (count != 4'd5);   // genuinely false: reachable in 5 steps
endmodule
