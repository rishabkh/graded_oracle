module counter (
    input wire clk,
    input wire rst,
    output reg [3:0] count
);
    always @(posedge clk) begin
        if (rst)
            count <= 4'd0;
        else
            count <= count + 1;
    end

    reg past_rst = 0;
    always @(posedge clk) past_rst <= rst;

    always @(posedge clk) begin
        if (past_rst)
            assert (count == 4'd0);
    end
endmodule
