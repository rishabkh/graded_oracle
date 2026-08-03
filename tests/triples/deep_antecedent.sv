module deep_antecedent (
    input wire clk,
    input wire rst,
    output reg [5:0] count
);
    initial count = 0;
    always @(posedge clk) begin
        if (rst)
            count <= 6'd0;
        else
            count <= count + 1;
    end

    // The antecedent first fires at cycle 40 — beyond a depth-20 cover
    // probe. A bounded vacuity check falsely discards this triple as
    // VACUOUS; the unbounded PDR check finds the reachability witness.
    always @(posedge clk) begin
        if (count == 6'd40)
            assert (count[0] == 1'b0);   // 40 is even: genuinely true
    end
endmodule
