module onehot_fsm (
    input wire clk,
    input wire rst,
    input wire step,
    output reg [3:0] state
);
    initial state = 4'b0001;
    always @(posedge clk) begin
        if (rst)
            state <= 4'b0001;
        else if (step)
            state <= {state[2:0], state[3]};   // rotate left
    end

    // Weak claim: bits 0 and 2 are never both set. True under one-hot,
    // but not inductive alone: state 4'b1010 satisfies it (bits 1,3)
    // and rotates into 4'b0101, which violates it. The hidden fact
    // that closes it: exactly one bit is ever set.
    always @(posedge clk)
        assert (!(state[0] && state[2]));
endmodule
