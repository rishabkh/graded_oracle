module fifo (
    input wire clk,
    input wire rst,
    input wire push,
    input wire pop,
    output reg [2:0] count
);
    reg [1:0] wptr, rptr;
    initial begin wptr = 0; rptr = 0; count = 0; end

    wire do_push = push && count != 3'd4;
    wire do_pop  = pop  && count != 3'd0;

    always @(posedge clk) begin
        if (rst) begin
            wptr <= 0; rptr <= 0; count <= 0;
        end else begin
            if (do_push) wptr <= wptr + 1;
            if (do_pop)  rptr <= rptr + 1;
            case ({do_push, do_pop})
                2'b10:   count <= count + 1;
                2'b01:   count <= count - 1;
                default: count <= count;
            endcase
        end
    end

    // Weak claim: empty means the pointers agree. True by construction,
    // but not inductive alone: from count==1 with disagreeing pointers
    // (unreachable, yet legal per this assertion), one pop reaches
    // count==0 with wptr != rptr. The hidden facts that close it:
    //   count <= 4  and  count[1:0] == wptr - rptr
    always @(posedge clk)
        if (count == 3'd0)
            assert (wptr == rptr);
endmodule
