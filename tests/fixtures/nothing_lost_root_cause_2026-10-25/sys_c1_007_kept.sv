module ec_jdbl_seq (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire        arm,
    input  wire        start,
    input  wire        clr,
    input  wire        sel_add,
    input  wire        wr_en,
    input  wire [3:0]  wr_addr,
    input  wire [31:0] wr_data,
    output wire        o_done,
    output wire [31:0] o_acc,
    output wire [31:0] o_x3
,
    output wire armed_o,
    output wire done_o,
    output wire pre_clr_o,
    output wire started_o,
    output wire [31:0] step_o
);
    localparam [31:0] PRIME = 32'hFFFFFFFB;
    localparam [31:0] LAST  = 32'd11;
    localparam [1:0]  OP_MUL = 2'd0;
    localparam [1:0]  OP_ADD = 2'd1;
    localparam [1:0]  OP_SUB = 2'd2;

    reg        armed;
    reg        started;
    reg        done;
    reg        pre_clr;
    reg        mode;
    reg [31:0] step;
    reg [31:0] acc;
    reg [31:0] rf [0:15];

    integer i;
    initial begin
        armed   = 1'b0;
        started = 1'b0;
        done    = 1'b0;
        pre_clr = 1'b1;
        mode    = 1'b0;
        step    = 32'd0;
        acc     = 32'd0;
        for (i = 0; i < 16; i = i + 1)
            rf[i] = 32'd0;
    end

    function [31:0] fadd;
        input [31:0] a;
        input [31:0] b;
        reg [32:0] s;
        begin
            s = {1'b0, a} + {1'b0, b};
            if (s >= {1'b0, PRIME})
                s = s - {1'b0, PRIME};
            fadd = s[31:0];
        end
    endfunction

    function [31:0] fsub;
        input [31:0] a;
        input [31:0] b;
        reg [32:0] s;
        begin
            if (a >= b)
                s = {1'b0, a} - {1'b0, b};
            else
                s = {1'b0, a} + {1'b0, PRIME} - {1'b0, b};
            fsub = s[31:0];
        end
    endfunction

    function [31:0] fmul;
        input [31:0] a;
        input [31:0] b;
        reg [63:0] p;
        reg [35:0] t;
        reg [35:0] u;
        begin
            p = {32'd0, a} * {32'd0, b};
            t = {4'd0, p[31:0]} + {4'd0, p[63:32]} * 36'd5;
            u = {4'd0, t[31:0]} + {32'd0, t[35:32]} * 36'd5;
            if (u >= {4'd0, PRIME})
                u = u - {4'd0, PRIME};
            if (u >= {4'd0, PRIME})
                u = u - {4'd0, PRIME};
            fmul = u[31:0];
        end
    endfunction

    reg [3:0] ra, rb, rd;
    reg [1:0] opc;
    always @(*) begin
        ra  = 4'd0;
        rb  = 4'd0;
        rd  = 4'd6;
        opc = OP_MUL;
        case (step)
            32'd1:  begin ra = 4'd1;  rb = mode ? 4'd4 : 4'd1; rd = 4'd6;  opc = OP_MUL; end
            32'd2:  begin ra = 4'd0;  rb = 4'd6;  rd = 4'd7;  opc = OP_MUL; end
            32'd3:  begin ra = 4'd0;  rb = mode ? 4'd3 : 4'd0; rd = 4'd8;  opc = OP_MUL; end
            32'd4:  begin ra = 4'd8;  rb = 4'd8;  rd = 4'd9;  opc = OP_ADD; end
            32'd5:  begin ra = 4'd9;  rb = 4'd8;  rd = 4'd9;  opc = OP_ADD; end
            32'd6:  begin ra = 4'd9;  rb = 4'd9;  rd = 4'd10; opc = OP_MUL; end
            32'd7:  begin ra = 4'd7;  rb = 4'd7;  rd = 4'd11; opc = OP_ADD; end
            32'd8:  begin ra = 4'd10; rb = 4'd11; rd = 4'd12; opc = OP_SUB; end
            32'd9:  begin ra = mode ? 4'd2 : 4'd1; rb = mode ? 4'd5 : 4'd2; rd = 4'd13; opc = OP_MUL; end
            32'd10: begin ra = 4'd6;  rb = 4'd6;  rd = 4'd15; opc = OP_MUL; end
            default: begin ra = 4'd0; rb = 4'd0; rd = 4'd6; opc = OP_MUL; end
        endcase
    end

    wire [31:0] opa = rf[ra];
    wire [31:0] opb = rf[rb];
    reg  [31:0] res;
    always @(*) begin
        case (opc)
            OP_MUL:  res = fmul(opa, opb);
            OP_ADD:  res = fadd(opa, opb);
            OP_SUB:  res = fsub(opa, opb);
            default: res = opa;
        endcase
    end

    wire [31:0] y3 = fsub(fmul(rf[9], fsub(rf[7], rf[12])), rf[15]);
    wire [31:0] z3 = fadd(rf[13], rf[13]);

    always @(posedge clk) begin
        if (rst) begin
            armed   <= 1'b0;
            started <= 1'b0;
            done    <= 1'b0;
            pre_clr <= 1'b1;
            mode    <= 1'b0;
            step    <= 32'd0;
            acc     <= 32'd0;
        end else if (en) begin
            if (clr) begin
                armed   <= 1'b0;
                started <= 1'b0;
                done    <= 1'b0;
                pre_clr <= 1'b0;
            end else begin
                if (arm)
                    armed <= 1'b1;
                if (start && armed && step == 32'd0)
                    started <= 1'b1;
                if (step == LAST)
                    done <= 1'b1;
            end
            if (step == 32'd0) begin
                if (start && armed && !clr) begin
                    step <= 32'd1;
                    mode <= sel_add;
                end
            end else if (step == LAST) begin
                step <= 32'd0;
                acc  <= y3;
            end else begin
                step <= step + 32'd1;
                acc  <= res;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst && en) begin
            if (step == 32'd0) begin
                if (wr_en)
                    rf[wr_addr] <= wr_data;
            end else if (step == LAST) begin
                rf[0] <= rf[12];
                rf[1] <= y3;
                rf[2] <= z3;
            end else begin
                rf[rd] <= res;
            end
        end
    end

    assign o_done = done;
    assign o_acc  = acc;
    assign o_x3   = rf[0];

    always @(*)
        if (pre_clr && done)
            assert (started && armed);
    assign armed_o = armed;
    assign done_o = done;
    assign pre_clr_o = pre_clr;
    assign started_o = started;
    assign step_o = step;
endmodule

module lim_mod_fold (
    input wire clk,
    input wire rst,
    input wire en,
    input wire start,
    input wire vld,
    input wire [63:0] chunk,
    output wire [15:0] res,
    output wire done
,
    output wire [8:0] idx_o,
    output wire [63:0] wexp_o,
    output wire [63:0] wreg_o
);
    localparam [63:0] M = 64'd65521;

    function [63:0] two64;
        input [31:0] dummy;
        integer i;
        reg [63:0] r;
        begin
            r = 64'd1;
            for (i = 0; i < 64; i = i + 1)
                r = (r * 64'd2) % M;
            two64 = r;
        end
    endfunction

    localparam [63:0] K = two64(0);

    function [63:0] kpow;
        input [31:0] n;
        integer i;
        reg [63:0] r;
        begin
            r = 64'd1;
            for (i = 0; i < n; i = i + 1)
                r = (r * K) % M;
            kpow = r;
        end
    endfunction

    localparam [63:0] C = kpow(256);

    reg [63:0] acc;
    reg [63:0] wreg;
    reg [8:0]  idx;
    reg        armed;
    initial acc = 64'd0;
    initial wreg = 64'd1;
    initial idx = 9'd0;
    initial armed = 1'b0;

    wire [64*257-1:0] wflat;
    genvar g;
    generate
        for (g = 0; g < 257; g = g + 1) begin : G_ROM
            localparam [63:0] WV = kpow(g);
            assign wflat[g*64 +: 64] = WV;
        end
    endgenerate

    wire [63:0] wexp = wflat[idx*64 +: 64];

    always @(posedge clk) begin
        if (rst) begin
            acc   <= 64'd0;
            wreg  <= 64'd1;
            idx   <= 9'd0;
            armed <= 1'b0;
        end else if (en) begin
            if (start) begin
                acc   <= 64'd0;
                wreg  <= 64'd1;
                idx   <= 9'd0;
                armed <= 1'b1;
            end else if (vld && (idx < 9'd256)) begin
                acc  <= (acc + (chunk % M) * wreg) % M;
                wreg <= (wreg * K) % M;
                idx  <= idx + 9'd1;
            end
        end
    end

    assign res  = acc[15:0];
    assign done = (idx == 9'd256);

    always @(posedge clk)
        if (armed && idx == 9'd256)
            assert (wreg == C);
    assign idx_o = idx;
    assign wexp_o = wexp;
    assign wreg_o = wreg;
endmodule

module ec_fold_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire        arm,
    input  wire        start,
    input  wire        clr,
    input  wire        sel_add,
    input  wire        wr_en,
    input  wire [3:0]  wr_addr,
    input  wire [31:0] wr_data,
    input  wire        fold_en,
    input  wire        fold_start,
    output wire        o_done,
    output wire [31:0] o_acc,
    output wire [31:0] o_x3,
    output wire [15:0] fold_res,
    output wire        fold_done,
    output wire [1:0]  q_cnt_o,
    output wire [1:0]  q_v_o,
    output wire        q_wr_o,
    output wire        q_rd_o,
    output wire        q_ovf_o,
    output wire [31:0] q_head_o
);
    wire        a_done;
    wire [31:0] a_acc;
    wire [31:0] a_x3;
    wire        a_armed;
    wire        a_done_m;
    wire        a_pre_clr;
    wire        a_started;
    wire [31:0] a_step;

    wire [15:0] b_res;
    wire        b_done;
    wire [8:0]  b_idx;
    wire [63:0] b_wexp;
    wire [63:0] b_wreg;

    reg        done_q;
    reg [1:0]  q_cnt;
    reg        q_wr;
    reg        q_rd;
    reg [1:0]  q_v;
    reg        q_ovf;
    reg [31:0] q_d [0:1];

    integer k;
    initial begin
        done_q = 1'b0;
        q_cnt  = 2'd0;
        q_wr   = 1'b0;
        q_rd   = 1'b0;
        q_v    = 2'b00;
        q_ovf  = 1'b0;
        for (k = 0; k < 2; k = k + 1)
            q_d[k] = 32'd0;
    end

    wire        a_fin   = a_done && !done_q;
    wire        q_vld   = (q_cnt != 2'd0);
    wire        b_start = fold_start || (b_done && q_vld);
    wire        q_pop   = q_vld && fold_en && !b_start && (b_idx < 9'd256);
    wire        q_push  = a_fin && (q_cnt != 2'd2);
    wire [31:0] q_head  = q_d[q_rd];
    wire [63:0] q_chunk = {32'd0, q_head};

    ec_jdbl_seq u_a (
        .clk(clk), .rst(rst), .en(en), .arm(arm), .start(start), .clr(clr),
        .sel_add(sel_add), .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .o_done(a_done), .o_acc(a_acc), .o_x3(a_x3),
        .armed_o(a_armed), .done_o(a_done_m), .pre_clr_o(a_pre_clr),
        .started_o(a_started), .step_o(a_step)
    );

    lim_mod_fold u_b (
        .clk(clk), .rst(rst), .en(fold_en), .start(b_start), .vld(q_vld),
        .chunk(q_chunk), .res(b_res), .done(b_done),
        .idx_o(b_idx), .wexp_o(b_wexp), .wreg_o(b_wreg)
    );

    always @(posedge clk) begin
        if (rst) begin
            done_q <= 1'b0;
            q_cnt  <= 2'd0;
            q_wr   <= 1'b0;
            q_rd   <= 1'b0;
            q_v    <= 2'b00;
            q_ovf  <= 1'b0;
            for (k = 0; k < 2; k = k + 1)
                q_d[k] <= 32'd0;
        end else begin
            done_q <= a_done;
            if (a_fin && q_cnt == 2'd2)
                q_ovf <= 1'b1;
            if (q_push) begin
                q_d[q_wr] <= a_acc;
                q_v[q_wr] <= 1'b1;
                q_wr      <= ~q_wr;
            end
            if (q_pop) begin
                q_v[q_rd] <= 1'b0;
                q_rd      <= ~q_rd;
            end
            case ({q_push, q_pop})
                2'b10:   q_cnt <= q_cnt + 2'd1;
                2'b01:   q_cnt <= q_cnt - 2'd1;
                default: q_cnt <= q_cnt;
            endcase
        end
    end

    always @(*)
        if (q_pop)
            assert (q_v[q_rd]);

    assign o_done    = a_done;
    assign o_acc     = a_acc;
    assign o_x3      = a_x3;
    assign fold_res  = b_res;
    assign fold_done = b_done;
    assign q_cnt_o   = q_cnt;
    assign q_v_o     = q_v;
    assign q_wr_o    = q_wr;
    assign q_rd_o    = q_rd;
    assign q_ovf_o   = q_ovf;
    assign q_head_o  = q_head;
endmodule
