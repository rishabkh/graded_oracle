module flow_match_table (
    input  wire        clk,
    input  wire        rst,
    input  wire        i_ce,
    input  wire        req,
    input  wire [63:0] req_key,
    input  wire [15:0] req_len,
    input  wire        wr_en,
    input  wire [6:0]  wr_idx,
    input  wire [63:0] wr_key,
    input  wire [31:0] wr_action,
    output reg         rsp_valid,
    output reg         rsp_hit,
    output reg  [31:0] rsp_action,
    output reg  [6:0]  rsp_rule
,
    output wire busy_o,
    output wire [6:0] idx_o,
    output wire [7:0] timer_o
);
    reg [63:0]  key_mem  [0:127];
    reg [31:0]  act_mem  [0:127];
    reg [63:0]  pkt_cnt  [0:127];
    reg [63:0]  byte_cnt [0:127];
    reg [127:0] valid;

    reg        busy;
    reg [6:0]  idx;
    reg [7:0]  timer;
    reg [63:0] q_key;
    reg [63:0] q_len;

    integer i;
    initial begin
        for (i = 0; i < 128; i = i + 1) begin
            key_mem[i]  = 64'd0;
            act_mem[i]  = 32'd0;
            pkt_cnt[i]  = 64'd0;
            byte_cnt[i] = 64'd0;
        end
        valid      = 128'd0;
        busy       = 1'b0;
        idx        = 7'd0;
        timer      = 8'd0;
        q_key      = 64'd0;
        q_len      = 64'd0;
        rsp_valid  = 1'b0;
        rsp_hit    = 1'b0;
        rsp_action = 32'd0;
        rsp_rule   = 7'd0;
    end

    wire hit  = valid[idx] && (key_mem[idx] == q_key);
    wire last = (idx == 7'd127);

    always @(posedge clk) begin
        if (rst) begin
            busy      <= 1'b0;
            idx       <= 7'd0;
            timer     <= 8'd0;
            valid     <= 128'd0;
            rsp_valid <= 1'b0;
            rsp_hit   <= 1'b0;
        end else if (i_ce) begin
            rsp_valid <= 1'b0;
            if (!busy) begin
                if (wr_en) begin
                    key_mem[wr_idx]  <= wr_key;
                    act_mem[wr_idx]  <= wr_action;
                    pkt_cnt[wr_idx]  <= 64'd0;
                    byte_cnt[wr_idx] <= 64'd0;
                    valid[wr_idx]    <= 1'b1;
                end
                if (req) begin
                    busy  <= 1'b1;
                    idx   <= 7'd0;
                    timer <= 8'd128;
                    q_key <= req_key;
                    q_len <= {48'd0, req_len};
                end
            end else begin
                if (hit) begin
                    pkt_cnt[idx]  <= pkt_cnt[idx] + 64'd1;
                    byte_cnt[idx] <= byte_cnt[idx] + q_len;
                    rsp_valid     <= 1'b1;
                    rsp_hit       <= 1'b1;
                    rsp_action    <= act_mem[idx];
                    rsp_rule      <= idx;
                    busy          <= 1'b0;
                    timer         <= 8'd0;
                end else if (last) begin
                    rsp_valid     <= 1'b1;
                    rsp_hit       <= 1'b0;
                    rsp_action    <= 32'd0;
                    rsp_rule      <= 7'd0;
                    busy          <= 1'b0;
                    timer         <= 8'd0;
                end else begin
                    idx   <= idx + 7'd1;
                    timer <= timer - 8'd1;
                end
            end
        end
    end

    // After a lookup is accepted (busy set) until it completes (busy cleared),
    // the completion countdown must never expire.
    always @(posedge clk)
        if (busy)
            assert (timer != 8'd0);
    assign busy_o = busy;
    assign idx_o = idx;
    assign timer_o = timer;
endmodule

module usb_route_lookup (
    input wire clk,
    input wire i_reset,
    input wire i_ce,
    input wire i_pkt_valid,
    input wire [31:0] i_dest,
    input wire cfg_we,
    input wire [4:0] cfg_addr,
    input wire [31:0] cfg_data,
    output wire o_done,
    output wire [31:0] o_port
,
    output wire fin_q_o,
    output wire [1:0] phase_o,
    output wire [31:0] step_cnt_o
);
    localparam [1:0] P_IDLE = 2'd0, P_ARM = 2'd1, P_START = 2'd2, P_FIN = 2'd3;

    reg [1:0]  phase    = 2'd0;
    reg [31:0] step_cnt = 32'd0;
    reg        fin_q    = 1'b0;
    reg [31:0] dest_q   = 32'd0;
    reg [31:0] port_q   = 32'd0;

    reg [31:0] rt_mem [0:31];
    integer i;
    initial begin
        for (i = 0; i < 32; i = i + 1)
            rt_mem[i] = 32'd0;
    end

    // routing table configuration write port
    always @(posedge clk)
        if (!i_reset && i_ce && cfg_we)
            rt_mem[cfg_addr] <= cfg_data;

    // lookup sequencer: arm (capture dest) -> start (table read) -> finish (port valid)
    always @(posedge clk) begin
        if (i_reset) begin
            phase    <= P_IDLE;
            step_cnt <= 32'd0;
            fin_q    <= 1'b0;
            dest_q   <= 32'd0;
            port_q   <= 32'd0;
        end else if (i_ce) begin
            fin_q <= (step_cnt == 32'd2);
            case (phase)
                P_IDLE:  if (i_pkt_valid) begin
                             phase  <= P_ARM;
                             dest_q <= i_dest;
                         end
                P_ARM:   phase <= P_START;
                P_START: begin
                             phase  <= P_FIN;
                             port_q <= rt_mem[dest_q[4:0]];
                         end
                default: phase <= P_IDLE;
            endcase
            case (step_cnt)
                32'd0:   if (i_pkt_valid) step_cnt <= 32'd1;
                32'd3:   step_cnt <= 32'd0;
                default: step_cnt <= step_cnt + 32'd1;
            endcase
        end
    end

    assign o_done = fin_q;
    assign o_port = port_q;

    reg pv = 1'b0;
    always @(posedge clk) pv <= 1'b1;

    always @(posedge clk)
        if (pv && fin_q)
            assert (($past(phase) == P_START) || ($past(phase) == P_FIN));
    assign fin_q_o = fin_q;
    assign phase_o = phase;
    assign step_cnt_o = step_cnt;
endmodule

module flow_route_pipe (
    input  wire        clk,
    input  wire        rst,
    input  wire        i_ce,
    input  wire        req,
    input  wire [63:0] req_key,
    input  wire [15:0] req_len,
    input  wire        wr_en,
    input  wire [6:0]  wr_idx,
    input  wire [63:0] wr_key,
    input  wire [31:0] wr_action,
    input  wire        cfg_we,
    input  wire [4:0]  cfg_addr,
    input  wire [31:0] cfg_data,
    output wire        o_done,
    output wire [31:0] o_port,
    output wire [6:0]  o_rule,
    output wire        o_miss,
    output wire        o_stall,
    output wire        pend_v_o,
    output wire        fly_v_o
);
    // flow table side
    wire        a_rsp_valid;
    wire        a_rsp_hit;
    wire [31:0] a_rsp_action;
    wire [6:0]  a_rsp_rule;
    wire        a_busy;
    wire [6:0]  a_idx;
    wire [7:0]  a_timer;

    // route lookup side
    wire        b_done;
    wire [31:0] b_port;
    wire        b_fin_q;
    wire [1:0]  b_phase;
    wire [31:0] b_step_cnt;

    // forwarding register between classifier and router
    reg        pend_v;
    reg [31:0] pend_dest;
    reg [6:0]  pend_rule;
    // rule tag for the packet currently inside the route lookup
    reg        fly_v;
    reg [6:0]  fly_rule;

    initial begin
        pend_v    = 1'b0;
        pend_dest = 32'd0;
        pend_rule = 7'd0;
        fly_v     = 1'b0;
        fly_rule  = 7'd0;
    end

    // classifier only advances while the forwarding slot is free
    wire a_ce   = i_ce && !pend_v;
    // matched response accepted into the forwarding slot
    wire a_fwd  = !pend_v && a_rsp_valid && a_rsp_hit;
    // route lookup takes the slot when idle
    wire b_take = pend_v && (b_phase == 2'd0);

    flow_match_table u_fmt (
        .clk(clk),
        .rst(rst),
        .i_ce(a_ce),
        .req(req),
        .req_key(req_key),
        .req_len(req_len),
        .wr_en(wr_en),
        .wr_idx(wr_idx),
        .wr_key(wr_key),
        .wr_action(wr_action),
        .rsp_valid(a_rsp_valid),
        .rsp_hit(a_rsp_hit),
        .rsp_action(a_rsp_action),
        .rsp_rule(a_rsp_rule),
        .busy_o(a_busy),
        .idx_o(a_idx),
        .timer_o(a_timer)
    );

    usb_route_lookup u_rt (
        .clk(clk),
        .i_reset(rst),
        .i_ce(i_ce),
        .i_pkt_valid(pend_v),
        .i_dest(pend_dest),
        .cfg_we(cfg_we),
        .cfg_addr(cfg_addr),
        .cfg_data(cfg_data),
        .o_done(b_done),
        .o_port(b_port),
        .fin_q_o(b_fin_q),
        .phase_o(b_phase),
        .step_cnt_o(b_step_cnt)
    );

    always @(posedge clk) begin
        if (rst) begin
            pend_v    <= 1'b0;
            pend_dest <= 32'd0;
            pend_rule <= 7'd0;
            fly_v     <= 1'b0;
            fly_rule  <= 7'd0;
        end else if (i_ce) begin
            if (b_take) begin
                pend_v   <= 1'b0;
                fly_v    <= 1'b1;
                fly_rule <= pend_rule;
            end else if (a_fwd) begin
                pend_v    <= 1'b1;
                pend_dest <= a_rsp_action;
                pend_rule <= a_rsp_rule;
            end
            if (b_done)
                fly_v <= 1'b0;
        end
    end

    assign o_done   = b_done;
    assign o_port   = b_port;
    assign o_rule   = fly_rule;
    assign o_miss   = a_rsp_valid && !a_rsp_hit;
    assign o_stall  = pend_v;
    assign pend_v_o = pend_v;
    assign fly_v_o  = fly_v;

    // every routed result must belong to a tagged, forwarded packet
    always @(posedge clk)
        if (b_done)
            assert (fly_v);
endmodule
