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

module mcast_demux (
    input  wire       clk,
    input  wire       i_reset,
    input  wire       i_ce,
    input  wire       i_arm,
    input  wire [7:0] i_mask,
    input  wire [7:0] i_data,
    input  wire [7:0] i_ready,
    output wire [7:0] o_valid,
    output wire [7:0] o_data,
    output wire       o_done,
    output wire       o_busy
,
    output wire [7:0] delivered_o,
    output wire [7:0] mask_o,
    output wire [1:0] phase_o
);
    localparam IDLE  = 2'd0;
    localparam ARMED = 2'd1;
    localparam SEND  = 2'd2;
    localparam FIN   = 2'd3;

    reg [1:0] phase;
    reg [7:0] mask;
    reg [7:0] delivered;
    reg [7:0] data;
    initial phase = IDLE;
    initial mask = 8'd0;
    initial delivered = 8'd0;
    initial data = 8'd0;

    assign o_valid = (phase == SEND) ? (mask & ~delivered) : 8'd0;
    assign o_data  = data;
    assign o_done  = (phase == FIN);
    assign o_busy  = (phase != IDLE);

    wire [7:0] accepted = o_valid & i_ready;
    wire [7:0] next_del = delivered | accepted;

    always @(posedge clk) begin
        if (i_reset) begin
            phase     <= IDLE;
            mask      <= 8'd0;
            delivered <= 8'd0;
            data      <= 8'd0;
        end else if (i_ce) begin
            case (phase)
                IDLE: begin
                    if (i_arm && (i_mask != 8'd0)) begin
                        mask      <= i_mask;
                        data      <= i_data;
                        delivered <= 8'd0;
                        phase     <= ARMED;
                    end
                end
                ARMED: begin
                    phase <= SEND;
                end
                SEND: begin
                    delivered <= next_del;
                    if (next_del == mask)
                        phase <= FIN;
                end
                FIN: begin
                    phase     <= IDLE;
                    mask      <= 8'd0;
                    delivered <= 8'd0;
                end
            endcase
        end
    end

    // finish implies a copy was started (at least one delivery)
    always @(*)
        if (phase == FIN) assert (delivered != 8'd0);

    // finish implies an arm with a non-empty destination mask happened
    always @(*)
        if (phase == FIN) assert (mask != 8'd0);
    assign delivered_o = delivered;
    assign mask_o = mask;
    assign phase_o = phase;
endmodule

module flow_mcast_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        ce,
    input  wire        req,
    input  wire [63:0] req_key,
    input  wire [15:0] req_len,
    input  wire        wr_en,
    input  wire [6:0]  wr_idx,
    input  wire [63:0] wr_key,
    input  wire [31:0] wr_action,
    input  wire [7:0]  port_ready,
    output wire [7:0]  port_valid,
    output wire [7:0]  port_tag,
    output wire        copy_done,
    output wire        pend_valid_o,
    output wire [7:0]  pend_mask_o,
    output wire [7:0]  pend_tag_o
);
    wire        rsp_valid;
    wire        rsp_hit;
    wire [31:0] rsp_action;
    wire [6:0]  rsp_rule;
    wire        a_busy;
    wire [6:0]  a_idx;
    wire [7:0]  a_timer;

    wire        b_busy;
    wire        b_done;
    wire [7:0]  b_tag;
    wire [7:0]  b_delivered;
    wire [7:0]  b_mask;
    wire [1:0]  b_phase;

    reg         g_valid;
    reg  [7:0]  g_mask;
    reg  [7:0]  g_tag;
    initial g_valid = 1'b0;
    initial g_mask  = 8'd0;
    initial g_tag   = 8'd0;

    // flow table advances only when the pending slot is empty and no copy is in flight
    wire a_en  = ce & !b_busy & !g_valid;
    // a hit with a non-empty egress bitmask becomes a pending multicast
    wire take  = a_en & rsp_valid & rsp_hit & (rsp_action[7:0] != 8'd0);
    // demux accepts the arm when it is idle
    wire issue = ce & g_valid & (b_phase == 2'd0);

    always @(posedge clk) begin
        if (rst) begin
            g_valid <= 1'b0;
            g_mask  <= 8'd0;
            g_tag   <= 8'd0;
        end else if (issue) begin
            g_valid <= 1'b0;
        end else if (take) begin
            g_valid <= 1'b1;
            g_mask  <= rsp_action[7:0];
            g_tag   <= {1'b0, rsp_rule};
        end
    end

    flow_match_table u_fmt (
        .clk(clk),
        .rst(rst),
        .i_ce(a_en),
        .req(req),
        .req_key(req_key),
        .req_len(req_len),
        .wr_en(wr_en),
        .wr_idx(wr_idx),
        .wr_key(wr_key),
        .wr_action(wr_action),
        .rsp_valid(rsp_valid),
        .rsp_hit(rsp_hit),
        .rsp_action(rsp_action),
        .rsp_rule(rsp_rule),
        .busy_o(a_busy),
        .idx_o(a_idx),
        .timer_o(a_timer)
    );

    mcast_demux u_mc (
        .clk(clk),
        .i_reset(rst),
        .i_ce(ce),
        .i_arm(g_valid),
        .i_mask(g_mask),
        .i_data(g_tag),
        .i_ready(port_ready),
        .o_valid(port_valid),
        .o_data(b_tag),
        .o_done(b_done),
        .o_busy(b_busy),
        .delivered_o(b_delivered),
        .mask_o(b_mask),
        .phase_o(b_phase)
    );

    assign port_tag     = b_tag;
    assign copy_done    = b_done;
    assign pend_valid_o = g_valid;
    assign pend_mask_o  = g_mask;
    assign pend_tag_o   = g_tag;

    // any tag being replicated must name a real rule (0..127)
    always @(*)
        if (b_busy) assert (b_tag[7] == 1'b0);
endmodule
