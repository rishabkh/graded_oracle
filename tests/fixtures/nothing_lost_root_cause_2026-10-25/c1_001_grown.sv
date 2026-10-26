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

module mpu_region_checker (
    input  wire        clk,
    input  wire        i_ce,
    input  wire        cfg_we,
    input  wire [4:0]  cfg_idx,
    input  wire [15:0] cfg_base,
    input  wire [15:0] cfg_limit,
    input  wire [1:0]  cfg_perm,
    input  wire        lock_req,
    input  wire        acc_valid,
    input  wire        acc_write,
    input  wire [15:0] acc_addr,
    output reg         grant,
    output reg         fault
,
    output wire hit_q_o,
    output wire miss_q_o
);
    // 24 regions: 16-bit base, 16-bit limit, 2-bit permission {write, read}
    reg [383:0] base_flat  = 384'd0;
    reg [383:0] limit_flat = 384'd0;
    reg [47:0]  perm_flat  = 48'd0;

    reg pre_lock = 1'b1;   // high until the configuration is locked for the first time
    reg hit_q    = 1'b0;
    reg miss_q   = 1'b0;
    reg pv       = 1'b0;

    initial grant = 1'b0;
    initial fault = 1'b0;

    reg hit;
    integer i;
    always @(*) begin
        hit = 1'b0;
        for (i = 0; i < 24; i = i + 1) begin
            if ((acc_write ? perm_flat[2*i+1] : perm_flat[2*i]) &&
                (acc_addr >= base_flat[16*i +: 16]) &&
                (acc_addr <= limit_flat[16*i +: 16]))
                hit = 1'b1;
        end
    end

    always @(posedge clk) begin
        if (i_ce) begin
            if (cfg_we && pre_lock && (cfg_idx < 5'd24)) begin
                base_flat[cfg_idx*16 +: 16]  <= cfg_base;
                limit_flat[cfg_idx*16 +: 16] <= cfg_limit;
                perm_flat[cfg_idx*2 +: 2]    <= cfg_perm;
            end
            if (lock_req)
                pre_lock <= 1'b0;
            hit_q  <= acc_valid && hit;
            miss_q <= acc_valid && !hit;
            grant  <= hit_q;
            fault  <= miss_q;
        end
    end

    always @(posedge clk) pv <= 1'b1;

    always @(posedge clk)
        if (pv && pre_lock && $rose(fault))
            assert (!grant);
    assign hit_q_o = hit_q;
    assign miss_q_o = miss_q;
endmodule

module flow_mpu_top (
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
    input  wire [4:0]  cfg_idx,
    input  wire [15:0] cfg_base,
    input  wire [15:0] cfg_limit,
    input  wire [1:0]  cfg_perm,
    input  wire        lock_req,
    output wire        grant,
    output wire        fault,
    output wire [6:0]  rsp_rule,
    output wire        g_valid_o,
    output wire [15:0] g_addr_o,
    output wire        g_write_o
);
    wire        rsp_valid;
    wire        rsp_hit;
    wire [31:0] rsp_action;
    wire        busy_a;
    wire [6:0]  idx_a;
    wire [7:0]  timer_a;
    wire        hit_q;
    wire        miss_q;

    // pending destination-buffer access handed from the classifier to the MPU
    reg        g_valid;
    reg [15:0] g_addr;
    reg        g_write;

    initial begin
        g_valid = 1'b0;
        g_addr  = 16'd0;
        g_write = 1'b0;
    end

    flow_match_table u_flow (
        .clk(clk),
        .rst(rst),
        .i_ce(i_ce),
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
        .busy_o(busy_a),
        .idx_o(idx_a),
        .timer_o(timer_a)
    );

    always @(posedge clk) begin
        if (rst) begin
            g_valid <= 1'b0;
            g_addr  <= 16'd0;
            g_write <= 1'b0;
        end else if (i_ce) begin
            g_valid <= rsp_valid && rsp_hit;
            if (rsp_valid && rsp_hit) begin
                g_addr  <= rsp_action[15:0];
                g_write <= rsp_action[16];
            end
        end
    end

    mpu_region_checker u_mpu (
        .clk(clk),
        .i_ce(i_ce),
        .cfg_we(cfg_we),
        .cfg_idx(cfg_idx),
        .cfg_base(cfg_base),
        .cfg_limit(cfg_limit),
        .cfg_perm(cfg_perm),
        .lock_req(lock_req),
        .acc_valid(g_valid),
        .acc_write(g_write),
        .acc_addr(g_addr),
        .grant(grant),
        .fault(fault),
        .hit_q_o(hit_q),
        .miss_q_o(miss_q)
    );

    assign g_valid_o = g_valid;
    assign g_addr_o  = g_addr;
    assign g_write_o = g_write;

    // the MPU never holds two packet steering decisions in its check stage at once
    always @(posedge clk)
        if (g_valid)
            assert (!(hit_q || miss_q));
endmodule
