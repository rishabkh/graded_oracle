module benes_lane_shuffler (
    input  wire         clk,
    input  wire         rst,
    input  wire         en,
    input  wire         in_valid,
    input  wire [511:0] in_data,
    input  wire [7:0]   in_sel,
    input  wire         cfg_we,
    input  wire [7:0]   cfg_addr,
    input  wire [19:0]  cfg_bits,
    output wire         out_valid,
    output wire [511:0] out_data,
    output wire [63:0]  o_beats_done
,
    output wire [63:0] inflight_o,
    output wire [4:0] valid_pipe_o
);
    reg [19:0]  cfg_tab   [0:255];
    reg [3:0]   stage_cfg [0:4];
    reg [19:0]  cfg_pipe  [0:3];
    reg [511:0] data_pipe [0:4];
    reg [4:0]   valid_pipe;
    reg [63:0]  inflight;
    reg [63:0]  beats_done;
    reg         armed;

    integer i;
    initial begin
        for (i = 0; i < 256; i = i + 1) cfg_tab[i] = 20'd0;
        for (i = 0; i < 5; i = i + 1) begin
            stage_cfg[i] = 4'd0;
            data_pipe[i] = 512'd0;
        end
        for (i = 0; i < 4; i = i + 1) cfg_pipe[i] = 20'd0;
        valid_pipe = 5'd0;
        inflight   = 64'd0;
        beats_done = 64'd0;
        armed      = 1'b0;
    end

    // butterfly stage, pair distance 4
    function [511:0] sw4;
        input [511:0] d;
        input [3:0]   c;
        begin
            sw4[0*64 +: 64] = c[0] ? d[4*64 +: 64] : d[0*64 +: 64];
            sw4[4*64 +: 64] = c[0] ? d[0*64 +: 64] : d[4*64 +: 64];
            sw4[1*64 +: 64] = c[1] ? d[5*64 +: 64] : d[1*64 +: 64];
            sw4[5*64 +: 64] = c[1] ? d[1*64 +: 64] : d[5*64 +: 64];
            sw4[2*64 +: 64] = c[2] ? d[6*64 +: 64] : d[2*64 +: 64];
            sw4[6*64 +: 64] = c[2] ? d[2*64 +: 64] : d[6*64 +: 64];
            sw4[3*64 +: 64] = c[3] ? d[7*64 +: 64] : d[3*64 +: 64];
            sw4[7*64 +: 64] = c[3] ? d[3*64 +: 64] : d[7*64 +: 64];
        end
    endfunction

    // pair distance 2
    function [511:0] sw2;
        input [511:0] d;
        input [3:0]   c;
        begin
            sw2[0*64 +: 64] = c[0] ? d[2*64 +: 64] : d[0*64 +: 64];
            sw2[2*64 +: 64] = c[0] ? d[0*64 +: 64] : d[2*64 +: 64];
            sw2[1*64 +: 64] = c[1] ? d[3*64 +: 64] : d[1*64 +: 64];
            sw2[3*64 +: 64] = c[1] ? d[1*64 +: 64] : d[3*64 +: 64];
            sw2[4*64 +: 64] = c[2] ? d[6*64 +: 64] : d[4*64 +: 64];
            sw2[6*64 +: 64] = c[2] ? d[4*64 +: 64] : d[6*64 +: 64];
            sw2[5*64 +: 64] = c[3] ? d[7*64 +: 64] : d[5*64 +: 64];
            sw2[7*64 +: 64] = c[3] ? d[5*64 +: 64] : d[7*64 +: 64];
        end
    endfunction

    // pair distance 1 (middle stage)
    function [511:0] sw1;
        input [511:0] d;
        input [3:0]   c;
        begin
            sw1[0*64 +: 64] = c[0] ? d[1*64 +: 64] : d[0*64 +: 64];
            sw1[1*64 +: 64] = c[0] ? d[0*64 +: 64] : d[1*64 +: 64];
            sw1[2*64 +: 64] = c[1] ? d[3*64 +: 64] : d[2*64 +: 64];
            sw1[3*64 +: 64] = c[1] ? d[2*64 +: 64] : d[3*64 +: 64];
            sw1[4*64 +: 64] = c[2] ? d[5*64 +: 64] : d[4*64 +: 64];
            sw1[5*64 +: 64] = c[2] ? d[4*64 +: 64] : d[5*64 +: 64];
            sw1[6*64 +: 64] = c[3] ? d[7*64 +: 64] : d[6*64 +: 64];
            sw1[7*64 +: 64] = c[3] ? d[6*64 +: 64] : d[7*64 +: 64];
        end
    endfunction

    wire [19:0] cfg0 = cfg_tab[in_sel];

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 256; i = i + 1) cfg_tab[i] <= 20'd0;
            for (i = 0; i < 5; i = i + 1) begin
                stage_cfg[i] <= 4'd0;
                data_pipe[i] <= 512'd0;
            end
            for (i = 0; i < 4; i = i + 1) cfg_pipe[i] <= 20'd0;
            valid_pipe <= 5'd0;
            inflight   <= 64'd0;
            beats_done <= 64'd0;
            armed      <= 1'b0;
        end else if (en) begin
            if (cfg_we) begin
                cfg_tab[cfg_addr] <= cfg_bits;
                armed <= 1'b1;
            end
            data_pipe[0] <= sw4(in_data, cfg0[3:0]);
            stage_cfg[0] <= cfg0[3:0];
            cfg_pipe[0]  <= cfg0;
            data_pipe[1] <= sw2(data_pipe[0], cfg_pipe[0][7:4]);
            stage_cfg[1] <= cfg_pipe[0][7:4];
            cfg_pipe[1]  <= cfg_pipe[0];
            data_pipe[2] <= sw1(data_pipe[1], cfg_pipe[1][11:8]);
            stage_cfg[2] <= cfg_pipe[1][11:8];
            cfg_pipe[2]  <= cfg_pipe[1];
            data_pipe[3] <= sw2(data_pipe[2], cfg_pipe[2][15:12]);
            stage_cfg[3] <= cfg_pipe[2][15:12];
            cfg_pipe[3]  <= cfg_pipe[2];
            data_pipe[4] <= sw4(data_pipe[3], cfg_pipe[3][19:16]);
            stage_cfg[4] <= cfg_pipe[3][19:16];
            valid_pipe <= {valid_pipe[3:0], in_valid};
            inflight   <= inflight + {63'd0, in_valid} - {63'd0, valid_pipe[4]};
            if (valid_pipe[4])
                beats_done <= beats_done + 64'd1;
        end
    end

    assign out_valid    = valid_pipe[4];
    assign out_data     = data_pipe[4];
    assign o_beats_done = beats_done;

    always @(posedge clk)
        if (armed)
            assert (inflight <= 64'd5);
    assign inflight_o = inflight;
    assign valid_pipe_o = valid_pipe;
endmodule

module lvds_port_perfmon (
    input  wire         clk,
    input  wire         rst,
    input  wire         en,
    input  wire         valid,
    input  wire         ready,
    input  wire [127:0] data,
    input  wire         win_start,
    output wire [31:0]  o_txn_cnt,
    output wire [35:0]  o_byte_cnt,
    output wire [31:0]  o_stall_cnt,
    output wire [47:0]  o_lat_cum,
    output wire [15:0]  o_win_cnt
,
    output wire [35:0] byte_cnt_o,
    output wire hs_d_o,
    output wire seen_txn_o,
    output wire [31:0] txn_cnt_o
);
    wire hs    = en & valid & ready;
    wire stall = en & valid & ~ready;

    reg [31:0] txn_cnt   = 32'd0;
    reg        hs_d      = 1'b0;
    reg [35:0] byte_cnt  = 36'd0;
    reg [31:0] stall_cnt = 32'd0;
    reg [15:0] lat_cur   = 16'd0;
    reg [47:0] lat_cum   = 48'd0;
    reg [15:0] win_cnt   = 16'd0;
    reg        armed     = 1'b0;
    reg        seen_txn  = 1'b0;

    always @(posedge clk) begin
        if (rst) begin
            txn_cnt   <= 32'd0;
            hs_d      <= 1'b0;
            byte_cnt  <= 36'd0;
            stall_cnt <= 32'd0;
            lat_cur   <= 16'd0;
            lat_cum   <= 48'd0;
            win_cnt   <= 16'd0;
            armed     <= 1'b0;
            seen_txn  <= 1'b0;
        end else if (en) begin
            hs_d <= hs;
            if (hs) begin
                txn_cnt  <= txn_cnt + 32'd1;
                seen_txn <= 1'b1;
                lat_cum  <= lat_cum + {32'd0, lat_cur};
                lat_cur  <= 16'd0;
            end else if (valid) begin
                lat_cur  <= lat_cur + 16'd1;
            end
            if (hs_d)
                byte_cnt <= byte_cnt + 36'd16;
            if (stall)
                stall_cnt <= stall_cnt + 32'd1;
            if (win_start) begin
                win_cnt <= 16'd0;
                armed   <= 1'b1;
            end else begin
                win_cnt <= win_cnt + 16'd1;
            end
        end
    end

    assign o_txn_cnt   = txn_cnt;
    assign o_byte_cnt  = byte_cnt;
    assign o_stall_cnt = stall_cnt;
    assign o_lat_cum   = lat_cum;
    assign o_win_cnt   = win_cnt;

    always @(posedge clk)
        if (armed && byte_cnt != 36'd0)
            assert (seen_txn);
    assign byte_cnt_o = byte_cnt;
    assign hs_d_o = hs_d;
    assign seen_txn_o = seen_txn;
    assign txn_cnt_o = txn_cnt;
endmodule

module benes_perfmon_top (
    input  wire         clk,
    input  wire         rst,
    input  wire         en,
    input  wire         in_valid,
    input  wire [511:0] in_data,
    input  wire [7:0]   in_sel,
    input  wire         cfg_we,
    input  wire [7:0]   cfg_addr,
    input  wire [19:0]  cfg_bits,
    input  wire         ready,
    input  wire         win_start,
    output wire [63:0]  o_beats_done,
    output wire         o_port_valid,
    output wire [127:0] o_port_data,
    output wire [31:0]  o_cap_cnt,
    output wire [31:0]  o_drop_cnt,
    output wire [31:0]  o_txn_cnt,
    output wire [35:0]  o_byte_cnt,
    output wire [31:0]  o_stall_cnt,
    output wire [47:0]  o_lat_cum,
    output wire [15:0]  o_win_cnt
);
    wire         a_out_valid;
    wire [511:0] a_out_data;
    wire [63:0]  a_inflight;
    wire [4:0]   a_valid_pipe;

    wire [35:0]  b_byte_cnt;
    wire         b_hs_d;
    wire         b_seen_txn;
    wire [31:0]  b_txn_cnt;

    reg          g_valid  = 1'b0;
    reg  [127:0] g_data   = 128'd0;
    reg  [31:0]  cap_cnt  = 32'd0;
    reg  [31:0]  drop_cnt = 32'd0;

    benes_lane_shuffler u_shuf (
        .clk(clk),
        .rst(rst),
        .en(en),
        .in_valid(in_valid),
        .in_data(in_data),
        .in_sel(in_sel),
        .cfg_we(cfg_we),
        .cfg_addr(cfg_addr),
        .cfg_bits(cfg_bits),
        .out_valid(a_out_valid),
        .out_data(a_out_data),
        .o_beats_done(o_beats_done),
        .inflight_o(a_inflight),
        .valid_pipe_o(a_valid_pipe)
    );

    wire slot_free = !g_valid || ready;

    always @(posedge clk) begin
        if (rst) begin
            g_valid  <= 1'b0;
            g_data   <= 128'd0;
            cap_cnt  <= 32'd0;
            drop_cnt <= 32'd0;
        end else if (en) begin
            if (slot_free) begin
                g_valid <= a_out_valid;
                if (a_out_valid) begin
                    g_data  <= a_out_data[127:0];
                    cap_cnt <= cap_cnt + 32'd1;
                end
            end else if (a_out_valid) begin
                drop_cnt <= drop_cnt + 32'd1;
            end
        end
    end

    lvds_port_perfmon u_mon (
        .clk(clk),
        .rst(rst),
        .en(en),
        .valid(g_valid),
        .ready(ready),
        .data(g_data),
        .win_start(win_start),
        .o_txn_cnt(o_txn_cnt),
        .o_byte_cnt(o_byte_cnt),
        .o_stall_cnt(o_stall_cnt),
        .o_lat_cum(o_lat_cum),
        .o_win_cnt(o_win_cnt),
        .byte_cnt_o(b_byte_cnt),
        .hs_d_o(b_hs_d),
        .seen_txn_o(b_seen_txn),
        .txn_cnt_o(b_txn_cnt)
    );

    assign o_port_valid = g_valid;
    assign o_port_data  = g_data;
    assign o_cap_cnt    = cap_cnt;
    assign o_drop_cnt   = drop_cnt;

    always @(posedge clk)
        if (!g_valid)
            assert (cap_cnt == b_txn_cnt);
endmodule
