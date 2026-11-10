// ====================================================================
// Copyright (C) 2026 
// Licensed under GNU GPLv3 / Text concepts under CC BY-SA 4.0
// Original source post on Threads. References required.
// ====================================================================

// --- 1. CORE HAMILTON MULTIPLIER KERNEL ---
module hamilton_lut_kernel (
    input  signed [7:0] q0, q1, q2, q3,
    input  signed [7:0] w0, w1, w2, w3,
    output signed [17:0] r0, r1, r2, r3
);
    wire signed [15:0] prod_00 = q0 * w0; wire signed [15:0] prod_01 = q0 * w1;
    wire signed [15:0] prod_02 = q0 * w2; wire signed [15:0] prod_03 = q0 * w3;
    wire signed [15:0] prod_10 = q1 * w0; wire signed [15:0] prod_11 = q1 * w1;
    wire signed [15:0] prod_12 = q1 * w2; wire signed [15:0] prod_13 = q1 * w3;
    wire signed [15:0] prod_20 = q2 * w0; wire signed [15:0] prod_21 = q2 * w1;
    wire signed [15:0] prod_22 = q2 * w2; wire signed [15:0] prod_23 = q2 * w3;
    wire signed [15:0] prod_30 = q3 * w0; wire signed [15:0] prod_31 = q3 * w1;
    wire signed [15:0] prod_32 = q3 * w2; wire signed [15:0] prod_33 = q3 * w3;

    assign r0 = prod_00 - prod_11 - prod_22 - prod_33;
    assign r1 = prod_01 + prod_10 + prod_23 - prod_32;
    assign r2 = prod_02 - prod_13 + prod_20 + prod_31;
    assign r3 = prod_03 + prod_12 - prod_21 + prod_30;
endmodule

// --- 2. HARDWARE QUATERNION RELU ACTIVATIE ---
module quaternion_split_relu (
    input  signed [7:0] in_x0, input  signed [7:0] in_x1,
    input  signed [7:0] in_x2, input  signed [7:0] in_x3,
    output signed [7:0] out_y0, output signed [7:0] out_y1,
    output signed [7:0] out_y2, output signed [7:0] out_y3
);
    assign out_y0 = (in_x0[7] == 1'b1) ? 8'sd0 : in_x0;
    assign out_y1 = (in_x1[7] == 1'b1) ? 8'sd0 : in_x1;
    assign out_y2 = (in_x2[7] == 1'b1) ? 8'sd0 : in_x2;
    assign out_y3 = (in_x3[7] == 1'b1) ? 8'sd0 : in_x3;
endmodule

// --- 3. COMPLETE INTEGRATED ACCELERATOR ENGINE WITH FIXED INDEXING ---
module full_quaternion_llm_engine (
    input clk,
    input rst,
    
    input hbm2_tvalid,
    input [512:0] hbm2_tdata,
    output hbm2_tready,
    
    input prompt_tvalid,
    input signed [31:0] prompt_tdata, 
    
    output reg out_valid,
    output reg [7:0] chosen_token_id
);

    assign hbm2_tready = 1'b1;

    wire signed [7:0] clk_q0 = prompt_tdata[31:24];
    wire signed [7:0] clk_q1 = prompt_tdata[23:16];
    wire signed [7:0] clk_q2 = prompt_tdata[15:8];
    wire signed [7:0] clk_q3 = prompt_tdata[7:0];

    wire signed [7:0] key_matrix_r0 [0:3]; wire signed [7:0] key_matrix_r1 [0:3];
    wire signed [7:0] key_matrix_r2 [0:3]; wire signed [7:0] key_matrix_r3 [0:3];

    genvar b;
    generate
        for (b = 0; b < 4; b = b + 1) begin: hbm_slicing
            assign key_matrix_r0[b] = hbm2_tdata[(b*32)+31 : (b*32)+24];
            assign key_matrix_r1[b] = hbm2_tdata[(b*32)+23 : (b*32)+16];
            assign key_matrix_r2[b] = hbm2_tdata[(b*32)+15 : (b*32)+8];
            assign key_matrix_r3[b] = hbm2_tdata[(b*32)+7  : (b*32)+0];
        end
    endgenerate

    wire signed [17:0] kr0 [0:3]; wire signed [17:0] kr1 [0:3];
    wire signed [17:0] kr2 [0:3]; wire signed [17:0] kr3 [0:3];

    genvar lane;
    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin: simd_matrix
            hamilton_lut_kernel k_inst (
                .q0(clk_q0), .q1(clk_q1), .q2(clk_q2), .q3(clk_q3),
                .w0(key_matrix_r0[lane]), .w1(key_matrix_r1[lane]), 
                .w2(key_matrix_r2[lane]), .w3(key_matrix_r3[lane]),
                .r0(kr0[lane]), .r1(kr1[lane]), .r2(kr2[lane]), .r3(kr3[lane])
            );
        end
    endgenerate

    reg pipe_valid;
    reg signed [7:0] p_r0 [0:3]; reg signed [7:0] p_r1 [0:3];
    reg signed [7:0] p_r2 [0:3]; reg signed [7:0] p_r3 [0:3];

    integer i;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pipe_valid <= 1'b0;
            for (i = 0; i < 4; i = i + 1) begin
                p_r0[i] <= 0; p_r1[i] <= 0; p_r2[i] <= 0; p_r3[i] <= 0;
            end
        end else begin
            pipe_valid <= hbm2_tvalid && prompt_tvalid;
            if (hbm2_tvalid && prompt_tvalid) begin
                for (i = 0; i < 4; i = i + 1) begin
                    p_r0[i] <= kr0[i][14:7]; 
                    p_r1[i] <= kr1[i][14:7];
                    p_r2[i] <= kr2[i][14:7];
                    p_r3[i] <= kr3[i][14:7];
                end
            end
        end
    end

    wire signed [7:0] act_r0 [0:3]; wire signed [7:0] act_r1 [0:3];
    wire signed [7:0] act_r2 [0:3]; wire signed [7:0] act_r3 [0:3];
    
    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin: act_layer
            quaternion_split_relu relu_inst (
                .in_x0(p_r0[lane]),    .in_x1(p_r1[lane]),    .in_x2(p_r2[lane]),    .in_x3(p_r3[lane]),
                .out_y0(act_r0[lane]), .out_y1(act_r1[lane]), .out_y2(act_r2[lane]), .out_y3(act_r3[lane])
            );
        end
    endgenerate

    // --- CRUCIALE FIX: HARDWARE AMPLITUDE VERGELIJKING PER INSTANTIE ---
    wire [15:0] amp0 = (act_r0[0]*act_r0[0]) + (act_r1[0]*act_r1[0]) + (act_r2[0]*act_r2[0]) + (act_r3[0]*act_r3[0]);
    wire [15:0] amp1 = (act_r0[1]*act_r0[1]) + (act_r1[1]*act_r1[1]) + (act_r2[1]*act_r2[1]) + (act_r3[1]*act_r3[1]);
    wire [15:0] amp2 = (act_r0[2]*act_r0[2]) + (act_r1[2]*act_r1[2]) + (act_r2[2]*act_r2[2]) + (act_r3[2]*act_r3[2]);
    wire [15:0] amp3 = (act_r0[3]*act_r0[3]) + (act_r1[3]*act_r1[3]) + (act_r2[3]*act_r2[3]) + (act_r3[3]*act_r3[3]);

    reg [7:0] winner_id;
    reg [15:0] max_amp;

    always @(*) begin
        max_amp = amp0;
        winner_id = 8'd0; 
        if (amp1 > max_amp) begin max_amp = amp1; winner_id = 8'd1; end
        if (amp2 > max_amp) begin max_amp = amp2; winner_id = 8'd2; end
        if (amp3 > max_amp) begin max_amp = amp3; winner_id = 8'd3; end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out_valid <= 1'b0;
            chosen_token_id <= 8'd0;
        end else begin
            out_valid <= pipe_valid;
            if (pipe_valid) begin
                chosen_token_id <= winner_id;
            end
        end
    end
endmodule
