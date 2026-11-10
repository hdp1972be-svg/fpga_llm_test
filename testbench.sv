`timescale 1ns / 1ps

module full_quaternion_llm_engine_prompt_tb;

    reg clk; reg rst; reg hbm2_tvalid; reg prompt_tvalid;
    reg [512:0] hbm2_tdata;
    reg signed [31:0] prompt_tdata;

    wire hbm2_tready; wire out_valid;
    wire [7:0] chosen_token_id;

    full_quaternion_llm_engine uut (
        .clk(clk), .rst(rst), 
        .hbm2_tvalid(hbm2_tvalid), .hbm2_tdata(hbm2_tdata), .hbm2_tready(hbm2_tready),
        .prompt_tvalid(prompt_tvalid), .prompt_tdata(prompt_tdata),
        .out_valid(out_valid), .chosen_token_id(chosen_token_id)
    );

    always #5 clk = ~clk;

    integer weight_file;
    integer prompt_file;
    integer w_bytes, p_bytes;
    
    reg [7:0] weight_buffer [0:63];
    reg [7:0] prompt_buffer [0:3];
    integer idx;

    initial begin
        $dumpfile("dump.vcd"); 
        $dumpvars(0, full_quaternion_llm_engine_prompt_tb);
        
        clk = 0; rst = 1; hbm2_tvalid = 0; prompt_tvalid = 0; hbm2_tdata = 0; prompt_tdata = 0;

        #15 rst = 0;

        weight_file = $fopen("fpga_hbm2_stream.bin", "rb");
        prompt_file = $fopen("prompt_activations.bin", "rb");
        
        if (weight_file == 0 || prompt_file == 0) begin
            $display("[-] CRITICAL ERROR: Bestanden missen!");
            $finish;
        end

        while (!$feof(prompt_file) && !$feof(weight_file)) begin
            幕 = @(posedge clk);
            
            p_bytes = $fread(prompt_buffer, prompt_file);
            w_bytes = $fread(weight_buffer, weight_file);
            
            if (p_bytes > 0 && w_bytes > 0) begin
                hbm2_tvalid = 1;
                prompt_tvalid = 1;
                
                // FIX: Expliciete byte-volgorde indexatie voor Icarus Verilog unpack
                prompt_tdata = {prompt_buffer[0], prompt_buffer[1], prompt_buffer[2], prompt_buffer[3]};
                
                for (idx = 0; idx < 64; idx = idx + 1) begin
                    hbm2_tdata[(idx*8)+:8] = weight_buffer[idx];
                end
            end else begin
                hbm2_tvalid = 0; prompt_tvalid = 0;
            end
        end

        @(posedge clk);
        hbm2_tvalid = 0; prompt_tvalid = 0;
        $fclose(weight_file); $fclose(prompt_file);
        
        #50;
        $finish;
    end

    always @(posedge clk) begin
        if (out_valid) begin
            $strobe("CHIP PREDICTS NEXT TOKEN-ID: %d", chosen_token_id);
        end
    end

endmodule
