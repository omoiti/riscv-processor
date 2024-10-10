// Program Counter: Holds the current instruction address
module ProgramCounter (
    input clk, reset,
    input [31:0] pc_in,  // Next PC value (input)
    output reg [31:0] pc_out  // Current PC value (output)
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            pc_out <= 32'b0;  // Reset PC to 0
        else
            pc_out <= pc_in;  // Update PC with next value
    end
endmodule

// PC + 4 Adder: Calculates the next instruction address (PC + 4)
module PCAdder (
    input [31:0] current_pc,  // Current PC value
    output [31:0] next_pc     // Next PC value (PC + 4)
);
    assign next_pc = current_pc + 4;
endmodule

// Instruction Memory: Stores and outputs instructions based on read address
module InstructionMemory (
    input clk, reset,
    input [31:0] read_addr,  // PC used as read address
    output reg [31:0] instruction  // Fetched instruction
);
    reg [31:0] instruction_mem [63:0];  // 64-word instruction memory
    integer i;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Initialize instruction memory with zeros on reset
            for (i = 0; i < 64; i = i + 1)
                instruction_mem[i] <= 32'b0;
        end else begin
            instruction <= instruction_mem[read_addr];
        end
    end
endmodule

// Register File: Stores register values and supports reading/writing
module RegisterFile (
    input clk, reset,
    input reg_write,         // Write enable signal
    input [4:0] rs1, rs2, rd,  // Register addresses
    input [31:0] write_data,  // Data to write to 'rd'
    output [31:0] read_data1, read_data2  // Data read from 'rs1' and 'rs2'
);
    reg [31:0] registers [31:0];  // 32 registers
    integer i;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Initialize registers to zero on reset
            for (i = 0; i < 32; i = i + 1)
                registers[i] <= 32'b0;
        end else if (reg_write) begin
            registers[rd] <= write_data;  // Write data to register 'rd'
        end
    end

    assign read_data1 = registers[rs1];  // Read from 'rs1'
    assign read_data2 = registers[rs2];  // Read from 'rs2'
endmodule

// Immediate Generator: Generates sign-extended immediate values from instructions
module ImmediateGenerator (
    input [6:0] opcode,           // Opcode to determine immediate format
    input [31:0] instruction,     // Instruction input
    output reg [31:0] imm_ext     // Sign-extended immediate output
);
    always @(*) begin
        case (opcode)
            7'b0000011: imm_ext = {{20{instruction[31]}}, instruction[31:20]};  // Load
            7'b0100011: imm_ext = {{20{instruction[31]}}, instruction[31:25], instruction[11:7]};  // Store
            7'b1100011: imm_ext = {{19{instruction[31]}}, instruction[31], instruction[30:25], instruction[11:8], 1'b0};  // Branch
            default: imm_ext = 32'b0;  // Default case
        endcase
    end
endmodule

// Control Unit: Generates control signals based on the opcode
module ControlUnit (
    input [6:0] opcode,  // Opcode from the instruction
    output reg branch, mem_read, mem_to_reg, mem_write, alu_src, reg_write,
    output reg [1:0] alu_op  // ALU operation code
);
    always @(*) begin
        case (opcode)
            7'b0110011: {alu_src, mem_to_reg, reg_write, mem_read, mem_write, branch, alu_op} <= 8'b001000_01;  // R-type
            7'b0000011: {alu_src, mem_to_reg, reg_write, mem_read, mem_write, branch, alu_op} <= 8'b111100_00;  // Load
            7'b0100011: {alu_src, mem_to_reg, reg_write, mem_read, mem_write, branch, alu_op} <= 8'b100010_00;  // Store
            7'b1100011: {alu_src, mem_to_reg, reg_write, mem_read, mem_write, branch, alu_op} <= 8'b000001_10;  // Branch
            default: {alu_src, mem_to_reg, reg_write, mem_read, mem_write, branch, alu_op} <= 8'b0;
        endcase
    end
endmodule

// ALU: Executes arithmetic and logic operations
module ALU (
    input [31:0] a, b,         // Inputs
    input [3:0] alu_control,   // Control signal to determine the operation
    output reg [31:0] alu_result,  // Result of ALU operation
    output reg zero  // Zero flag, raised if result is zero
);
    always @(*) begin
        case (alu_control)
            4'b0000: alu_result = a & b;  // AND operation
            4'b0001: alu_result = a | b;  // OR operation
            4'b0010: alu_result = a + b;  // ADD operation
            4'b0110: begin
                alu_result = a - b;        // SUB operation
                zero = (alu_result == 0);  // Set zero flag if result is zero
            end
            default: alu_result = 32'b0;
        endcase
    end
endmodule

// ALU Control: Determines ALU operation based on ALUOp, func7, and func3
module ALUControl (
    input [1:0] alu_op,        // ALUOp from control unit
    input func7,               // Func7 bit from instruction
    input [2:0] func3,         // Func3 field from instruction
    output reg [3:0] alu_control  // Control signal for ALU
);
    always @(*) begin
        case ({alu_op, func7, func3})
            6'b00_0_000: alu_control = 4'b0010;  // ADD for Load/Store
            6'b01_0_000: alu_control = 4'b0110;  // SUB for Branch
            6'b10_0_000: alu_control = 4'b0010;  // ADD for R-type
            6'b10_1_000: alu_control = 4'b0110;  // SUB for R-type
            6'b10_0_111: alu_control = 4'b0000;  // AND for R-type
            6'b10_0_110: alu_control = 4'b0001;  // OR for R-type
            default: alu_control = 4'b0000;
        endcase
    end
endmodule

// Data Memory: Reads/writes data based on the control signals
module DataMemory (
    input clk, reset,
    input mem_write, mem_read,
    input [31:0] address, write_data,
    output [31:0] mem_data
);
    reg [31:0] data_mem [63:0];  // 64-word data memory
    integer i;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Initialize memory on reset
            for (i = 0; i < 64; i = i + 1)
                data_mem[i] <= 32'b0;
        end else if (mem_write) begin
            data_mem[address] <= write_data;  // Write data to memory
        end
    end

    assign mem_data = (mem_read) ? data_mem[address] : 32'b0;  // Read from memory
endmodule

// 2-to-1 MUX: Selects between two inputs based on the select signal
module Mux2to1 (
    input sel,  // Select signal
    input [31:0] in0, in1,  // Two inputs
    output [31:0] out  // Selected output
);
    assign out = (sel == 1'b0) ? in0 : in1;
endmodule

// AND Gate: Generates a signal based on branch and zero flags
module AndGate (
    input branch, zero,
    output and_out
);
    assign and_out = branch & zero;  // AND operation
endmodule

// Adder: Adds two 32-bit inputs
module Adder (
    input [31:0] in1, in2,
    output [31:0] sum
);
    assign sum = in1 + in2;
endmodule

// Top Module: Integrates all components of the CPU
module Top (
    input clk, reset  // Clock and reset signals
);
    // Internal wires for interconnecting various modules
    wire [31:0] pc_current, pc_next, instruction;
    wire [31:0] imm_ext, alu_result, read_data1, read_data2, mem_data;
    wire [31:0] alu_input, pc_imm_adder_out;
    wire branch, mem_read, mem_to_reg, mem_write, alu_src, reg_write, zero;
    wire [1:0] alu_op;
    wire [3:0] alu_control;
    wire branch_and_zero;  // AND gate output (branch & zero)

    // Program Counter: Holds the current instruction address
    ProgramCounter pc_inst(
        .clk(clk), .reset(reset), 
        .pc_in(pc_next), .pc_out(pc_current)
    );
    
    // PC Adder: Calculates PC + 4 (next sequential PC)
    PCAdder pc_adder(
        .current_pc(pc_current), 
        .next_pc(pc_next)
    );
    
    // Instruction Memory: Fetches instruction from memory
    InstructionMemory imem_inst(
        .clk(clk), .reset(reset), 
        .read_addr(pc_current), 
        .instruction(instruction)
    );

    // Register File: Reads from and writes to registers
    RegisterFile rf_inst(
        .clk(clk), .reset(reset), 
        .reg_write(reg_write), 
        .rs1(instruction[19:15]), .rs2(instruction[24:20]), .rd(instruction[11:7]), 
        .write_data(mem_data), 
        .read_data1(read_data1), .read_data2(read_data2)
    );
    
    // Immediate Generator: Generates sign-extended immediate values
    ImmediateGenerator imm_gen_inst(
        .opcode(instruction[6:0]), 
        .instruction(instruction), 
        .imm_ext(imm_ext)
    );
    
    // Control Unit: Generates control signals based on opcode
    ControlUnit ctrl_inst(
        .opcode(instruction[6:0]), 
        .branch(branch), .mem_read(mem_read), .mem_to_reg(mem_to_reg), 
        .mem_write(mem_write), .alu_src(alu_src), .reg_write(reg_write), 
        .alu_op(alu_op)
    );

    // 2-to-1 Mux: Selects ALU input between register data and immediate
    Mux2to1 alu_src_mux(
        .sel(alu_src), 
        .in0(read_data2), .in1(imm_ext), 
        .out(alu_input)
    );
    
    // ALU Control: Determines which operation the ALU should perform
    ALUControl alu_ctrl_inst(
        .alu_op(alu_op), 
        .func7(instruction[30]), .func3(instruction[14:12]), 
        .alu_control(alu_control)
    );

    // ALU: Executes arithmetic/logic operations
    ALU alu_inst(
        .a(read_data1), .b(alu_input), 
        .alu_control(alu_control), 
        .alu_result(alu_result), 
        .zero(zero)
    );

    // Data Memory: Reads or writes data based on control signals
    DataMemory dmem_inst(
        .clk(clk), .reset(reset), 
        .mem_write(mem_write), .mem_read(mem_read), 
        .address(alu_result), .write_data(read_data2), 
        .mem_data(mem_data)
    );

    // AND Gate: Branch if condition (branch & zero)
    AndGate and_gate_inst(
        .branch(branch), .zero(zero), 
        .and_out(branch_and_zero)
    );

    // PC + Immediate Adder: Calculates target address for branch instructions
    Adder pc_imm_adder(
        .in1(pc_current), .in2(imm_ext), 
        .sum(pc_imm_adder_out)
    );
    
    // 2-to-1 Mux: Selects next PC value (PC + 4 or branch target)
    Mux2to1 pc_mux(
        .sel(branch_and_zero), 
        .in0(pc_next), .in1(pc_imm_adder_out), 
        .out(pc_next)
    );

endmodule
