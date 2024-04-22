from dataclasses import dataclass

bitmask_dict = {
    "op": 0b11111100000000000000000000000000,
    "rs": 0b00000011111000000000000000000000,
    "rt": 0b00000000000111110000000000000000,
    "rd": 0b00000000000000001111100000000000,
    "shamt": 0b00000000000000000000011111000000,
    "funct": 0b00000000000000000000000000111111,
    "off": 0b00000000000000001111111111111111,
}


def disassemble(instruction) -> dict:

    funct_dict = {
        0x20: "add",
        0x22: "sub",
        0x0: "nop",
    }

    opcode_dict = {
        0x20: "lb",
        0x28: "sb",
    }

    # Extracting fields from the instruction
    op: int = (instruction & bitmask_dict["op"]) >> 32 - 6
    rs: int = (instruction & bitmask_dict["rs"]) >> 32 - 11
    rt: int = (instruction & bitmask_dict["rt"]) >> 32 - 16
    rd: int = (instruction & bitmask_dict["rd"]) >> 32 - 21
    # funct and offset do not need to be shifted
    funct: int = instruction & bitmask_dict["funct"]
    off: int = instruction & bitmask_dict["off"]

    assembly_dict = {
        "operation": None,
        "rs": None,
        "rt": None,
        "rd": None,
        "off": None,
    }

    # R-Format instruction
    if op == 0:
        assembly_dict["operation"] = funct_dict[funct]
        assembly_dict["rs"] = rs
        assembly_dict["rt"] = rt
        assembly_dict["rd"] = rd
        assembly_str = f"{funct_dict[funct]} ${rd}, ${rs}, ${rt}"

    # I-Format instruction
    else:
        # Offset is a signed 16-bit integer, this is not natively supported in python
        # Check if offset is negative by checking if the most significant bit is 1
        if off & 0b1000000000000000:
            off -= 0b10000000000000000  # Subtract range of a signed 16-bit integer (2^16) to get the negative val

        assembly_dict["operation"] = opcode_dict[op]
        assembly_dict["rs"] = rs
        assembly_dict["rt"] = rt
        assembly_dict["off"] = off

        # load and store instructions (lb, sb) use opcodes
        # rt is the destination register for lb, and the source register for sb
        assembly_str = f"{opcode_dict[op]} ${rt}, {off}(${rs})"

    return assembly_dict


def initialize_memory(size):
    memory = [i % 0x100 for i in range(size + 1)]
    return memory


def get_hex_list(lst: []):
    lst = [hex(i) for i in lst]
    return lst


@dataclass
class ControlSignals:
    """
    Control Signals Explanation:

    - RegDst (Register Destination):
      Determines the register destination for the result of an ALU operation.
      - If RegDst=1, the destination register is 'rd'; if RegDst=0, it is 'rt'.

    - ALUSrc (ALU Source):
      Specifies the second operand of the ALU operation.
      - if ALUSrc=0, it is the value from register 'rt'; if ALUSrc=1, it is the immediate value (offset) from the instruction.

    - ALUOp (ALU Operation):
      Specifies the operation to be performed by the ALU based on the opcode or function code of the instruction.
        - ALUOp=1 for R-format instructions; ALUOp=0 for I-format instructions.

    - MemRead (Memory Read):
      Indicates whether the instruction involves reading data from memory.
      - MemRead=1 for load instructions ('lw', 'lb'); MemRead=0 for other instructions.

    - MemWrite (Memory Write):
      Indicates whether the instruction involves writing data to memory.
      - MemWrite=1 for store instructions ('sw', 'sb'); MemWrite=0 for other instructions.

    - MemToReg (Memory to Register):
      Specifies whether data read from memory should be written back to a register.
      - MemToReg=1 for load instructions; MemToReg=0 for other instructions.

    - RegWrite (Register Write):
      Indicates whether the result of an ALU operation should be written back to a register.
      - RegWrite=1 for instructions producing a result; RegWrite=0 for branch or jump instructions.

    """

    RegDst: int
    ALUSrc: int
    ALUOp: int
    MemRead: int
    MemWrite: int
    MemToReg: int
    RegWrite: int


control_signals_dict = {
    "lb": {
        "RegDst": 0,  # RegDst=0 because the result is stored in 'rt' (destination register)
        "ALUSrc": 1,  # ALUSrc=1 because the offset is an immediate value
        "ALUOp": 0,  # ALUOp does not apply to this memory operation
        "MemRead": 1,  # MemRead=1 to read data from memory
        "MemWrite": 0,  # MemWrite=0 because it's a load operation, not a store
        "MemToReg": 1,  # MemToReg=1 because the data from memory is written to a register
        "RegWrite": 1,  # RegWrite=1 to write the data to a register
    },
    "sb": {
        "RegDst": 0,  # RegDst=0 because there's no result stored in a register
        "ALUSrc": 1,  # ALUSrc=1 because the offset is an immediate value
        "ALUOp": 0,  # ALUOp does not apply to this memory operation
        "MemRead": 0,  # MemRead=0 because it's a store operation
        "MemWrite": 1,  # MemWrite=1 to write data to memory
        "MemToReg": 0,  # MemToReg does not apply to this operation
        "RegWrite": 0,  # RegWrite=0 because there's no result to be written to a register
    },
    "add": {
        "RegDst": 1,  # RegDst=1 because the result is stored in 'rd' (destination register)
        "ALUSrc": 0,  # ALUSrc=0 because operands come from registers 'rs' and 'rt'
        "ALUOp": 1,  # ALUOp=1 specifies that this is a R-format operation and so we will use the funct field
        "MemRead": 0,  # MemRead=0 because it's not a memory operation
        "MemWrite": 0,  # MemWrite=0 because it's not a memory operation
        "MemToReg": 0,  # MemToReg does not apply to this operation
        "RegWrite": 1,  # RegWrite=1 to write the result to a register
    },
    "sub": {
        "RegDst": 1,  # RegDst=1 because the result is stored in 'rd' (destination register)
        "ALUSrc": 0,  # ALUSrc=0 because operands come from registers 'rs' and 'rt'
        "ALUOp": 1,  # ALUOp=1 specifies that this is a R-format operation and so we will use the funct field
        "MemRead": 0,  # MemRead=0 because it's not a memory operation
        "MemWrite": 0,  # MemWrite=0 because it's not a memory operation
        "MemToReg": 0,  # MemToReg does not apply to this operation
        "RegWrite": 1,  # RegWrite=1 to write the result to a register
    },
}


@dataclass
class PipelineRegister:
    write: dict
    read: dict


def get_nop_control_signals():
    return ControlSignals(
        RegDst=0,
        ALUSrc=0,
        ALUOp=0,
        MemRead=0,
        MemWrite=0,
        MemToReg=0,
        RegWrite=0,
    )


def update_control_signals(disassembled_instruction):
    # Set the control signals based on the operation
    operation = disassembled_instruction["operation"]
    return control_signals_dict[operation]


def initialise_noop_pipeline_register():
    return PipelineRegister(
        read={"nop": "True", "control_signals": get_nop_control_signals()},
        write={"nop": "True", "control_signals": get_nop_control_signals()},
    )


class DataPipeline:
    main_mem: []
    regs: []
    instruction_cache: []

    clock_cycle = 0
    PC = 0

    def __init__(self, instruction_cache: []):
        self.main_mem = initialize_memory(1024)
        self.regs = [i + 0x100 for i in range(32)]
        self.instruction_cache = instruction_cache

        # Initialize all the pipeline registers to nop
        self.IF_ID = initialise_noop_pipeline_register()
        self.ID_EX = initialise_noop_pipeline_register()
        self.EX_MEM = initialise_noop_pipeline_register()
        self.MEM_WB = initialise_noop_pipeline_register()

        # Print the initial state of the pipeline
        print("Initialized Memory, Registers and Pipeline Registers")
        print("Main_Memory: ", self.main_mem)
        print("Registers: ", self.regs)
        print()
        print("Pipeline Registers: ")
        print("IF_ID", self.IF_ID)
        print("ID_EX", self.ID_EX)
        print("EX_MEM", self.EX_MEM)
        print("MEM_WB", self.MEM_WB)
        print("\n")

    # Prints the contents of regs and all pipeline registers
    def print_out_everything(self):
        print("Clock cycle: ", self.clock_cycle + 1)
        print("Registers: ", self.regs)
        print("IF_ID: ", self.IF_ID)
        print("ID_EX: ", self.ID_EX)
        print("EX_MEM: ", self.EX_MEM)
        print("MEM_WB: ", self.MEM_WB)
        print("\n")

    def copy_write_to_read(self):
        # Copy the write of the previous stage to the read of the current stage
        self.IF_ID.read = self.IF_ID.write.copy()
        self.ID_EX.read = self.ID_EX.write.copy()
        self.EX_MEM.read = self.EX_MEM.write.copy()
        self.MEM_WB.read = self.MEM_WB.write.copy()

    def IF_stage(self):
        # Get the instruction from the instruction cache
        instruction = self.instruction_cache[self.PC // 4]
        self.PC += 4
        self.IF_ID.write["instruction"] = instruction
        self.IF_ID.write["nop"] = False

        if instruction == 0x00000000:
            self.IF_ID.write["nop"] = True

        # Set the default control signals for the instruction
        self.IF_ID.write["control_signals"] = get_nop_control_signals()

    def ID_stage(self):
        # Copy the read from the previous stage to the write of the current stage
        self.ID_EX.write = self.IF_ID.read

        # If the instruction is a nop, do not execute the stage
        if self.ID_EX.write["nop"]:
            return

        # Decipher the instruction
        instruction = self.ID_EX.write["instruction"]
        disassembled_instruction = disassemble(instruction)

        # Update the control signals
        self.ID_EX.write["control_signals"] = update_control_signals(
            disassembled_instruction
        )

        # Extract the funct field from the instruction if it is an R-format instruction
        operation = disassembled_instruction["operation"]
        if operation in ["add", "sub"]:
            self.ID_EX.write["funct"] = operation
        else:
            self.ID_EX.write["funct"] = None

        # Get the values of the read registers
        read_reg_1 = disassembled_instruction["rs"]
        read_reg_2 = disassembled_instruction["rt"]
        self.ID_EX.write["read_reg_1_value"] = self.regs[read_reg_1]
        self.ID_EX.write["read_reg_2_value"] = self.regs[read_reg_2]

        # Set the SEoffset
        self.ID_EX.write["SEoffset"] = disassembled_instruction["off"]

        # Set the write registers
        self.ID_EX.write["write_reg_20_16"] = disassembled_instruction["rt"]
        self.ID_EX.write["write_reg_15_11"] = disassembled_instruction["rd"]

    def EX_stage(self):
        self.EX_MEM.write = self.ID_EX.read

        if self.EX_MEM.write["nop"]:
            return

        control_signals = self.EX_MEM.write["control_signals"]

        # Get the values of the read registers
        read_reg_1_value = self.EX_MEM.write["read_reg_1_value"]
        read_reg_2_value = self.EX_MEM.write["read_reg_2_value"]
        SEoffset = self.EX_MEM.write["SEoffset"]

        ALUOp = control_signals["ALUOp"]

        # ALU operation
        # Checking if the instruction is an I format instruction
        if ALUOp == 0:
            ALUResult = read_reg_1_value + SEoffset
        elif ALUOp == 1:
            # For r format instructions
            # Get the funct field from the instruction
            funct = self.EX_MEM.write["funct"]
            if funct == "add":
                ALUResult = read_reg_1_value + read_reg_2_value
            elif funct == "sub":
                ALUResult = read_reg_1_value - read_reg_2_value

        self.EX_MEM.write["ALUResult"] = ALUResult

    def MEM_stage(self):
        self.MEM_WB.write = self.EX_MEM.read

        if self.MEM_WB.write["nop"]:
            return

        control_signals = self.MEM_WB.write["control_signals"]

        # Examine the control signals to determine the operation
        # lb operation
        if control_signals["MemRead"]:
            # The target address is the ALU result
            address = self.MEM_WB.write["ALUResult"]
            # Load the byte from memory
            LBdata = self.main_mem[address]
            # Write the data to the pipeline register
            self.MEM_WB.write["LBdata"] = LBdata

        # sb operation
        elif control_signals["MemWrite"]:
            # The target address is the ALU result
            address = self.MEM_WB.write["ALUResult"]
            # The data to be written is the value in the second read register
            data = self.MEM_WB.write["read_reg_2_value"]
            # Write the data to memory
            self.main_mem[address] = data

    def WB_stage(self):
        final_dict = self.MEM_WB.read
        if final_dict["nop"]:
            return

        control_signals = final_dict["control_signals"]
        MemToReg = control_signals["MemToReg"]
        RegWrite = control_signals["RegWrite"]

        # load instruction
        if MemToReg and RegWrite:
            # Get the value to be written to the register
            write_data = final_dict["LBdata"]
            # Get the register to write to
            write_reg = final_dict["write_reg_20_16"]
            self.regs[write_reg] = write_data

        # R-format instruction
        elif not MemToReg and RegWrite:
            write_data = final_dict["ALUResult"]
            write_reg = final_dict["write_reg_15_11"]
            self.regs[write_reg] = write_data

    # Run the pipeline
    def run(self):

        print("Running the pipeline...")
        print("\n")
        print("Instruction Cache: ")
        counter = 1
        for instruction in self.instruction_cache:
            print(f"Instruction {counter}: ", instruction)
            counter += 1

        print("\n")

        while self.PC // 4 < len(self.instruction_cache):
            self.IF_stage()
            self.ID_stage()
            self.EX_stage()
            self.MEM_stage()
            self.WB_stage()
            self.print_out_everything()
            self.copy_write_to_read()

            self.clock_cycle += 1
            print("\n")


# Running the data pipeline simulation
if __name__ == "__main__":
    pipeline: DataPipeline = DataPipeline(
        instruction_cache=[
            0xA1020000,
            0x810AFFFC,
            0x00831820,
            0x01263820,
            0x01224820,
            0x81180000,
            0x81510010,
            0x00624022,
            0x00000000,
            0x00000000,
            0x00000000,
            0x00000000,
            0x00000000,
        ]
    )
    pipeline.run()
