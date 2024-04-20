from dataclasses import dataclass


def disassemble(instruction) -> dict:
    bitmask_dict = {
        "op": 0b11111100000000000000000000000000,
        "rs": 0b00000011111000000000000000000000,
        "rt": 0b00000000000111110000000000000000,
        "rd": 0b00000000000000001111100000000000,
        "shamt": 0b00000000000000000000011111000000,
        "funct": 0b00000000000000000000000000111111,
        "off": 0b00000000000000001111111111111111,
    }

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
        assembly_str = f"{opcode_dict[op]} ${rt}, {off}(${rs})"

    return assembly_dict


def initialize_memory(size):
    memory = [i % 0x100 for i in range(size + 1)]
    return memory


def get_hex_list(lst: []):
    lst = [hex(i) for i in lst]
    return lst


class PipelineRegister:

    def __init__(self, read, write):
        self.read = read
        self.write = write

    def __str__(self):
        # Print all attributes of the class
        return str(self.__dict__)


class DataPipeline:
    main_mem: []
    regs: []
    instruction_cache: []

    instruction_counter = 0
    completed_instructions = 0

    IF_ID = PipelineRegister(0, 0)
    ID_EX = PipelineRegister({}, 0)
    EX_MEM = PipelineRegister(0, 0)
    MEM_WB = PipelineRegister(0, 0)

    def __init__(self, instruction_cache: []):
        self.main_mem = initialize_memory(1024)
        self.regs = [i + 0x100 for i in range(32)]
        self.instruction_cache = instruction_cache

        # print("Memory: ", get_hex_list(self.main_mem))
        # print("Registers: ", get_hex_list(self.regs))
        print("Initialized Memory and Registers")

        self.IF_ID = PipelineRegister(0, 0)
        self.ID_EX = PipelineRegister({"operation": "nop"}, {"operation": "nop"})
        self.EX_MEM = PipelineRegister({"nop": True}, {"nop": True})
        self.MEM_WB = PipelineRegister({"nop": True}, {"nop": True})

        # Initialise all the pipeline registers
        print("Pipeline Registers Initialized")
        print("\n")

    def fetch(self):
        print(
            "Fetching instruction: ",
            hex(self.instruction_cache[self.instruction_counter]),
        )
        self.IF_ID.write = self.instruction_cache[self.instruction_counter]
        self.instruction_counter += 1
        self.IF_ID.read = self.IF_ID.write

    def decode(self):
        self.ID_EX.write = self.IF_ID.read
        print("Decoding instruction: ", hex(self.ID_EX.write))
        # Disassemble the instruction
        instruction = disassemble(self.ID_EX.write)
        read_register_1 = instruction["rs"]
        read_register_2 = instruction["rt"]
        # Get the data from the read registers
        data_1 = self.regs[read_register_1]
        data_2 = self.regs[read_register_2]

        instruction["data_1"] = data_1
        instruction["data_2"] = data_2

        self.ID_EX.read = instruction

        print("Decoded Instruction: ", instruction)

    def execute(self):
        self.EX_MEM.write = self.ID_EX.read

        print("Executing instruction: ", self.EX_MEM.write["operation"])
        info_dict = {
            "nop": False,  # This is not a no operation instruction
            "lb": False,  # This is a load instruction
            "alu_result": None,
            "data_to_load": None,
            "data_to_store": None,
            "destination_register": None,
        }

        match self.EX_MEM.write["operation"]:
            case "add":
                # Add instruction
                alu_result = self.EX_MEM.write["data_1"] + self.EX_MEM.write["data_2"]
                info_dict["alu_result"] = alu_result
                info_dict["destination_register"] = self.EX_MEM.write["rd"]
            case "sub":
                # Subtract instruction
                alu_result = self.EX_MEM.write["data_1"] - self.EX_MEM.write["data_2"]
                info_dict["alu_result"] = alu_result
                info_dict["destination_register"] = self.EX_MEM.write["rd"]

            case "nop":
                info_dict["nop"] = True
                # No operation
                pass

            case "lb":
                # Load instruction
                destination_register = self.EX_MEM.write["rt"]
                offset = self.EX_MEM.write["off"]
                base_address = self.EX_MEM.write["data_1"]

                info_dict["lb"] = True
                info_dict["alu_result"] = base_address + offset
                info_dict["destination_register"] = destination_register

            case "sb":
                # Store instruction
                destination_register = self.EX_MEM.write["rt"]
                offset = self.EX_MEM.write["off"]
                base_address = self.EX_MEM.write["data_1"]
                data_to_store = self.EX_MEM.write["data_2"]

                info_dict["alu_result"] = base_address + offset
                info_dict["data_to_store"] = data_to_store
                info_dict["destination_register"] = destination_register

        self.EX_MEM.read = info_dict

    def memory_access(self):
        self.MEM_WB.write = self.EX_MEM.read

        if self.MEM_WB.write["lb"]:
            # Load instruction
            data_to_load = self.main_mem[self.MEM_WB.write["alu_result"]]

            print("Memory Accessed at address: ", self.MEM_WB.write["alu_result"])
            print("Memory Value: ", data_to_load)
            self.MEM_WB.write["data_to_load"] = data_to_load

        self.MEM_WB.read = self.MEM_WB.write

    def write_back(self):
        # Write the results back
        data: dict = self.MEM_WB.read

        if data["nop"]:
            print("No operation")
            return

        destination_register = data["destination_register"]
        lb = data["lb"]
        alu_result = data["alu_result"]
        data_to_load = data["data_to_load"]
        data_to_store = data["data_to_store"]

        # load instruction
        if lb:
            self.regs[destination_register] = data_to_load
            print(
                "Loaded data: ", data_to_load, " into register: ", destination_register
            )
        # store instruction
        elif data_to_store:
            self.main_mem[alu_result] = data_to_store
            print("Stored data: ", data_to_store, " at address: ", alu_result)

        # R format instruction
        else:
            self.regs[destination_register] = alu_result
            print("Wrote data: ", alu_result, " to register: ", destination_register)

        self.completed_instructions += 1
        print("Completed Instructions: ", self.completed_instructions)

    def run(self):
        while self.instruction_counter < len(self.instruction_cache):
            print("Instruction Counter: ", self.instruction_counter)
            self.fetch()
            self.decode()
            self.execute()
            self.memory_access()
            self.write_back()
            print("\n")


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
