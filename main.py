def disassemble(instruction) -> []:
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
    funct: int = (instruction & bitmask_dict["funct"])
    off: int = (instruction & bitmask_dict["off"])

    # print("op: ", hex(op))
    # print("rs: ", hex(rs))
    # print("rt: ", hex(rt))
    # print("rd: ", hex(rd))
    # print("funct: ", hex(funct))
    # print("off: ", hex(off))

    # R-Format instruction
    if op == 0:
        assembly_string = f"{funct_dict[funct]} ${rd}, ${rs}, ${rt}"

    # I-Format instruction
    else:
        # Offset is a signed 16-bit integer, this is not natively supported in python
        # Check if offset is negative by checking if the most significant bit is 1
        if off & 0b1000000000000000:
            off -= 0b10000000000000000  # Subtract range of a signed 16-bit integer (2^16) to get the negative val

        # load and store instructions (lb, sb) use opcodes
        assembly_string = f"{opcode_dict[op]} ${rt}, {off}(${rs})"

    return assembly_string


def initialize_memory(size):
    memory = [i % 0x100 for i in range(size + 1)]
    return memory


def get_hex_list(lst: []):
    lst = [hex(i) for i in lst]
    return lst


def data_pipeline(instruction_cache: [] = None) -> []:
    instruction_counter = 0
    main_mem = initialize_memory(1024)
    regs = [i + 0x100 for i in range(32)]
    print("Initialized Memory and Registers")
    print("Memory: ", get_hex_list(main_mem))
    print("Registers: ", get_hex_list(regs))
    if instruction_cache is None:
        instruction_cache: list = [0xa1020000,
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
                                   0x00000000]

    result_assembly = [disassemble(instruction) for instruction in instruction_cache]
    print(f"Disassembled {len(result_assembly)} Instructions")
    for assembly in result_assembly:
        print(assembly)


if __name__ == "__main__":
    data_pipeline()
