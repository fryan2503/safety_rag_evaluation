from enum import IntFlag, auto
class LLM(IntFlag):
    GPT_5_MINI_2025_08_07 = auto()
    GPT_5_NANO_2025_08_07 = auto()
    
    def to_str_list(self):
        gpt_list = [LLM(x.value) for x in list(LLM)]
        str_list = []
        for gpt in gpt_list:
            if gpt in self:
                match gpt:
                    case LLM.GPT_5_MINI_2025_08_07:
                        str_list.append("gpt-5-mini-2025-08-07")
                    case LLM.GPT_5_NANO_2025_08_07:
                        str_list.append("gpt-5-nano-2025-08-07")
        return str_list


if __name__ == "__main__":
    gpts = LLM.GPT_5_MINI_2025_08_07 | LLM.GPT_5_NANO_2025_08_07
    print(gpts.to_str_list())
    gpts &= ~LLM.GPT_5_NANO_2025_08_07
    print(gpts.to_str_list())