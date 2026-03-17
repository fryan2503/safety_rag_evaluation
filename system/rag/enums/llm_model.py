"""
LLM - enum representing available model choices.

Like Approaches, this is implemented using IntFlag so that multiple
models can be OR'ed together in an experiment configuration.
"""

from enum import IntFlag, auto

class LLM(IntFlag):
    """
    Model identifiers for supported generation models.

    Each entry corresponds to a specific model string reference,
    resolvable via OpenAI or Anthropic model APIs.
    """
    GPT_5_MINI_2025_08_07 = auto()
    GPT_5_NANO_2025_08_07 = auto()
    GPT_5_4_2025_08_07 = auto()
    CLAUDE_OPUS_4_6 = auto()

    def to_str_list(self):
        """
        Converts enabled model flags into concrete model identifiers.

        Example:
            LLM.GPT_5_MINI_2025_08_07 => ["gpt-5-mini-2025-08-07"]
        """
        gpt_list = [LLM(x.value) for x in list(LLM)]
        str_list = []
        for gpt in gpt_list:
            if gpt in self:
                match gpt:
                    case LLM.GPT_5_MINI_2025_08_07:
                        str_list.append("gpt-5-mini-2025-08-07")
                    case LLM.GPT_5_NANO_2025_08_07:
                        str_list.append("gpt-5-nano-2025-08-07")
                    case LLM.GPT_5_4_2025_08_07:
                        str_list.append("gpt-5.4-2026-03-05")
                    case LLM.CLAUDE_OPUS_4_6:
                        str_list.append("claude-opus-4-6")
        return str_list


if __name__ == "__main__":
    gpts = LLM.GPT_5_MINI_2025_08_07 | LLM.GPT_5_NANO_2025_08_07
    print(gpts.to_str_list())
    gpts &= ~LLM.GPT_5_NANO_2025_08_07
    print(gpts.to_str_list())
