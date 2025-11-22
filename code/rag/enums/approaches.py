from enum import IntFlag, auto
class Approaches(IntFlag):
    OPENAI_KEYWORD = auto()
    OPENAI_SEMANTIC = auto()
    LC_BM25 = auto()
    GRAPH_EAGER = auto()
    GRAPH_MMR = auto()
    VANILLA = auto()
    
    def to_str_list(self):
        gpt_list = [Approaches(x.value) for x in list(Approaches)]
        str_list = []
        for gpt in gpt_list:
            if gpt in self:
                match gpt:
                    case Approaches.OPENAI_KEYWORD:
                        str_list.append("openai_keyword")
                    case Approaches.OPENAI_SEMANTIC:
                        str_list.append("openai_semantic")
                    case Approaches.LC_BM25:
                        str_list.append("lc_bm25")
                    case Approaches.GRAPH_EAGER:
                        str_list.append("graph_eager")
                    case Approaches.GRAPH_MMR:
                        str_list.append("graph_mmr")
                    case Approaches.VANILLA:
                        str_list.append("vanilla")
        return str_list


if __name__ == "__main__":
    gpts = Approaches.OPENAI_KEYWORD | Approaches.OPENAI_SEMANTIC | Approaches.LC_BM25 | Approaches.GRAPH_EAGER | Approaches.VANILLA | Approaches.GRAPH_MMR
    print(gpts.to_str_list())