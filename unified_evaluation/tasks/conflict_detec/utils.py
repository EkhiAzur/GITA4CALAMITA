def doc_to_text(x):

    PRE_PROMPT = "The story is as follows: "
    POST_PROMPT = "The conflicting sentence and the breakpoint are:"

    instance = PRE_PROMPT + "\n"

    for i, sentence in enumerate(x["sentences"]):
        instance += f'{i}. {sentence}\n'

    instance += "\n"

    instance += POST_PROMPT

    return instance

def doc_to_target(x):
    return f"{x['confl_sents'][0]} and {x['breakpoint']}"