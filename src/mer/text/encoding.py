"""Use the same target-first representation for training, evaluation and demo."""


def encode_example(tokenizer, target, context="", max_length=256):
    if max_length < tokenizer.num_special_tokens_to_add(pair=False) + 1:
        raise ValueError("max_length leaves no room for text")
    # only_second cannot handle a target that already exceeds the entire budget.
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=bool(context))
    if not context or len(tokenizer.encode(target, add_special_tokens=False)) >= budget:
        return tokenizer(target, truncation=True, max_length=max_length)
    return tokenizer(
        target, context, truncation="only_second", max_length=max_length,
    )
