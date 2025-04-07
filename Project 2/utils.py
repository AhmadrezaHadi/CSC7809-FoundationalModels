from typing import Tuple
import os
import sentencepiece as spm
from torch import nn


def add_special_tokens(pairs: Tuple[list]):
    """
    Insert <bos> and <eos> tokens into the prompts and completions.
    :param pairs: A tuple of lists containing prompts and completions.
    :return: A tuple of lists with the modified prompts and completions.
    """
    new_prompts = []
    new_completions = []

    for prompt, completion in pairs:

        if prompt[0].isupper():
            prompt = '<bos>' + prompt
        if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
            completion = completion + '<eos>'
        new_prompts.append(prompt)
        new_completions.append(completion)

    return new_prompts, new_completions


def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=3)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch
