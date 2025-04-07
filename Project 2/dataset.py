from torch.utils.data import Dataset
import torch
import json



class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_sequence_length=128):
        """
        Create a text dataset for PyTorch Dataset that handles our jsonl prompts+completions
        for causal LM.
        :param file_path: Path to the JSONL file containing prompts and completions.
        :param tokenizer: Tokenizer object for encoding text.
        :param max_sequence_length: Maximum sequence length for tokenization.
        """
        super().__init__()

        self.samples = []
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item["prompt"] + item["completion"]
                token_ids = tokenizer.encode(text, out_type=int)[:self.max_sequence_length]
                if len(token_ids) < 2:
                    continue

                self.samples.append(token_ids)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get and format a sample at the given index.
        for causual LM, we will train the model to predict every next token in the sequence give the previous tokens.
        :param idx: Index of the sample.
        :return: Tuple of input IDs and target IDs.
        """
        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids



