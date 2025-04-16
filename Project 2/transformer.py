import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.2):
        super(DecoderOnlyLM, self).__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model  # Dimension of the model
        self.nhead = nhead
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask=mask)
        x = self.fc_out(x)
        return x, None

    def predict_next_token(self, input_ids, temperature=1.0):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids.
        :param input_ids: Input token IDs.
        :param temperature: Sampling temperature.
        :return: next token ID, hidden state
        """

        self.eval()

        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            return next_token_id.squeeze(0)
        
    def generate(self, tokenizer, prompt, max_length=50, eos_token_ids=None, temperature=1.0, device="cuda"):
        """
        Generate text given a prompt.
        :param tokenizer: Tokenizer for encoding/decoding.
        :param prompt: Input prompt.
        :param max_length: Maximum length of the generated sequence.
        :param eos_token_ids: End-of-sequence token IDs (optional).
        :param temperature: Sampling temperature.
        :param device: Device to run the model on ('mps', 'cuda', 'cpu').
        :return: Generated text.
        """

        self.eval()

        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
        generated_ids = []

        for _ in range(max_length):
            next_token_id = self.predict_next_token(input_tensor, temperature)
            generated_ids.append(next_token_id.item())
            input_tensor = torch.cat([input_tensor, next_token_id.unsqueeze(0)], dim=1)
            if eos_token_ids and next_token_id.item() in eos_token_ids:
                break
        generated_text = tokenizer.decode(generated_ids)
        return generated_text
    
    
