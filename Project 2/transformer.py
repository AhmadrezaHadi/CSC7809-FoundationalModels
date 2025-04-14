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

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6, dropout=0.2, pad_token_id=0):
        """
        Create a Transformer-based language model.
        :param vocab_size: Size of the vocabulary.
        :param embed_dim: Dimension of the word embeddings.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of Transformer layers.
        :param dropout: Dropout rate.
        :param pad_token_id: Padding token ID.
        """
        super(TransformerModel, self).__init__()

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        
        # Define positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        # Define the Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # Output layer that maps hidden state of final Transformer layer to output
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        """
        Compute the forward pass of the Transformer model.
        :param input_ids: Input token IDs.
        :return: Output logits.
        """
        embeds = self.embedding(input_ids)
        embeds = self.positional_encoding(embeds)
        output = self.transformer_encoder(embeds)
        logits = self.fc(self.dropout(output))
        return logits
    
    def predict_next_token(self, input_ids, temperature=1.0):
        """
        Predict the next token ID from the last token in input_ids.
        :param input_ids: Input token IDs.
        :param temperature: Sampling temperature.
        :return: next token ID
        """
        self.eval()

        with torch.no_grad():
            embeds = self.embedding(input_ids)
            embeds = self.positional_encoding(embeds)
            output = self.transformer_encoder(embeds)
            logits = self.fc(self.dropout(output))
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            return next_token_id.squeeze(0)
        
    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='mps'):
        """
        Generate text based on a prompt.
        :param tokenizer: Tokenizer to convert text to token IDs.
        :param prompt: Input text prompt.
        :param max_length: Maximum length of generated text.
        :param eos_token_id: End-of-sequence token ID (optional).
        :param temperature: Sampling temperature.
        :param device: Device to run the model on ('cpu', 'cuda', or 'mps').
        :return: Generated text.
        """
        self.eval()
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        generated_text = prompt

        for _ in range(max_length):
            next_token_id = self.predict_next_token(input_ids, temperature=temperature)
            generated_text += tokenizer.decode(next_token_id.item())

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)

        return generated_text
    
    
    
