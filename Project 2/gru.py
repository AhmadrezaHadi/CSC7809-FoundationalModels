from torch import nn
import torch


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6, dropout=0.2, pad_token_id=0):
        """
        Create a GRU-based language model.
        :param vocab_size: Size of the vocabulary.
        :param embed_dim: Dimension of the word embeddings.
        :param hidden_dim: Dimension of the GRU hidden state.
        :param num_layers: Number of GRU layers.
        :param dropout: Dropout rate.
        :param pad_token_id: Padding token ID.
        """
        super(GRULanguageModel, self).__init__()

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        # Define the stacked GRU
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # Output layer that maps hidden state of final GRU to output
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, hidden=None):
        """
        Compute the forward pass of the GRU model.
        :param input_ids: Input token IDs.
        :param hidden: Initial hidden state (optional).
        :return: Output logits and the final hidden state.
        """

        embeds = self.embedding(input_ids)
        output, hidden = self.gru(embeds, hidden)
        logits = self.fc(output)

        return logits, hidden
    
    def predict_next_token(self, input_ids, temperature=1.0):
        """
        Predict the next token ID (and hidden state) from the last token in input_file.
        :param input_ids: Input token IDs.
        :param temprature: Sampling temperature.
        :return: next token ID, hidden state
        """

        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1)
            return next_token_id, hidden
    
    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='mps'):
        """
        Generate a full output sequence from a prompt.

        :param tokenizer: trained SentencePiece tokenizer.
        :param prompt: Input prompt string.
        :param max_length: Maximum length of the generated sequence.
        :param eos_token_id: End-of-sequence token ID.
        :param temperature: Sampling temperature.
        :param device: Device to run the model on ('cpu', 'cuda', or 'mps').
        
        :return: Generated sequence as a string.
        """

        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        generated_ids = []
        hidden = None

        for _ in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor, temperature)
            
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        return tokenizer.decode(generated_ids, out_type=str)
        