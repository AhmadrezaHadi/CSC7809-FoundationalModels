import torch.nn as nn
import torch




class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.2, pad_token_id=0):
        """
        Initialize the RNN model.

        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimension of the word embeddings.
        :param hidden_dim: Dimension of the hidden state.
        :param num_layers: Number of RNN layers.
        :param dropout: Dropout rate.
        :param pad_token_id: Padding token ID.
        """
        super(RNNModel, self).__init__()

        # Define the embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_token_id
        )

        # Define the RNN layer
        self.rnn = nn.RNN(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )

        # Output layer that maps hidden state of final RNN to output
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
        Compute the forward pass of the RNN model.

        :param input_ids: Input token IDs.
        :param hidden: Initial hidden state (optional).
        :return: Output logits and the final hidden state.
        """
        embeds = self.embedding(input_ids)
        output, hidden = self.rnn(embeds, hidden)
        logits = self.fc(output)

        return logits, hidden
    
    def predict_next_token(self, input_ids, hidden=None, temperature=1.0):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids.
        :param input_ids: Input token IDs.
        :param hidden: Initial hidden state (optional).
        :param temperature: Sampling temperature.
        :return: next token ID, hidden state
        """

        self.eval()

        with torch.no_grad():
            logits, hidden = self.forward(input_ids, hidden)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            return next_token_id.squeeze(0), hidden
        
    # def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='mps'):
    #     """
    #     Generate text based on a prompt.
    #     :param tokenizer: Tokenizer for encoding/decoding.
    #     :param prompt: Input prompt string.
    #     :param max_length: Maximum length of generated text.
    #     :param eos_token_id: End-of-sequence token ID (optional).
    #     :param temperature: Sampling temperature.
    #     :param device: Device to run the model on ('mps', 'cuda', 'cpu').
    #     :return: Generated text string.
    #     """
        
    #     self.eval()
    #     input_ids = tokenizer.encode(prompt, out_type=int)
    #     input_tensor = torch.tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)  # shape (1, seq_len)

    #     generated_ids = []
    #     hidden = None

    #     for _ in range(max_length):
    #         next_token_id, hidden = self.predict_next_token(input_tensor, hidden, temperature)

    #         if eos_token_id is not None and next_token_id.item() == eos_token_id:
    #             break

    #         generated_ids.append(next_token_id.item())
    #         input_tensor = next_token_id.unsqueeze(0)

    #     return tokenizer.decode(generated_ids, out_type=str)

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='mps'):
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        generated_ids = input_ids.copy()

        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        hidden = None

        for _ in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor, hidden, temperature)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            generated_ids.append(next_token_id.item())
            input_tensor = next_token_id.unsqueeze(0).to(device)  # shape [1, 1]

        return tokenizer.decode(generated_ids, out_type=str)

    
