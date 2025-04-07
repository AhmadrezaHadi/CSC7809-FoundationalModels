import os
import sentencepiece as spm
from tqdm import tqdm

def merge_text_files(data_dir, output_file):
    """
    Merge all text files in a directory into a single file.
    :param data_dir: Directory containing text files.
    :param output_file: Output file path.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in tqdm(files, desc="Merging text files"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

if __name__ == "__main__":
    DATA_DIR = "./data/raw"
    TOKENIZER_PREFIX = "bpe_tokenizer"
    VOCAB_SIZE = 10000
    CORPUS_FILE = "corpus.txt"

    if not os.path.exists(CORPUS_FILE):
        print("Merging text files into corpus...")
        merge_text_files(DATA_DIR, CORPUS_FILE)

    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.Train(
        input=CORPUS_FILE,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"]),
    )

    print(f"Tokenizer trained and saved as {TOKENIZER_PREFIX}.model and {TOKENIZER_PREFIX}.vocab")
