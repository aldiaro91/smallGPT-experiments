from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import os

# Ścieżka do istniejącego tokenizera i nowego zbioru danych
existing_tokenizer_path = "./smallGPT/"
new_data_file = "./data-1-file/dane.txt"
new_tokenizer_path = "./smallGPT/tokenizer.json"

# Wczytanie istniejącego tokenizera
tokenizer = Tokenizer.from_file(os.path.join(existing_tokenizer_path, "tokenizer.json"))

# Konfiguracja trenera
trainer = trainers.BpeTrainer(
    vocab_size=8192,  # Nowa maksymalna liczba tokenów
    min_frequency=2,  # Minimalna częstotliwość występowania tokenów
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Ponowne trenowanie tokenizera
files = [new_data_file]
tokenizer.train(files, trainer)

# Zapisanie nowo wytrenowanego tokenizera
tokenizer.save(new_tokenizer_path)

print(f"Tokenizer został przetrenowany i zapisany w {new_tokenizer_path}")
