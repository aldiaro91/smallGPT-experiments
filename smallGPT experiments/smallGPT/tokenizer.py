from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

# Inicjalizacja modelu BPE z domyślnym tokenem nieznanym
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Użyj ByteLevel jako pre-tokenizera
tokenizer.pre_tokenizer = ByteLevel()

# Ustaw specjalne tokeny
tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])

# Konfiguracja trenera
trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"], vocab_size=8192)

# Trenowanie tokenizera
files = ["./data-1-file/dane.txt"]
tokenizer.train(files, trainer)

# Ustawienie post-procesora do obsługi tokenów specjalnych dla GPT-2 (opcjonalnie)
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# Zapis tokenizera
tokenizer.save("./smallGPT/tokenizer.json")
