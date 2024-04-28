import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# Załaduj wcześniej wytrenowany tokenizer i model
tokenizer_path = "./smallGPT/tokenizer.json"
model_path = "./smallGPT/model"
#model_path = "EleutherAI/pythia-31m"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, pad_token="[PAD]", sep_token="[SEP]", cls_token="[CLS]", mask_token="[MASK]", unk_token="[UNK]")
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto')
#model = GPT2LMHeadModel.from_pretrained(model_path)

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenized_text = tokenizer.encode(text)
        input_ids = []
        labels = []
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            input_ids.append(tokenized_text[i:i+block_size])
            labels.append(tokenized_text[i+1:i+block_size+1])
        self.examples = {'input_ids': input_ids, 'labels': labels}

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, i):
        input_ids = torch.tensor(self.examples['input_ids'][i])
        labels = torch.tensor(self.examples['labels'][i])
        return {'input_ids': input_ids, 'labels': labels}

block_size = 1024
file_path = "./data-1-file/$$$.txt"
dataset = TextDataset(tokenizer, file_path, block_size)
device = torch.device("cuda")
model.to(device)

# Wyświetl liczbę parametrów
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

count_parameters(model)

training_args = TrainingArguments(
    output_dir="./smallGPT/results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-3,
    lr_scheduler_type="constant",
    save_steps=900_000,
    save_total_limit=2,
    logging_dir='./smallGPT/logs',
    logging_steps=10,
    do_train=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

tokenizer.save_pretrained("./smallGPT/model")
model.save_pretrained("./smallGPT/model")
print("Trening zakończony. Model został zaktualizowany.")
