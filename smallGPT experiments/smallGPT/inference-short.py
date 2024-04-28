from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM

# Załaduj wytrenowany tokenizer
tokenizer_path = "./smallGPT/model"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Załaduj model
model_path = "./smallGPT/model"
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m").cuda()

# Przygotuj prompt
prompt = "Billy"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

output_tokens = model.generate(
  **inputs,
  max_new_tokens=64,
  temperature=0.1,
  #top_p=0.95,
  repetition_penalty=1.2,
  do_sample=True,
).to('cuda')

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
