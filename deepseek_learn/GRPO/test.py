from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('/home/bmh/project/llm_related/deepseek_learn/GRPO/outputs/Qwen-0.5B-GRPO/checkpoint-8600')
model = AutoModelForCausalLM.from_pretrained('/home/bmh/project/llm_related/deepseek_learn/GRPO/outputs/Qwen-0.5B-GRPO/checkpoint-8600')
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
messages = [
    {'role':'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': '树上一只猴子，树下一只猴子，一共有几只猴子？'}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
model_inputs = tokenizer(text, return_tensors='pt')
print(model_inputs)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)