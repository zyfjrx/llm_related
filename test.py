from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://14.103.162.45:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="/root/sft/Qwen3-14B-sft-epoch3-r8",
    messages=[
        {"role": "user", "content": "狗发烧了怎么办？"},
    ],
    max_tokens=8192,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
)
print("Chat response:", chat_response)