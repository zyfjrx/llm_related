from openai import OpenAI

client = OpenAI(
    api_key="none",
    base_url="http://localhost:8998/v1"
)



while True:
    conversation_history = []
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="minimind",
        messages=conversation_history,
        stream=False
    )

    assistant_res = response.choices[0].message.content
    print('[A]: ', assistant_res)

