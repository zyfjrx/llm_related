import os
from flask import Flask, render_template, request, jsonify, Response
from openai import OpenAI
import json
from gevent.pywsgi import WSGIServer
from gevent import monkey

monkey.patch_all()

app = Flask(__name__)

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-xxx"),  # 替换为你的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat')
def chat():
    # 从查询参数获取数据
    prompt = request.args.get('message', '')
    enable_thinking = request.args.get('enable_thinking', 'true').lower() == 'true'

    def generate():
        try:
            # 准备对话消息
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ]

            # 调用百炼API
            completion = client.chat.completions.create(
                model="qwen-plus",  # 可根据需要更换模型
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
            )

            # 处理流式响应
            for chunk in completion:
                chunk_data = json.loads(chunk.model_dump_json())

                # 处理内容响应
                if 'choices' in chunk_data:
                    for choice in chunk_data['choices']:
                        if 'delta' in choice and 'content' in choice['delta']:
                            content = choice['delta']['content']
                            if content:
                                # 修复f-string多行问题：先构建字典再转换为JSON
                                response_data = {
                                    'type': 'content',
                                    'content': content,
                                    'done': False
                                }
                                yield f"data: {json.dumps(response_data)}\n\n"

                # 处理使用情况统计
                if 'usage' in chunk_data:
                    usage_data = {
                        'type': 'usage',
                        'usage': chunk_data['usage'],
                        'done': False
                    }
                    yield f"data: {json.dumps(usage_data)}\n\n"

            # 发送完成信号
            complete_data = {
                'type': 'complete',
                'done': True
            }
            yield f"data: {json.dumps(complete_data)}\n\n"

        except Exception as e:
            print(f"生成过程中发生错误: {e}")
            error_data = {
                'type': 'error',
                'message': str(e),
                'done': True
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    # 使用gevent服务器提高SSE稳定性
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print("服务器启动: http://0.0.0.0:5000")
    http_server.serve_forever()