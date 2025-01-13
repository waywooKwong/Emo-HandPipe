有多种安装部署方式 Windows / Linux / Docker

这里出于便捷，直接使用 Windows 的安装方式

1. 官网下载安装

Ollama 官网：https://ollama.com/

Windows 下载安装

![](https://nankai.feishu.cn/space/api/box/stream/download/asynccode/?code=YWFkMmViMWJhYjkzNzg3OGQyZjE5MDJlNTVlYjdmMDBfRHdCU2c1Slk4Sm1wMGUzUmxLeDhrbUZWRGhCaWE1UTJfVG9rZW46RlNHdWJxUkJDb3ZMNm94QmtlMWNDaEF1bk5xXzE3MjcwMTA2NzY6MTcyNzAxNDI3Nl9WNA)

2. 设置模型存储路径

![](https://nankai.feishu.cn/space/api/box/stream/download/asynccode/?code=YjdmODY1NDY5NTI0YmIyYTdhMTU2NmFlYjExZTdhMmFfWU1JZ2ZueUVHVmNkdWhLT0tyb2RTWVh3cHplcm9JOHFfVG9rZW46UGRUY2I4MDZjb1plOVJ4YVNBc2NYNVZJbkFmXzE3MjcwMTA2NzY6MTcyNzAxNDI3Nl9WNA)

系统变量里新建 OLLAMA_MODELS，

赋值为你设定的模型存储路径，这里我放在 Y:\LLM\Ollama

3. 查看并拉取 Ollama 中支持的模型到本地

![](https://nankai.feishu.cn/space/api/box/stream/download/asynccode/?code=NzgxZGJhMmU3NTA4MGFmOTkyMDViOTU0ZjgxMGJiYWVfb0tqRlQ2WXpuS2ZrenhBb25iT3pQQzRNVmx1RjJ0V2NfVG9rZW46UFRqR2J2NlRob3RTNE94UUM1TWNicll5bjdlXzE3MjcwMTA2NzY6MTcyNzAxNDI3Nl9WNA)

![](https://nankai.feishu.cn/space/api/box/stream/download/asynccode/?code=MWI1NGM4OWUyZTlmMjczYTM1YTEyYWZmNTI5MDk1NDhfY2xab3dzeG1IbWtnbG5KMkRQNDZlODI3QmZhcFpNSGlfVG9rZW46VVMxcGJjdERKb2ZBZUJ4TWc4WmNWZ0RvbmhnXzE3MjcwMTA2NzY6MTcyNzAxNDI3Nl9WNA)

4. 选定模型，以 **qwen2** 为例

```Bash
命令行启动 cmd
ollama serve //启动 ollama
ollama pull qwen2 // 拉取 qwen2, 等待下载完成
ollama run qwen2 // 测试对话效果
```

```Python
// pip install langchain_community

// python 测试 Ollama 代码
from langchain_community.chat_models import ChatOllama
chat_model = ChatOllama(model = "qwen2")
response = chat_model.invoke("introduce yourself")
print("response:",response)
```
