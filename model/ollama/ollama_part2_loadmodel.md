# Ollama 加载模型教程

1. ollama 加载 qwen2.5 模型，默认是 7B 版本

   ```
   Win + R ，输入 cmd 进入命令行

   # 启动 ollama
   ollama serve

   # 加载模型
   ollama pull qwen2.5
   ```
2. ollama 文件夹目录下的 ModelFile 文本，记录 model prompt

   ```
   cd model\ollama

   # 创建一个名为 Lunyu 的模型
   ollama create Lunyu -f ./Modelfile

   # 启动模型
   ollama run Lunyu
   ```
