# RAG Robot

基于检索增强生成(RAG)的文档问答机器人，主要功能：

- 加载并处理PDF文档(如Nike 10-K财报)
- 初始化大语言模型(LLM)
- 实现检索增强生成(RAG)功能
- 管理API密钥等敏感信息

## 目录结构

- `document/`: PDF文档处理模块
  - `pdf_loader.py`: PDF文档加载器
  - `rag.py`: RAG实现
- `llm/`: 大语言模型模块
  - `init_llm.py`: LLM初始化
- `utils/`: 工具模块
  - `key_keeper.py`: API密钥管理