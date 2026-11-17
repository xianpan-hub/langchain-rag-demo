# Local Ollama RAG Demo
> 极简检索增强原型，仅用于学习调试，效果有限，Query Rewrite增益不大，属于最小可用版本。

## 功能说明
- 底层：Ollama，使用 OpenAI 兼容接口（`/v1`）调用
- 能力：Query Rewrite（查询改写） + 向量检索 + LLM回答
- 对外：FastAPI 封装接口
- 依赖管理：uv
- 说明：改写只是做一层问题扩展，样本量很小，增强效果一般，仅作为学习链路验证

## 目录结构
```text
├── ollama_service.py    # LLM 调用服务
├── rag_service.py       # 向量检索、RAG 逻辑
├── main.py              # 程序入口
├── data/                # 样例文档、知识库文件
├── samples/             # 零散测试脚本、demo
├── runs/                # 临时输出、缓存文件
├── .env                 # 环境变量配置（密钥，git忽略不上传）
├── requirements.in      # 手写顶层依赖（pip-tools源文件）
├── requirements.txt     # pip-tools自动生成、锁定版本的依赖清单
├── .gitignore           # git忽略配置
└── README.md            # 项目说明
```

## 前置依赖

1. Ollama 本地服务，默认地址 `http://127.0.0.1:11434`
2. 模型固定硬编码：`qwen2.5:7b`
3. 拉取模型命令：
`ollama pull qwen2.5:7b`

## Python 依赖库
>本项目用到的包（后续可以导出到 requirements.txt）
- `langchain-community`
- `langchain-ollama`
- `langchain-text-splitters`
- `langchain-core`
- `langchain-openai`
- `python-dotenv`
- `fastapi`
- `uvicorn`

 补充说明：
`samples/` 文件夹是历史学习草稿、零散试验脚本。里面代码用到很多额外、五花八门的包，**不属于主程序依赖，不保证可以直接运行**，仅作为自己的代码片段存档。
后续导出 `requirements.txt` 只包含上面核心包。

## 启动步骤

1. 准备虚拟环境，安装上面核心 Python 包
2. 确保 Ollama 已经启动，并且拉取 `qwen2.5:7b`
3. 启动 FastAPI 服务
`uvicorn main:app --reload`
4. 接口文档访问：
[http://127.0.0.1:8000/docs]
