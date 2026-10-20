# astrbot_plugin_long_memory

**跨会话长期记忆插件** —— 机器人自主判断什么值得记，用本地语义向量 + 重排序双层检索精准召回，群聊私聊共享记忆。

## 功能

- **自主记忆**：每轮对话后由辅助 LLM 判断是否值得保存（宁缺勿滥：只记姓名/生日/偏好/约定/重大经历，不记寒暄玩笑）；
- **三级召回管线**（v1.4.0）：
  1. **核心记忆**：以【】开头的档案条目每轮必注入，永不遗漏；
  2. **语义向量粗筛**：本地 BAAI/bge-small-zh-v1.5（fastembed + ONNX，纯 CPU 95MB），余弦相似度 ≥0.30 捞 Top10；
  3. **重排序精排**：本地 BAAI/bge-reranker-base 交叉编码器对候选精排（截断 120 字控速），取 Top5 注入；
- **关键词兜底**：向量模型异常时自动降级关键词召回，聊天永不受影响；
- **群私共享**：同一主人跨群、跨私聊记忆互通；
- **管理指令**：查看/删除/清空记忆。

## 安装

```bash
cd /path/to/AstrBot/data/plugins
git clone https://github.com/snlyj/astrbot_plugin_long_memory.git
# 重启 AstrBot 或在 WebUI 面板热重载
```

首次对话会自动下载语义模型（约 95MB，从 hf-mirror.com，国内可达）；重排序模型 1.1GB 需手动放置（见下）。

## 依赖

```bash
pip install fastembed numpy
```

## 配置（重要）

插件通过**环境变量**配置，在启动 AstrBot 的环境中设置：

| 环境变量 | 必填 | 默认值 | 说明 |
|----------|------|--------|------|
| `ASTRBOT_MASTER_ID` | ✅ | 空 | 主人的平台用户 ID（QQ 号），只有此人的对话会被记忆/召回 |
| `ASTRBOT_MASTER_NICK` | ❌ | `主人` | 主人昵称，用于记忆摘要中的称谓 |
| `ASTRBOT_MEMORY_MODEL_DIR` | ❌ | 插件目录下 `models/` | 语义模型缓存目录。默认会自动下载到这里 |

不习惯环境变量的话，直接编辑 `main.py` 顶部同名常量也可以。

### 重排序模型（可选，推荐）

v1.4.0 的精排能力需要 `BAAI/bge-reranker-base` ONNX 模型（1.1GB）。**不放置也能正常工作**（自动降级纯向量排序）。

手动安装方法：

```bash
# 设定模型目录（示例）
mkdir -p /path/to/models
# 用 huggingface_hub 下载（国内可设 HF_ENDPOINT=https://hf-mirror.com）
pip install -U huggingface_hub
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qdrant/bge-reranker-base', cache_dir='/path/to/models')
"
# 然后让 ASTRBOT_MEMORY_MODEL_DIR 指向该目录
```

> 下载 embedding 模型同理，插件首次运行会自动处理。

## 指令说明

| 指令 | 说明 |
|------|------|
| `/memory list` | 列出全部记忆 |
| `/memory del <id>` | 删除指定 id 的记忆 |
| `/memory clear` | 清空全部记忆（慎用） |
| `/memory add <内容>` | 手动添加记忆，用【】开头可标记为核心记忆（每轮必注入） |

## 记忆判定标准（内置，改提示词可调）

保存：明确要求记住的、核心个人信息、稳定长期偏好、重要日期约定、重大经历。
不保存：寒暄、临时指令、功能测试、玩笑情绪话。

相关提示词常量在 `main.py` 顶部：`JUDGE_PROMPT`（判定）、`RECALL_PROMPT`（辅助召回）、`MEMORY_INJECT`（注入格式），可按需定制。

## 使用示例

```
你：记住我喜欢吃牛蛙
（记忆系统保存：「主人喜欢吃牛蛙」）

# 三天后，新会话
你：晚上想吃点什么
🤖 要不要去吃牛蛙呀主人～（记忆自动召回）
```

## 数据文件

| 文件 | 说明 |
|------|------|
| `memories.db` | SQLite 记忆库（备份它=备份记忆） |
| `memory_vec.db` | 向量缓存（删了会自动重建） |

## 性能参考

- 冷启动（模型加载）：约 12 秒一次
- 稳态单轮检索：约 3.5 秒（含 rerank）/ 0.1 秒（纯向量降级模式）
- 内存占用：约 1.2GB（含两个模型常驻）

## License

MIT
