import asyncio
import json
import os
import sqlite3
import re
import threading

from astrbot.api import logger, star
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.command import GreedyStr

MASTER_ID = os.environ.get("ASTRBOT_MASTER_ID", "")
MASTER_NICK = os.environ.get("ASTRBOT_MASTER_NICK", "主人")
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "memories.db"
)

# 判断是否值得记忆的提示词
JUDGE_PROMPT = """你是记忆判断模块。请仔细判断下面这段对话中，用户所说/所发生的内容是否值得作为长期记忆保存。

【保存标准——从严，只记重点】：
只有以下内容才保存（宁缺勿滥）：
1. 用户明确要求记住的内容（无论多小，必须保存）；
2. 核心个人信息：姓名、年龄、生日、所在地、家人朋友等长期不变的人物关系；
3. 稳定的长期偏好：最喜欢的歌手/歌曲/食物/游戏等（一时兴起提的不算）；
4. 重要日期与约定：纪念日、生日、承诺过的事；
5. 重大经历：生病、搬家、工作/学业重大变动等。

【不保存】（命中任意一条就不保存）：
- 纯寒暄、语气词、玩笑；
- 临时性指令、临时提问；
- 关于机器人功能本身的测试/讨论；
- 顺口一提的、一次性的、情绪化的内容（除非用户明确要求记住）；
- 拿不准是否"重点"的——一律不保存。

【注意】：
- 宁可少记不可滥记：只有明确的重点才保存；
- 不要把静（助手）的回复内容当成记忆主体，记忆只关于用户本人；
- 概括时保留关键细节（具体的人名、日期、歌名、食物名等不要丢）。

对话内容：
{dialog}

只输出一个 JSON，格式：
{{"remember": true/false, "memory": "若值得记忆，用第三人称概括（主语用'主人'），60字以内，保留关键细节；否则留空"}}"""

# 判断需要回忆哪些记忆的提示词
RECALL_PROMPT = """你是记忆检索模块。以下是一批关于某位用户的长期记忆条目（编号+内容），以及当前对话消息。
请判断当前对话中需要提供哪些记忆条目来帮助理解和回复。

【需要召回】（命中任意一条就返回对应编号）：
1. 当前消息直接提到或隐含相关的记忆（提到歌→相关的音乐记忆；提到吃→食物偏好；提到游戏→游戏偏好；提到日期→生日/纪念日/约定）；
2. 当前消息与某条记忆主题相同、相关或可以补充上下文的；
3. 用户询问过去的事、"还记得吗"、试探记忆内容时，召回所有可能相关的条目（可多选）；
4. 涉及情感交流时，召回与人物关系、偏好、重要日期相关的条目。

【注意】：
- 宁可多召回不可漏召回：拿不准时倾向包含；
- 无法判断或确定都不相关时才返回空列表。

记忆条目：
{memories}

当前消息：{query}

只输出 JSON 数组，如 [1, 3]；无相关则输出 []"""

MEMORY_INJECT = (
    "\n\n[长期记忆 - 关于主人的事，由记忆系统自动检索]"
    "\n以下是主人过往的重要事情，自然地运用它们，不要生硬复述：\n"
)

_lock = threading.Lock()

# ---------- 本地语义向量检索（BAAI/bge-small-zh-v1.5, fastembed） ----------
_MODEL_DIR = os.environ.get("ASTRBOT_MEMORY_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
_MODEL = None
_RERANKER = None
_VEC_LOCK = threading.Lock()
# 记忆向量缓存表：(memory_id, vec BLOB)。向量随记忆增删增量维护。
def _vec_db():
    conn = sqlite3.connect(os.path.join(os.path.dirname(DB_PATH), "memory_vec.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS vecs (
        memory_id INTEGER PRIMARY KEY,
        dim INTEGER, vec BLOB NOT NULL)""")
    return conn

def _get_embedder():
    """懒加载本地 embedding 模型（首次约1.1s，常驻进程内复用）。失败返回 None 优雅降级。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _VEC_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from fastembed import TextEmbedding
            _MODEL = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir=_MODEL_DIR)
            logger.info("[long_memory] 语义向量模型已加载(本地bge-small-zh)")
        except Exception as e:
            logger.warning(f"[long_memory] 向量模型加载失败，降级纯关键词: {e}")
            _MODEL = False
    return _MODEL or None

def _get_reranker():
    """懒加载本地重排序模型（BAAI/bge-reranker-base, 1.1GB，加载约6s一次，常驻复用）。
    失败返回 None 优雅降级为纯向量排序。"""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    with _VEC_LOCK:
        if _RERANKER is not None:
            return _RERANKER
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _RERANKER = TextCrossEncoder(
                "BAAI/bge-reranker-base", cache_dir=_MODEL_DIR, threads=2
            )
            logger.info("[long_memory] 重排序模型已加载(本地bge-reranker-base)")
        except Exception as e:
            logger.warning(f"[long_memory] 重排序模型加载失败，降级向量排序: {e}")
            _RERANKER = False
    return _RERANKER or None

def _rerank(query: str, scored):
    """对 [(score, id, content)] 做交叉编码器精排。候选少于3条不值得排；长记忆截断控制耗时。"""
    rm = _get_reranker()
    if rm is None or len(scored) < 3:
        return scored
    try:
        cand = scored[:8]
        # 交叉编码器耗时与文本长度成正比；记忆的【标题】+开头已足够判相关性，取前120字控速
        docs = [c[:120] for _, _, c in cand]
        raw = list(rm.rerank(query=query, documents=docs))
        pairs = sorted(zip(raw, cand), key=lambda x: -x[0])
        out = [(r, sid, c) for r, (_, sid, c) in pairs]
        logger.info(f"[long_memory] rerank 精排 {len(cand)} 条候选")
        return out
    except Exception as e:
        logger.warning(f"[long_memory] rerank 异常，回退向量排序: {e}")
        return scored

def _embed_one(text: str):
    m = _get_embedder()
    if m is None:
        return None
    import numpy as np
    return np.asarray(list(m.embed([text[:512]]))[0], dtype=np.float32)

def _sync_vecs(rows):
    """把记忆库全量对齐到向量表（新增的算向量，删除的清掉）。"""
    m = _get_embedder()
    if m is None:
        return
    conn = _vec_db()
    try:
        have = {r[0] for r in conn.execute("SELECT memory_id FROM vecs")}
        ids = {i for i, _ in rows}
        for gone in have - ids:
            conn.execute("DELETE FROM vecs WHERE memory_id=?", (gone,))
        todo = [(i, c) for i, c in rows if i not in have]
        if todo:
            import numpy as np
            vecs = list(m.embed([c[:512] for _, c in todo]))
            for (i, _), v in zip(todo, vecs):
                conn.execute("INSERT OR REPLACE INTO vecs VALUES (?,?,?)",
                             (i, len(v), np.asarray(v, dtype=np.float32).tobytes()))
        conn.commit()
    finally:
        conn.close()

def _semantic_search(query: str, rows, limit: int = 8, floor: float = 0.30):
    """本地语义检索：返回 [(score, id, content)]，余弦相似度 floor 以上才入选。"""
    m = _get_embedder()
    if m is None or not rows:
        return []
    import numpy as np
    _sync_vecs(rows)
    conn = _vec_db()
    try:
        blobs = dict(conn.execute("SELECT memory_id, vec FROM vecs"))
    finally:
        conn.close()
    if not blobs:
        return []
    qv = _embed_one(query)
    if qv is None:
        return []
    qn = np.linalg.norm(qv)
    if qn < 1e-9:
        return []
    scored = []
    for i, c in rows:
        b = blobs.get(i)
        if not b:
            continue
        v = np.frombuffer(b, dtype=np.float32)
        n = np.linalg.norm(v)
        if n < 1e-9:
            continue
        s = float(np.dot(qv, v) / (qn * n))
        if s >= floor:
            scored.append((s, i, c))
    scored.sort(key=lambda x: -x[0])
    return scored[:limit]


# 与人格正文完全重复、无需再召回的条目标题（精确匹配，避免误伤夫君个人信息）
RULE_TITLES = ("【待客与认主铁律】",)


def _extract_json(text: str):
    """从模型输出中稳健提取第一个完整 JSON（对象或数组）。

    先整体解析；失败则从每个 { 或 [ 起始位置尝试 raw_decode，
    跳过前后夹杂的说明文字，避免贪婪正则跨越多段内容。
    """
    decoder = json.JSONDecoder()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
            return obj
        except ValueError:
            continue
    return None

# 常用停用词集合（单字拆开）
STOPWORDS = set("的主了是在我有和就不都一人你什么怎么这个那个可以还有就是不是吗吧呢啊哦嗯呀")


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        )"""
    )
    return conn


@register(
    "astrbot_plugin_long_memory",
    "静",
    "自主判断并保存/召回关于主人的跨会话长期记忆",
    "1.3.0",
)
class LongMemory(Star):
    def __init__(self, context: Context):
        super().__init__(context)


    # ---------- 存储 ----------
    def _add_memory(self, content: str):
        with _lock:
            conn = _db()
            conn.execute(
                "INSERT INTO memories (user_id, content) VALUES (?, ?)",
                (MASTER_ID, content),
            )
            conn.commit()
            conn.close()

    def _get_memories(self) -> list[tuple[int, str]]:
        """读取可用于召回的记忆。以【…】开头的人格/运维规则已写入人格正文，
        此处跳过，避免重复占用召回名额与上下文（数据仍保留在库中）。"""
        with _lock:
            conn = _db()
            rows = conn.execute(
                "SELECT id, content FROM memories WHERE user_id = ? "
                "ORDER BY id ASC",
                (MASTER_ID,),
            ).fetchall()
            conn.close()
        return [(i, c) for i, c in rows if not c.startswith(RULE_TITLES)]

    def _keyword_filter(self, query: str, limit: int = 20) -> list[tuple[int, str]]:
        """全库关键词粗筛：取与 query 词汇重叠的记忆条目。"""
        rows = self._get_memories()
        if not rows or not query:
            return []

        words = set()
        for w in re.findall(r"[\u4e00-\u9fa5]{2,}|[a-zA-Z0-9]+", query):
            if all(ch in STOPWORDS for ch in w):
                continue
            if len(w) > 2:
                for j in range(len(w) - 1):
                    sub = w[j:j + 2]
                    if not all(ch in STOPWORDS for ch in sub):
                        words.add(sub)
            else:
                words.add(w)

        # 同义词扩展
        syn = {
            "名字": ["姓名"], "姓名": ["名字"], "生日": ["出生"],
            "歌": ["歌曲", "音乐"], "纪念日": ["结婚"], "女朋友": ["恋人"],
            "游戏": ["玩"], "喜欢": ["偏好", "爱听", "爱吃"]
        }
        add_words = []
        for base, aliases in syn.items():
            if base in words:
                for a in aliases:
                    add_words.append(a)
        words.update(add_words)

        if not words:
            return []

        scored = []
        for i, c in rows:
            score = sum(1 for w in words if w in c)
            if score <= 0:
                continue
            if "明确要求记住" in c:
                score += 2
            if "各种回忆" in c:
                score -= 3
            scored.append((score, i, c))

        scored.sort(key=lambda x: -x[0])
        return [(i, c) for _, i, c in scored[:limit]]

    # ---------- 获取低成本辅助 Provider（优先选便宜小模型，为主人省 Token） ----------
    async def _get_helper_provider(self, umo: str):
        """优先从已配置列表中选取轻量模型（如 qwen/glm/flash），避免每轮都烧昂贵主模型"""
        try:
            cfg = self.context.get_config(umo=umo)
            # 获取所有已配置的 provider 列表
            all_providers = cfg.get("provider", [])
            # 候选便宜模型特征关键词
            cheap_patterns = ["flash", "glm", "qwen", "mini", "3.8-flash"]
            for target in cheap_patterns:
                for p in all_providers:
                    if not p.get("enable", True):
                        continue
                    pid = p.get("id", "")
                    mc = p.get("model_config") or {}
                    pmodel = p.get("model") or (mc.get("model") if isinstance(mc, dict) else "") or ""
                    if target in pid.lower() or target in pmodel.lower():
                        inst = self.context.get_provider_by_id(pid)
                        if inst:
                            return inst
        except Exception:
            pass
        # 兜底：获取当前使用的 provider
        return await self.context.get_using_provider_async(umo)

    # ---------- LLM 辅助 ----------
    async def _llm_json(self, prompt: str, umo: str):
        """用轻量辅助 Provider 做判断，返回解析后的 JSON 或 None"""
        provider = await self._get_helper_provider(umo)
        if provider is None:
            return None
        try:
            result = await provider.text_chat(
                prompt=prompt, session_id=None, contexts=[]
            )
            text = (result.completion_text or "").strip()
            return _extract_json(text)
        except Exception as e:
            logger.debug(f"[long_memory] LLM 判断失败: {e}")
            return None

    # ---------- 记忆写入（LLM 请求后，取本轮对话判断） ----------
    @filter.on_llm_response()
    async def on_response(self, event: AstrMessageEvent, resp: LLMResponse):
        try:
            if event.get_sender_id() != MASTER_ID:
                return
            user_text = (event.message_str or "").strip()
            for w in ("静", "静静", "阿静", "小静"):
                if user_text.startswith(w):
                    user_text = user_text[len(w):].strip()
                    break
            if not user_text or len(user_text) < 4:
                return

            # 1. 用户明确要求记住 → 直接保存，完全不需要调用 LLM
            force_kw = ("帮我记住", "请记住", "记住", "记一下", "帮我记下",
                        "记下", "帮我记", "请记")
            if any(k in user_text for k in force_kw):
                raw = user_text
                for kw in force_kw:
                    if kw in raw:
                        raw = raw.split(kw, 1)[1].strip(" ：:，,。")
                        break
                if raw and len(raw) >= 2:
                    memory = f"主人（明确要求记住）：{raw[:100]}"
                    existing = [c for _, c in self._get_memories()]
                    if memory not in existing:
                        self._add_memory(memory)
                        logger.info(f"[long_memory] 已保存(强制): {memory}")
                    return

            # 2. 如果只是一般闲聊或短问答，不盲目调用 LLM 烧 token，粗判包含关键属性才调用
            trigger_patterns = ["我叫", "我是", "我的生日", "我喜欢", "我讨厌", "我买了", "我决定", "我明天", "我今天去了"]
            if not any(tp in user_text for tp in trigger_patterns):
                return

            reply = (resp.completion_text or "").strip()
            dialog = f"用户：{user_text}\n静：{reply[:200]}"
            prompt = JUDGE_PROMPT.format(dialog=dialog)
            umo = event.unified_msg_origin

            data = await self._llm_json(prompt, umo)
            if not isinstance(data, dict) or not data.get("remember"):
                return
            memory = str(data.get("memory") or "").strip()
            if not memory:
                return
            existing = [c for _, c in self._get_memories()]
            if memory in existing:
                return
            self._add_memory(memory)
            logger.info(f"[long_memory] 已保存记忆: {memory}")
        except Exception as e:
            logger.error(f"[long_memory] on_response 异常: {e}", exc_info=True)

    # ---------- 记忆召回（注入 system prompt） ----------
    @filter.on_llm_request()
    async def on_request(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            if event.get_sender_id() != MASTER_ID:
                return
            rows = self._get_memories()
            if not rows:
                return
            query = (event.message_str or "").strip() or (req.prompt or "")
            for w in ("静", "静静", "阿静", "小静"):
                if query.startswith(w):
                    query = query[len(w):].strip()
                    break
            if not query:
                return

            # 核心记忆（以【…】开头的档案、偏好、铁律、规范）每轮必定注入，
            # 不依赖关键词命中，避免“说话没碰到关键词就失忆”。
            pinned = [(i, c) for i, c in rows if c.startswith("【")]
            pinned_ids = {i for i, _ in pinned}
            # 其余零散记忆：本地语义向量检索为主（语义相关即命中），
            # 关键词粗筛兜底（防向量模型异常时仍可用），合并去重后最多注入 5 条
            kw_ids = {i for i, _ in self._keyword_filter(query)}
            # 语义向量粗筛放宽到 0.30 捞更多候选，再由 reranker 精排取最优
            sem = _semantic_search(query, rows, limit=10, floor=0.30)
            if sem:
                sem = _rerank(query, sem)[:5]
            sem_ids = {i for _, i, _ in sem}
            extra_ids = list(sem_ids | (kw_ids - sem_ids))[:5]
            recall_hints = ["还记得", "是谁", "生日", "喜欢", "哪首", "我说过", "之前"]
            is_probing = any(hint in query for hint in recall_hints)

            picked = [f"- {c}" for _, c in pinned]
            id2c_map = {i: c for i, c in rows}
            picked += [f"- {id2c_map[i]}" for i in extra_ids if i in id2c_map]
            if picked:
                req.system_prompt += MEMORY_INJECT + "\n".join(picked)
                logger.info(
                    f"[long_memory] 注入核心记忆 {len(pinned)} 条 + 语义/关键词记忆 {len(extra_ids)} 条"
                )
            if extra_ids or not is_probing:
                return

            # 仅在主人询问过往时，才动用辅助小模型做批量精筛
            recent = [(i, c) for i, c in rows if i not in pinned_ids][-30:]
            if not recent:
                return
            listing = "\n".join(f"{i}. {c}" for i, c in recent)
            prompt = RECALL_PROMPT.format(memories=listing, query=query[:300])
            umo = event.unified_msg_origin
            data = await self._llm_json(prompt, umo)
            if not isinstance(data, list) or not data:
                return
            
            id2content = {i: c for i, c in rows}
            selected = [
                f"- {id2content[idx]}"
                for idx in data
                if isinstance(idx, int) and idx in id2content
            ]
            if selected:
                head = "\n" if picked else MEMORY_INJECT
                req.system_prompt += head + "\n".join(selected[:5])
                logger.info(f"[long_memory] 辅助 LLM 筛选注入 {len(selected)} 条记忆")
        except Exception as e:
            logger.error(f"[long_memory] on_request 异常: {e}", exc_info=True)

    # ---------- 管理命令 ----------
    @filter.command("memory")
    async def memory_cmd(self, event: AstrMessageEvent, action: str = "", text: GreedyStr = ""):
        """记忆管理：/memory list | /memory add 内容 | /memory del 编号 | /memory clear"""
        if event.get_sender_id() != MASTER_ID:
            event.set_result(MessageEventResult().message("不接受主人以外的命令。"))
            event.stop_event()
            return
        text = str(text or "").strip()
        rows = self._get_memories()

        def reply(msg: str):
            event.set_result(MessageEventResult().message(msg))

        if action == "list":
            if not rows:
                reply("记忆还是一片空白……")
            else:
                lines = [f"{i}. {c}" for i, (_, c) in enumerate(rows, 1)]
                reply(f"共 {len(rows)} 条记忆：\n" + "\n".join(lines))
        elif action == "add" and text:
            self._add_memory(text)
            reply("已记下了。")
        elif action == "del":
            try:
                idx = int(text)
            except ValueError:
                reply("用法：/memory del 编号")
                return
            if not (1 <= idx <= len(rows)):
                reply("没有这条编号的记忆。")
                return
            target = rows[idx - 1][0]
            with _lock:
                conn = _db()
                conn.execute("DELETE FROM memories WHERE id = ?", (target,))
                conn.commit()
                conn.close()
            reply("已删除。")
        elif action == "clear":
            # 需要二次确认，避免误触清空全部记忆
            if text != "confirm":
                reply("此操作会清空全部记忆且无法恢复。确认请发：/memory clear confirm")
                return
            with _lock:
                conn = _db()
                conn.execute("DELETE FROM memories WHERE user_id = ?", (MASTER_ID,))
                conn.commit()
                conn.close()
            reply("记忆已清空。")
        else:
            reply(
                "用法：/memory list | /memory add 内容 | /memory del 编号 | /memory clear confirm"
            )
