"""
src/llm_client.py - LLM客户端（Dashscope OpenAI兼容接口）

面试要点：
"我没有用LangChain的LLM封装，而是直接用OpenAI SDK调Dashscope的兼容接口。
 关键设计决策：
 1. 用 extra_body 传递厂商特定参数（如 enable_thinking）
    ——这是OpenAI SDK处理非标准参数的官方方式
 2. 模型档位路由（STRONG/FAST）实现成本优化
 3. 所有调用统一经过事件总线，可观测性内建

 踩坑记录（重要！）：
 Dashscope qwen3系列模型要求非流式调用必须显式设置
 enable_thinking=false，否则返回400错误。
 解决方案：通过 extra_body={'enable_thinking': False} 注入，
 OpenAI SDK会把extra_body的内容合并到请求JSON中。"
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

from src.event_bus import Event, EventType, get_event_bus
from src.models import ModelError, ModelTier, TokenUsage

logger = logging.getLogger(__name__)


# ==============================================================================
# 模型定价参考（每1K tokens，美元估算）
# 实际以百炼控制台账单为准
# ==============================================================================
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Qwen Max系列
    "qwen-max":          {"input": 0.04,   "output": 0.12},
    "qwen-max-latest":   {"input": 0.04,   "output": 0.12},
    # Qwen Plus系列
    "qwen-plus":         {"input": 0.004,  "output": 0.012},
    "qwen-plus-latest":  {"input": 0.004,  "output": 0.012},
    # Qwen Turbo系列
    "qwen-turbo":        {"input": 0.001,  "output": 0.003},
    "qwen-turbo-latest": {"input": 0.001,  "output": 0.003},
    # Qwen3系列（新增）
    "qwen3-235b-a22b":   {"input": 0.02,   "output": 0.06},
    "qwen3-32b":         {"input": 0.006,  "output": 0.018},
    "qwen3-14b":         {"input": 0.003,  "output": 0.009},
    "qwen3-8b":          {"input": 0.0015, "output": 0.0045},
}

# 已知需要 enable_thinking 参数的模型前缀
# qwen3系列在非流式调用时必须显式设置 enable_thinking=false
_THINKING_MODELS_PREFIXES = ("qwen3",)


# ==============================================================================
# 配置类
# ==============================================================================

@dataclass
class LLMConfig:
    """
    LLM客户端配置

    设计决策：
    用dataclass而不是直接读os.environ，
    原因是测试时可以直接传入配置对象，不依赖环境变量。
    生产使用时用默认的lambda从环境变量读取。
    """
    api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )
    strong_model: str = field(
        default_factory=lambda: os.getenv("AGENT_MODEL_STRONG", "qwen-max")
    )
    fast_model: str = field(
        default_factory=lambda: os.getenv("AGENT_MODEL_FAST", "qwen-plus")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("AGENT_EMBEDDING_MODEL", "text-embedding-v3")
    )
    max_retries: int = 3
    timeout: float = 120.0
    temperature: float = 0.7

    # 是否启用思考链（enable_thinking）
    # False = 关闭（默认，兼容性最好，适合工具调用场景）
    # True  = 开启（qwen3系列支持，输出包含<think>标签）
    enable_thinking: bool = field(
        default_factory=lambda: os.getenv("AGENT_ENABLE_THINKING", "false").lower() == "true"
    )

    def _needs_thinking_param(self, model_name: str) -> bool:
        """
        判断指定模型是否需要显式传递 enable_thinking 参数

        面试要点：
        "这是防御性编程：我不假设所有模型行为一致。
         通过模型名称前缀判断是否需要注入特定参数，
         这样未来新增模型时只需维护 _THINKING_MODELS_PREFIXES 列表，
         不需要改核心调用逻辑。"
        """
        model_lower = model_name.lower()
        return any(model_lower.startswith(prefix) for prefix in _THINKING_MODELS_PREFIXES)


# ==============================================================================
# LLM客户端核心
# ==============================================================================

class LLMClient:
    """
    LLM客户端 - 封装Dashscope OpenAI兼容接口

    核心功能：
    1. chat()            - 基础对话（支持tool calling）
    2. stream_chat()     - 流式输出
    3. embed()           - 文本向量化（用于ChromaDB）

    兼容性处理：
    - qwen3系列：非流式调用注入 enable_thinking=false（via extra_body）
    - 旧模型：不传 enable_thinking，保持向后兼容

    错误分类：
    - AuthenticationError → API Key问题
    - BadRequestError     → 参数问题（包括enable_thinking相关的400）
    - RateLimitError      → 配额问题
    - APITimeoutError     → 超时问题
    - APIConnectionError  → 网络问题
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self._validate_config()

        # 初始化OpenAI客户端（指向Dashscope兼容端点）
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,  # 禁用SDK内置重试，我们自己控制重试逻辑
        )

        self._bus = get_event_bus()

        logger.info(
            f"LLMClient初始化完成\n"
            f"  强模型: {self.config.strong_model}\n"
            f"  快模型: {self.config.fast_model}\n"
            f"  Embedding: {self.config.embedding_model}\n"
            f"  端点: {self.config.base_url}\n"
            f"  enable_thinking: {self.config.enable_thinking}"
        )

    def _validate_config(self) -> None:
        """验证配置，提供清晰的错误信息"""
        if not self.config.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置！\n"
                "修复方法：在 .env 文件中添加：\n"
                "  DASHSCOPE_API_KEY=sk-your-key-here\n"
                "获取地址：https://bailian.console.aliyun.com/"
            )
        if not self.config.api_key.startswith("sk-"):
            logger.warning(
                f"API Key 格式异常（期望以 'sk-' 开头，当前前缀: '{self.config.api_key[:5]}...'）"
                "如果确认无误可忽略此警告"
            )

    def _resolve_model(self, tier: ModelTier) -> str:
        """根据档位解析实际模型名称"""
        return self.config.strong_model if tier == ModelTier.STRONG else self.config.fast_model

    def _build_extra_body(self, model_name: str) -> Optional[dict[str, Any]]:
        """
        构建厂商特定的额外请求参数

        面试要点：
        "OpenAI SDK的 extra_body 参数会把dict内容
         直接合并到请求的JSON body里，
         这是处理非OpenAI标准参数的官方推荐方式。
         比修改SDK源码或自己实现HTTP请求要干净得多。"

        Returns:
            dict: 需要注入到请求body的额外参数
            None: 不需要注入额外参数
        """
        if self.config._needs_thinking_param(model_name):
            return {"enable_thinking": self.config.enable_thinking}
        return None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tier: ModelTier = ModelTier.STRONG,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,  # 直接指定模型名（跳过tier路由）
    ) -> ChatCompletion:
        """
        基础对话接口（非流式）

        Args:
            messages:       OpenAI格式消息列表
            tier:           模型档位（STRONG/FAST）
            tools:          工具定义列表（JSON Schema格式）
            temperature:    温度，None则用配置默认值
            max_tokens:     最大输出Token数
            model_override: 直接指定模型名，跳过tier路由

        Returns:
            ChatCompletion: OpenAI标准响应对象

        Raises:
            ModelError: 包含分类错误信息和修复建议
        """
        model = model_override or self._resolve_model(tier)
        start_time = time.time()

        # 构建请求参数
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # 注入厂商特定参数（如 qwen3 的 enable_thinking）
        extra_body = self._build_extra_body(model)
        if extra_body:
            kwargs["extra_body"] = extra_body
            logger.debug(f"注入 extra_body: {extra_body}（模型: {model}）")

        # 发布请求事件
        self._bus.publish(Event(
            event_type=EventType.LLM_REQUEST,
            data={
                "model": model,
                "tier": tier.value,
                "message_count": len(messages),
                "has_tools": tools is not None,
                "extra_body": extra_body,
            },
            source="llm_client"
        ))

        # 执行调用（带分类错误处理）
        try:
            response = self._client.chat.completions.create(**kwargs)

        except AuthenticationError as e:
            raise ModelError(
                f"API Key 认证失败",
                model=model,
                suggestion=(
                    "请检查：\n"
                    "  1. DASHSCOPE_API_KEY 是否正确设置\n"
                    "  2. API Key 是否已过期或被禁用\n"
                    f"  原始错误: {e}"
                )
            )

        except BadRequestError as e:
            error_msg = str(e)
            # 精准识别 enable_thinking 相关错误
            if "enable_thinking" in error_msg:
                raise ModelError(
                    f"模型 '{model}' 要求设置 enable_thinking 参数",
                    model=model,
                    suggestion=(
                        f"模型 '{model}' 属于需要显式设置思考模式的新模型。\n"
                        "修复方法：在 .env 中添加：\n"
                        "  AGENT_MODEL_STRONG=qwen-max  （使用兼容性更好的模型）\n"
                        "或：此问题应已由 llm_client.py 自动处理，请检查模型名前缀配置。\n"
                        f"原始错误: {e}"
                    )
                )
            raise ModelError(
                f"请求参数错误: {e}",
                model=model,
                suggestion=f"请检查请求参数是否符合 {model} 的要求。原始错误: {e}"
            )

        except RateLimitError as e:
            raise ModelError(
                f"API 调用频率超限或配额不足",
                model=model,
                suggestion=(
                    "请检查：\n"
                    "  1. 账号余额是否充足（https://bailian.console.aliyun.com/）\n"
                    "  2. 是否触发了 RPM/TPM 限制（稍后重试）\n"
                    f"  原始错误: {e}"
                )
            )

        except APITimeoutError as e:
            raise ModelError(
                f"API 请求超时（>{self.config.timeout}秒）",
                model=model,
                suggestion=(
                    "请检查：\n"
                    "  1. 网络连接是否正常\n"
                    "  2. 可尝试增大超时：在 LLMConfig 中调整 timeout 参数\n"
                    f"  原始错误: {e}"
                )
            )

        except APIConnectionError as e:
            raise ModelError(
                f"无法连接到 API 端点: {self.config.base_url}",
                model=model,
                suggestion=(
                    "请检查：\n"
                    "  1. 网络连接是否正常（curl 测试）\n"
                    "  2. DASHSCOPE_BASE_URL 是否正确\n"
                    "  3. 国内用户应使用：https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                    f"  原始错误: {e}"
                )
            )

        except Exception as e:
            raise ModelError(
                f"LLM调用发生未预期错误: {type(e).__name__}: {e}",
                model=model,
                suggestion="请查看完整的错误堆栈进行诊断"
            )

        # 统计与事件
        elapsed_ms = (time.time() - start_time) * 1000
        usage = self._extract_usage(response, model)

        self._bus.publish(Event(
            event_type=EventType.LLM_RESPONSE,
            data={
                "model": model,
                "tokens": usage.total_tokens,
                "elapsed_ms": elapsed_ms,
                "estimated_cost_usd": usage.estimated_cost_usd,
                "has_tool_calls": bool(
                    response.choices[0].message.tool_calls if response.choices else False
                ),
            },
            source="llm_client"
        ))

        logger.debug(
            f"LLM响应 ✓ | 模型: {model} | "
            f"Tokens: {usage.total_tokens} | "
            f"耗时: {elapsed_ms:.0f}ms | "
            f"估算费用: ${usage.estimated_cost_usd:.5f}"
        )

        return response

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tier: ModelTier = ModelTier.STRONG,
        model_override: Optional[str] = None,
    ) -> Iterator[str]:
        """
        流式对话接口 - 逐字输出

        注意：流式调用不需要 enable_thinking 参数（qwen3默认兼容）
        但如果 config.enable_thinking=True，流式输出会包含 <think> 标签

        Yields:
            str: 每个文本片段（已过滤thinking标签内的内容）
        """
        model = model_override or self._resolve_model(tier)

        # 流式调用的 extra_body 处理
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        extra_body = self._build_extra_body(model)
        if extra_body:
            kwargs["extra_body"] = extra_body

        stream = self._client.chat.completions.create(**kwargs)

        in_thinking = False  # 用于过滤 <think>...</think> 内容
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta.content:
                continue

            content = delta.content

            # 过滤思考链内容（enable_thinking=True 时会出现）
            if "<think>" in content:
                in_thinking = True
            if "</think>" in content:
                in_thinking = False
                continue
            if in_thinking:
                continue

            self._bus.publish(Event(
                event_type=EventType.LLM_STREAM_CHUNK,
                data={"content": content, "model": model},
                source="llm_client"
            ))
            yield content

    def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """
        文本向量化接口

        Args:
            texts: 要向量化的文本列表
            model: 向量模型名，默认使用配置中的 embedding_model

        Returns:
            list[list[float]]: 每个文本对应的向量

        注意：
        Embedding API 不受 enable_thinking 影响，无需注入 extra_body
        """
        embed_model = model or self.config.embedding_model

        try:
            response = self._client.embeddings.create(
                model=embed_model,
                input=texts,
            )
        except AuthenticationError as e:
            raise ModelError(
                "Embedding API 认证失败",
                model=embed_model,
                suggestion=f"请检查 DASHSCOPE_API_KEY 是否有效。原始错误: {e}"
            )
        except Exception as e:
            raise ModelError(
                f"Embedding 调用失败: {e}",
                model=embed_model,
                suggestion="请检查 AGENT_EMBEDDING_MODEL 配置和网络连接"
            )

        return [item.embedding for item in response.data]

    def _extract_usage(self, response: ChatCompletion, model: str) -> TokenUsage:
        """从响应中提取Token使用量并估算费用"""
        usage = TokenUsage(model=model)
        if response.usage:
            usage.prompt_tokens = response.usage.prompt_tokens
            usage.completion_tokens = response.usage.completion_tokens
            usage.total_tokens = response.usage.total_tokens

        pricing = MODEL_PRICING.get(model, {"input": 0.01, "output": 0.03})
        usage.estimated_cost_usd = round(
            usage.prompt_tokens / 1000 * pricing["input"] +
            usage.completion_tokens / 1000 * pricing["output"],
            6
        )
        return usage

    def get_model_info(self) -> dict[str, str]:
        """返回当前模型配置（调试/诊断用）"""
        return {
            "strong_model":    self.config.strong_model,
            "fast_model":      self.config.fast_model,
            "embedding_model": self.config.embedding_model,
            "base_url":        self.config.base_url,
            "enable_thinking": str(self.config.enable_thinking),
            "timeout":         str(self.config.timeout),
        }

    def probe(self) -> dict[str, Any]:
        """
        探针：快速验证客户端可用性（不消耗太多Token）

        面试要点：
        "probe() 是一个轻量级健康检查方法。
         在系统启动时调用，确保API可用再进入主循环，
         比在第一次真实调用时才发现问题要好得多。
         类似 k8s 的 readinessProbe。"

        Returns:
            dict: 包含 success、model、latency_ms、error 等字段
        """
        start = time.time()
        try:
            response = self.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tier=ModelTier.FAST,
                max_tokens=5,
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            return {
                "success": True,
                "model": self.config.fast_model,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "response_preview": content[:30],
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            }
        except ModelError as e:
            return {
                "success": False,
                "model": self.config.fast_model,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "error": str(e),
                "suggestion": e.suggestion,
            }
        except Exception as e:
            return {
                "success": False,
                "model": self.config.fast_model,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "error": f"{type(e).__name__}: {e}",
                "suggestion": "请查看完整错误堆栈",
            }