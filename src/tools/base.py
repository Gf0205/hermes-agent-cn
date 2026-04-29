"""
src/tools/base.py - 工具抽象基类

面试要点：
"我设计了BaseTool抽象类，强制每个工具实现：
 1. name/description - 让LLM知道何时用这个工具
 2. parameters - JSON Schema，让LLM知道如何调用
 3. execute() - 实际执行逻辑
 4. validate_args() - 参数校验，在执行前捕获错误

 这个设计让工具的添加变得非常简单：
 继承BaseTool，实现3个方法，注册到registry就完成了。"

设计模式：Template Method - 基类定义流程，子类实现细节
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from src.models import ToolParameter, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """
    工具抽象基类

    所有工具必须继承此类并实现抽象方法。
    基类负责：
    - 统一的参数校验流程
    - 执行时间统计
    - 异常捕获和标准化错误返回
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（唯一标识，用于LLM调用）"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        工具描述（给LLM看的说明）

        写作要点：
        1. 说清楚"什么时候用这个工具"
        2. 说清楚"会返回什么"
        3. 举一个简单例子
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """参数定义列表"""
        ...

    @abstractmethod
    def _execute(self, **kwargs: Any) -> str:
        """
        工具的实际执行逻辑（子类实现）

        Returns:
            str: 工具的输出（会被放入LLM的上下文中）

        注意：不要在这里处理异常，让基类统一处理
        """
        ...

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        执行工具（模板方法）

        流程：
        1. 参数校验
        2. 计时
        3. 调用_execute()
        4. 捕获异常，返回标准化ToolResult

        面试要点：
        "这是Template Method模式的经典应用。
         基类控制执行流程（校验→执行→统计），
         子类只需要实现核心逻辑。
         这保证了所有工具都有一致的错误处理和性能统计。"
        """
        start_time = time.time()

        # 步骤1：参数校验
        validation_error = self.validate_args(**kwargs)
        if validation_error:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILURE,
                output="",
                error=f"参数校验失败: {validation_error}",
                execution_time_ms=0.0,
            )

        # 步骤2：执行
        try:
            output = self._execute(**kwargs)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"工具 [{self.name}] 执行成功 | 耗时: {elapsed_ms:.0f}ms"
            )

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                output=output,
                execution_time_ms=elapsed_ms,
            )

        except PermissionError as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.PERMISSION_DENIED,
                output="",
                error=f"权限不足: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except TimeoutError as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.TIMEOUT,
                output="",
                error=f"执行超时: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(f"工具 [{self.name}] 执行失败: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILURE,
                output="",
                error=str(e),
                execution_time_ms=elapsed_ms,
            )

    def validate_args(self, **kwargs: Any) -> Optional[str]:
        """
        参数校验（可被子类覆盖以添加自定义校验）

        Returns:
            str: 错误信息（有错误时）
            None: 校验通过
        """
        # 检查必需参数是否存在
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return f"缺少必需参数: '{param.name}'"

            if param.name in kwargs and param.enum_values:
                if kwargs[param.name] not in param.enum_values:
                    return (
                        f"参数 '{param.name}' 的值 '{kwargs[param.name]}' 不在允许范围内。"
                        f"允许的值: {param.enum_values}"
                    )

        return None  # 校验通过

    def to_openai_schema(self) -> dict[str, Any]:
        """
        将工具定义转换为OpenAI Function Calling格式

        面试要点：
        "OpenAI的tool_choice协议要求工具以JSON Schema格式描述。
         这个方法把我们的ToolParameter对象转成LLM能理解的格式，
         让LLM知道：这个工具叫什么？有什么参数？参数类型是什么？"

        输出格式：
        {
            "type": "function",
            "function": {
                "name": "read_file.py",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum_values:
                prop["enum"] = param.enum_values
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def __repr__(self) -> str:
        param_names = [p.name for p in self.parameters]
        return f"Tool({self.name}, params={param_names})"