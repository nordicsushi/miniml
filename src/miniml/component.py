from __future__ import annotations

from functools import wraps
import logging
from typing import Callable, Any, ParamSpec

from rich.logging import RichHandler
from pydantic import BaseModel, Field

from .io import InputSpec, OutputSpec
from .step import Step
from .artifacts import ArtifactStore, InMemoryArtifactStore
from .exceptions import (
    StepNotFoundError,
    DuplicateStepError,
    InvalidSegmentError,
)

logging.basicConfig(
    level=logging.WARNING,
    format="[%(filename)s:%(lineno)s: %(funcName)15s() ] %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_level=True)],
)
logger = logging.getLogger("component")
logger.setLevel(logging.DEBUG)


class ComponentContract(BaseModel):
    """对外契约（类似 AzureML v2 的 Component inputs/outputs）"""

    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    outputs: dict[str, OutputSpec] = Field(default_factory=dict)
    # 组件输入注入到哪个内部 step（作为该 step 的调用参数 kwargs）
    # 约定当前版本：组件 input 名 == 对应 step 的参数名
    # 映射为：component_input_name -> target_step_name
    input_to_step: dict[str, str] = Field(default_factory=dict)
    # 组件输出来自哪个内部 step的结果，或该结果中的某个字段
    # 两种写法：
    #   "out_name": "step_name"                          -> 使用该 step 的完整返回值
    #   "out_name": ("step_name", "field_key")           -> 使用该 step 返回 dict 中的某个键
    output_from_step: dict[str, str | tuple[str, str | None]] = Field(
        default_factory=dict
    )


class Component:
    """
    A self-contained DAG (directed acyclic graph) of steps + IO contract.

    - Steps are registered via `@component.step(...)`（内部 DAG）
    - 对外通过 contract（inputs/outputs + map）定义接口
    - `run(start_step, end_step, inputs)`：支持子图执行并注入输入
    """

    def __init__(
        self,
        name: str | None = None,
        artifact_store: ArtifactStore | None = None,
        contract: ComponentContract | None = None,
    ) -> None:
        self.name = name or "component"
        self.steps: dict[str, Step] = {}
        self._children: dict[str, list[str]] = {}  # 直接下游
        self._artifact_store: ArtifactStore = artifact_store or InMemoryArtifactStore()
        self.contract = contract or ComponentContract()

    # --------------------- 定义/修改 IO 契约 ---------------------
    def set_inputs(
        self, specs: dict[str, InputSpec], input_to_step: dict[str, str]
    ) -> None:
        self.contract.inputs = specs
        self.contract.input_to_step = input_to_step

    def set_outputs(
        self,
        specs: dict[str, OutputSpec],
        output_from_step: dict[str, str | tuple[str, str | None]],
    ) -> None:
        self.contract.outputs = specs
        self.contract.output_from_step = output_from_step

    # --------------------- 注册 step ---------------------
    def step(
        self, name: str | None = None, depends_on: list[str] | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        P = ParamSpec("P")

        def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
            step_name = name or func.__name__

            if step_name in self.steps:
                raise DuplicateStepError(
                    f"Step {step_name!r} already exists in Component {self.name!r}"
                )

            if depends_on:
                if step_name in depends_on:
                    raise InvalidSegmentError(
                        f"Step {step_name!r} cannot depend on itself."
                    )
            self.steps[step_name] = Step(step_name, func, depends_on)
            self._children.clear()

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return func(*args, **kwargs)

            return wrapper

        return decorator

    # --------------------- 执行 ---------------------
    def run(
        self,
        start_step: str | None = None,
        end_step: str | None = None,
        inputs: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        返回值：dict[str, Any] —— 按契约返回 outputs（不是返回最后一个 step 的值）
        """
        if not self.steps:
            raise InvalidSegmentError("No steps defined in this Component.")

        if not self._children:
            self._build_children()

        self._validate_all_dep_exist()

        if start_step and start_step not in self.steps:
            raise StepNotFoundError(f"Unknown start_step: {start_step!r}")
        if end_step and end_step not in self.steps:
            raise StepNotFoundError(f"Unknown end_step: {end_step!r}")

        # 校验 inputs 完整性（基于契约）
        inputs = inputs or {}
        self._validate_component_inputs(inputs)

        # 1) 计算执行 segment
        segment = self._compute_segment(start_step, end_step)
        logger.info(f"[{self.name}] Computed segment: {segment}")

        # 2) 构建「按 step 聚合」的 kwargs 注入（不写进 cache，避免覆盖上游输出）
        # 约定：组件 input 名 == 对应 step 的参数名
        injections_by_step: dict[str, dict[str, Any]] = {}
        for in_name, in_value in inputs.items():
            target_step = self.contract.input_to_step.get(in_name)
            if not target_step:
                raise InvalidSegmentError(
                    f"Input {in_name!r} has no input_to_step mapping in Component {self.name!r}"
                )
            injections_by_step.setdefault(target_step, {})[in_name] = in_value

        # 3) 初始化本次运行的内存 cache（仅历史 artifact，不包含 inputs）
        cache: dict[str, Any] = {}
        if use_cache:
            for step_name in segment:
                cached = self._artifact_store.get(self.name, step_name)
                if cached is not None:
                    cache[step_name] = cached

        # 4) 递归执行
        def exec_step(name: str) -> Any:
            logger.debug(f"[{self.name}] Backpropagating step '{name}' ...")
            if name in cache:
                logger.debug(f"[{self.name}] Using cached value for '{name}'")
                return cache[name]

            if name not in self.steps:
                raise StepNotFoundError(
                    f"Step {name!r} is not defined in Component {self.name!r}."
                )

            step = self.steps[name]
            dep_results: list[Any] = []

            for dep in step.depends_on:
                # 依赖在 cache（可能来自历史缓存）
                if dep in cache:
                    dep_results.append(cache[dep])
                    continue
                # 依赖在 segment 内：重算
                if dep in segment:
                    dep_results.append(exec_step(dep))
                    continue
                # 依赖不在 segment：尝试历史缓存（允许从 segment 外复用）
                if use_cache:
                    prev = self._artifact_store.get(self.name, dep)
                    if prev is not None:
                        cache[dep] = prev
                        dep_results.append(prev)
                        continue
                # 走到这里，说明依赖既不重算、也没有可用结果
                raise InvalidSegmentError(
                    f"Missing dependency {dep!r} for step {name!r}: "
                    f"not scheduled and no cached value available."
                )

            # 真正执行当前 step：上游输出作为位置参数，组件顶层 inputs 注入为 kwargs
            logger.debug(f"[{self.name}] Forward executing step '{name}' ...")
            kwargs = injections_by_step.get(name, {})
            result = step.func(*dep_results, **kwargs)
            cache[name] = result
            if use_cache:
                self._artifact_store.set(self.name, name, result)
            return result

        # 5) 产出 outputs（按契约从指定内部 step 读取；可选取字段）
        outputs: dict[str, Any] = {}
        for out_name in self.contract.outputs.keys():
            mapping = self.contract.output_from_step.get(out_name)
            if not mapping:
                raise InvalidSegmentError(
                    f"Output {out_name!r} has no output_from_step mapping in Component {self.name!r}"
                )

            # 兼容两种写法： str 或 (str, field_key|None)
            if isinstance(mapping, str):
                src_step, field_key = mapping, None
            else:
                src_step, field_key = mapping  # pyright: ignore[reportGeneralTypeIssues]

            value = exec_step(src_step)
            if field_key is not None:
                try:
                    value = value[field_key]
                except Exception as e:
                    raise InvalidSegmentError(
                        f"Output {out_name!r} expects field {field_key!r} "
                        f"from step {src_step!r} result, but got error: {e}"
                    ) from e
            outputs[out_name] = value

        return outputs

    # --------------------- 内部工具 ---------------------
    def _validate_component_inputs(self, inputs: dict[str, Any]) -> None:
        for in_name, spec in self.contract.inputs.items():
            if not spec.optional and in_name not in inputs:
                if spec.default is None:
                    raise InvalidSegmentError(
                        f"Component {self.name!r} missing required input: {in_name!r}"
                    )

    def _build_children(self) -> None:
        if self._children:
            return
        children: dict[str, list[str]] = {name: [] for name in self.steps}
        for name, step in self.steps.items():
            for dep in step.depends_on:
                children.setdefault(dep, []).append(name)
        self._children = children

    def _dfs_forward(self, start: str) -> set[str]:
        visited: set[str] = set()
        stack: list[str] = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._children.get(node, []))
        return visited

    def _dfs_backward(self, end: str) -> set[str]:
        visited: set[str] = set()
        stack: list[str] = [end]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.steps[node].depends_on)
        return visited

    def _compute_segment(
        self,
        start_step: str | None,
        end_step: str | None,
    ) -> set[str]:
        all_nodes = set(self.steps.keys())
        forward = self._dfs_forward(start_step) if start_step else all_nodes
        backward = self._dfs_backward(end_step) if end_step else all_nodes
        segment = forward & backward
        if not segment:
            raise InvalidSegmentError(
                f"No executable segment for start_step={start_step!r}, end_step={end_step!r}"
            )
        if start_step and start_step not in segment:
            raise InvalidSegmentError(
                f"start_step={start_step!r} not in computed segment."
            )
        if end_step and end_step not in segment:
            raise InvalidSegmentError(
                f"end_step={end_step!r} not reachable from start_step={start_step!r}."
            )
        return segment

    def _validate_all_dep_exist(self) -> None:
        for name, step in self.steps.items():
            for dep in step.depends_on:
                if dep not in self.steps:
                    raise StepNotFoundError(
                        f"Step {name!r} depends on {dep!r}, which is not defined "
                        f"in Component {self.name!r}."
                    )
