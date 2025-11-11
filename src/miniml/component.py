from __future__ import annotations

from functools import wraps
import logging
from typing import Callable, Any
from rich.logging import RichHandler
from .step import Step
from .artifacts import ArtifactStore, InMemoryArtifactStore
from .exceptions import (
    StepNotFoundError,
    DuplicateStepError,
    InvalidSegmentError,
)

logging.basicConfig(level=logging.WARNING, format="[%(filename)s:%(lineno)s: %(funcName)15s() ] %(message)s", datefmt="[%X]", handlers=[RichHandler(show_level=True)])
logger = logging.getLogger("component")
logger.setLevel(logging.DEBUG)



class Component:
    """
    A self-contained DAG(directed acyclic graph) of steps.

    - Steps are registered via `@component.step(...)`.
    - No global registry: multiple Components are safe.
    - `run(start_step, end_step, inputs, use_cache)` executes a subgraph.
    - `rerun(start_step, ...)` 清楚该 step 下游缓存后再执行。
    """

    def __init__(
        self, name: str | None = None, artifact_store: ArtifactStore | None = None
    ) -> None:
        self.name = name or "component"
        self.steps: dict[str, Step] = {}
        self._children: dict[str, list[str]] = {}  # 下一个直接依赖
        self._artifact_store: ArtifactStore = (
            artifact_store or InMemoryArtifactStore()
        )  # artifact cache

    # ------------------------------------------------------------------
    # Step registration
    # ------------------------------------------------------------------
    def step(self, name: str | None = None, depends_on: list[str] | None = None):
        def decorator(func: Callable[..., Any]):
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
                # 依赖是否存在可以延后到 run 时统一校验

            self.steps[step_name] = Step(step_name, func, depends_on)
            self._children.clear()  # 标记需要重建 children

            @wraps(func)
            def wrapper(*args, **kwargs):
                # 允许用户直接调用原始函数做本地调试
                return func(*args, **kwargs)

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Public execution APIs
    # ------------------------------------------------------------------
    def run(
        self,
        start_step: str | None = None,
        end_step: str | None = None,
        inputs: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> Any:
        if not self.steps:
            raise InvalidSegmentError("No steps defined in this Component.")

        if not self._children:
            self._build_children()

        self._validate_all_dep_exist()

        if start_step and start_step not in self.steps:
            raise StepNotFoundError(f"Unknown start_step: {start_step!r}")
        if end_step and end_step not in self.steps:
            raise StepNotFoundError(f"Unknown end_step: {end_step!r}")

        inputs = inputs or {}

        # 1. 计算这次要「可能执行」的节点集合（segment）
        segment = self._compute_segment(start_step, end_step)
        logger.info(f"Computed segment: {segment}")

        # 2. 初始化本次运行的内存 cache
        cache: dict[str, Any] = {}

        # 先写入 inputs 和（可选）缓存的结果（仅对 segment 内的节点）
        for step_name in segment:
            if step_name in inputs:
                cache[step_name] = inputs[step_name]
            elif use_cache:
                cached = self._artifact_store.get(self.name, step_name)
                if cached is not None:
                    cache[step_name] = cached

        # 3. 定义执行逻辑
        def exec_step(name: str) -> Any:
            """
            执行的时候因为采用递归执行，
            所以是深度优先的。这个时候缓存很重要，因为可能
            在第一个递归分支的时候很多东西已经计算过了
            """
            logger.debug(f"Backpropagting on step '{name}' ... ")
            # 直接返回最后cache的结果，不实际run节点
            if name in cache:
                logger.debug(
                    f"Loading cached value {result} for Step {name} in Component '{self.name}' "
                )
                return cache[name]

            if name not in self.steps:
                raise StepNotFoundError(
                    f"Step {name!r} is not defined in Component {self.name!r}."
                )

            step = self.steps[name]

            # 收集依赖结果
            dep_results: list[Any] = []
            for dep in step.depends_on:
                # 优先：内存 cache（包括 inputs）
                if dep in cache:
                    logger.debug(f"Loading cached dep {cache[dep]} for Step {name} in Component '{self.name}' ")
                    dep_results.append(cache[dep])
                    continue

                # 如果依赖也在 segment 中：说明我们本次打算（或需要）重算它
                if dep in segment:
                    dep_results.append(exec_step(dep))
                    continue

                # 不在 segment：尝试从 artifact_store 读取历史结果
                # 比如当start_step是一个中间节点的时候，我们需要从一个非segment节点（即start_step的上一个节点）
                # 读取cached value作为start_step的输入
                if use_cache:
                    logger.debug(f"using cache '{dep}' not in segment...")
                    prev = self._artifact_store.get(self.name, dep)
                    if prev is not None:
                        cache[dep] = prev
                        dep_results.append(prev)
                        continue

                # 再尝试从 inputs（允许用户为非 segment 依赖手动提供值）
                if dep in inputs:
                    logger.debug("reading deps in input....")
                    cache[dep] = inputs[dep]
                    dep_results.append(inputs[dep])
                    continue

                # 到这里，说明依赖既不重算、也没有可用结果
                raise InvalidSegmentError(
                    f"Missing dependency {dep!r} for step {name!r}: "
                    f"it is not scheduled in the execution segment and "
                    f"no cached or input value is available."
                )

            # 真正执行当前 step（到了顶部节点，不再存在依赖，或者依赖已经计算完毕）
            # 在顶部节点，dep_results是一个空的list
            logger.debug(f"Forward propagting on step '{name}' (Calcuting)... ")
            result = step.func(*dep_results)
            cache[name] = result
            if use_cache:
                logger.debug(
                    f"Setting cached value for Step '{name}' in Component '{self.name}' "
                )
                self._artifact_store.set(self.name, name, result)

            return result

        # 4. 决定返回值
        if end_step:
            return exec_step(end_step)
        else:
            # 没有🔝end_step, 需要找出可能的一个或者多个实际上的end_step
            # 一个 sink 是指 在有向图（Directed Graph）中没有任何出边(outgoing edge)的节点。
            sinks: list[str] = [
                step_name
                for step_name in segment
                if not any(child in segment for child in self._children.get(step_name, []))
            ] # 遍历所有segment中的step中没有children为空的step name

            #执行这些找到的end_step
            logger.debug(f"Found sinks : {sinks}")
            return {name: exec_step(name) for name in sinks}


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_children(self) -> None:
        if self._children:
            return

        children: dict[str, list[str]] = {name: [] for name in self.steps}
        for name, step in self.steps.items():
            for dep in step.depends_on:
                children.setdefault(dep, []).append(
                    name
                )  # 这里返回的list是一个reference list，指向的是children中的dict中的value
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
        """
        这里定义「本次可能会执行哪些节点」：

        - start=None, end=None: 整个图
        - start=None, end=E: 所有能到达 E 的上游（含 E）
        - start=S, end=None: S 以及所有从 S 出发能到的下游
        - start=S, end=E: 从 S 出发能到、且能到达 E 的节点交集
          （这些是“会被重算的候选节点”；依赖可以来自 segment 外的缓存）
        """
        all_nodes = set(self.steps.keys())

        if start_step:
            forward = self._dfs_forward(start_step)
        else:
            forward = all_nodes

        if end_step:
            backward = self._dfs_backward(end_step)
        else:
            backward = all_nodes

        segment = forward & backward
        if not segment:
            raise InvalidSegmentError(
                f"No executable segment for start_step={start_step!r}, end_step={end_step!r}"
            )

        if start_step and start_step not in segment:
            raise InvalidSegmentError(
                f"start_step={start_step!r} is not included in computed segment."
            )
        if end_step and end_step not in segment:
            raise InvalidSegmentError(
                f"end_step={end_step!r} is not reachable given start_step={start_step!r}."
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
