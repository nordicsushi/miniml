from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .bindings import BindingValue, PipelineInputRef, UpstreamOutputRef, ConstantValue
from .exceptions import InvalidSegmentError
from .component import Component


class PipelineInputSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    type: str = "any"
    optional: bool = False
    default: Any | None = None
    description: str | None = None


class PipelineNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str  # 节点在 Pipeline 内的别名（唯一）
    component: Component
    # 组件输入名 -> 绑定（pipeline 输入 / 上游输出 / 常量）
    input_bindings: dict[str, BindingValue] = Field(default_factory=dict)


class Pipeline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    # 顶层 pipeline 输入（可作为各节点绑定来源）
    inputs: dict[str, PipelineInputSpec] = Field(default_factory=dict)
    nodes: dict[str, PipelineNode] = Field(default_factory=dict)
    # 组件级拓扑缓存（children 图）：node_name -> 下游节点列表
    _children: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    # ------------ 构建 ------------
    def add_input(self, spec: PipelineInputSpec) -> None:
        self.inputs[spec.name] = spec

    def add_node(self, node: PipelineNode) -> None:
        if node.name in self.nodes:
            raise InvalidSegmentError(f"Duplicate node name in Pipeline: {node.name!r}")
        self.nodes[node.name] = node
        self._children.clear()

    def link(  # 便捷 API：把上游输出连到下游输入
        self,
        upstream_node: str,
        upstream_output: str,
        downstream_node: str,
        downstream_input: str,
    ) -> None:
        if downstream_node not in self.nodes or upstream_node not in self.nodes:
            raise InvalidSegmentError("link(): node not found.")
        self.nodes[downstream_node].input_bindings[downstream_input] = (
            UpstreamOutputRef(node=upstream_node, output=upstream_output)
        )
        self._children.clear()

    # ------------ 执行 ------------
    def run(
        self,
        pipeline_inputs: dict[str, Any] | None = None,
        start_component: str | None = None,
        end_component: str | None = None,
        provided_outputs: dict[tuple[str, str], Any] | None = None,
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """
        返回：
        - 若指定 `end_component`：返回该节点的 outputs（dict[str, Any]）
        - 否则：返回子图内所有 sink 节点的 outputs：dict[node_name, outputs_dict]

        provided_outputs:
            当从中间组件开始执行、其输入来自子图外的上游节点时，
            用于显式注入这些上游的输出。
            key = (upstream_node_name, upstream_output_name)
            value = 该输出的具体对象
        """
        if not self.nodes:
            raise InvalidSegmentError("Pipeline has no nodes.")

        pipeline_inputs = pipeline_inputs or {}
        provided_outputs = provided_outputs or {}

        self._build_children()
        self._validate_bindings()

        if start_component and start_component not in self.nodes:
            raise InvalidSegmentError(f"Unknown start_component: {start_component!r}")
        if end_component and end_component not in self.nodes:
            raise InvalidSegmentError(f"Unknown end_component: {end_component!r}")

        segment = self._compute_segment(start_component, end_component)

        # topo 执行
        context: dict[str, dict[str, Any]] = {}  # node_name -> outputs dict
        order = self._topo_order(segment)

        for node_name in order:
            node = self.nodes[node_name]
            # 解析该节点的 component inputs
            comp_inputs: dict[str, Any] = {}
            for in_name, binding in node.input_bindings.items():
                if isinstance(binding, ConstantValue):
                    comp_inputs[in_name] = binding.value
                    continue

                if isinstance(binding, PipelineInputRef):
                    spec = self.inputs.get(binding.name)
                    if binding.name in pipeline_inputs:
                        comp_inputs[in_name] = pipeline_inputs[binding.name]
                    else:
                        if spec is None:
                            raise InvalidSegmentError(
                                f"Unknown pipeline input {binding.name!r}."
                            )
                        # 注意：默认值可能是 0、""、False 等，要直接使用；
                        # 只有 default is None 再看 optional，optional 才允许传 None
                        if spec.default is not None:
                            comp_inputs[in_name] = spec.default
                        elif spec.optional:
                            comp_inputs[in_name] = None
                        else:
                            raise InvalidSegmentError(
                                f"Pipeline input {binding.name!r} is required and missing (no default)."
                            )
                    continue

                if isinstance(binding, UpstreamOutputRef):
                    key = (binding.node, binding.output)

                    # 1) 本次 segment 已执行的上游
                    if binding.node in context:
                        up_out = context[binding.node]
                        if binding.output not in up_out:
                            raise InvalidSegmentError(
                                f"Upstream {binding.node!r} has no output {binding.output!r}."
                            )
                        comp_inputs[in_name] = up_out[binding.output]
                        continue

                    # 2) 上游不在本次 segment：允许 provided_outputs 注入
                    if binding.node not in segment:
                        if key in provided_outputs:
                            comp_inputs[in_name] = provided_outputs[key]
                            continue
                        raise InvalidSegmentError(
                            f"Node {node.name!r} requires upstream {binding.node!r}.{binding.output!r}, "
                            f"which is outside the execution segment. Provide it via "
                            f"provided_outputs={{({binding.node!r}, {binding.output!r}): value}}."
                        )

                    # 3) 逻辑错误（在 segment 里却没跑到）
                    raise InvalidSegmentError(
                        f"Upstream node {binding.node!r} has not executed yet."
                    )

                raise InvalidSegmentError("Unknown binding type")

            # 真正执行该组件（黑盒）
            outputs = node.component.run(inputs=comp_inputs)
            context[node_name] = outputs

        # 返回 end_component 或子图内 sink 集合
        if end_component:
            return context[end_component]

        sinks = [
            n
            for n in segment
            if not any(c in segment for c in self._children.get(n, []))
        ]
        return {n: context[n] for n in sinks}

    # ------------ 内部工具 ------------
    def _build_children(self) -> None:
        if self._children:
            return
        children: dict[str, list[str]] = {n: [] for n in self.nodes}
        for dn, node in self.nodes.items():
            for b in node.input_bindings.values():
                if isinstance(b, UpstreamOutputRef):
                    children.setdefault(b.node, []).append(dn)
        self._children = children

    def _dfs_forward(self, start: str) -> set[str]:
        visited: set[str] = set()
        stack: list[str] = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(self._children.get(cur, []))
        return visited

    def _dfs_backward(self, end: str) -> set[str]:
        # 逆图：根据 _children 找到所有能到达 end 的上游
        visited: set[str] = set()

        parents: dict[str, list[str]] = {n: [] for n in self.nodes}
        for p, ch in self._children.items():
            for c in ch:
                parents[c].append(p)

        stack: list[str] = [end]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(parents.get(cur, []))
        return visited

    def _compute_segment(self, start: str | None, end: str | None) -> set[str]:
        all_nodes = set(self.nodes)
        forward = self._dfs_forward(start) if start else all_nodes
        backward = self._dfs_backward(end) if end else all_nodes
        seg = forward & backward
        if not seg:
            raise InvalidSegmentError(
                f"No executable component segment for start={start!r}, end={end!r}"
            )
        if start and start not in seg:
            raise InvalidSegmentError(
                f"start_component={start!r} not in computed segment."
            )
        if end and end not in seg:
            raise InvalidSegmentError(
                f"end_component={end!r} not reachable from {start!r}."
            )
        return seg

    def _topo_order(self, segment: set[str]) -> list[str]:
        # 简易 Kahn 算法（仅对 segment 内）
        indeg: dict[str, int] = {n: 0 for n in segment}
        for p, children in self._children.items():
            if p not in segment:
                continue
            for c in children:
                if c in segment:
                    indeg[c] += 1

        q: list[str] = [n for n, d in indeg.items() if d == 0]
        order: list[str] = []
        while q:
            n = q.pop()
            order.append(n)
            for ch in self._children.get(n, []):
                if ch in segment:
                    indeg[ch] -= 1
                    if indeg[ch] == 0:
                        q.append(ch)

        if len(order) != len(segment):
            raise InvalidSegmentError("Cycle detected in component-level DAG.")
        return order

    def _validate_bindings(self) -> None:
        # 校验所有 node 的 input_bindings 是否映射到：
        # - pipeline 输入 或
        # - 上游输出 或
        # - 常量
        # 并简要做类型/存在性检查
        for node in self.nodes.values():
            comp = node.component
            for in_name in comp.contract.inputs.keys():
                if in_name not in node.input_bindings:
                    spec = comp.contract.inputs[in_name]
                    if not spec.optional and spec.default is None:
                        raise InvalidSegmentError(
                            f"Node {node.name!r} missing binding for required input {in_name!r}"
                        )
            # 上游输出存在性检查（轻量；运行期还会再查）
            for b in node.input_bindings.values():
                if isinstance(b, UpstreamOutputRef) and b.node not in self.nodes:
                    raise InvalidSegmentError(
                        f"Binding references unknown upstream node: {b.node!r}"
                    )
