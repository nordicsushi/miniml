# examples/pipeline_inputs_showcase.py
from __future__ import annotations

from typing import Any

from miniml.component import Component, ComponentContract
from miniml.io import InputSpec, OutputSpec
from miniml.pipeline import Pipeline, PipelineNode, PipelineInputSpec
from miniml.bindings import PipelineInputRef, UpstreamOutputRef, ConstantValue


# ======================
# Component A: 数据准备
# ======================
comp_prep = Component(
    name="prep",
    contract=ComponentContract(
        inputs={
            "dataset_path": InputSpec(
                name="dataset_path", type="string", optional=True, default=None
            ),
            "debug": InputSpec(name="debug", type="bool", optional=True, default=False),
        },
        outputs={
            "dataset": OutputSpec(name="dataset", type="table"),
            "meta": OutputSpec(name="meta", type="json"),
        },
        input_to_step={
            "dataset_path": "load",
            "debug": "load",
        },
        output_from_step={
            "dataset": "normalize",
            "meta": "summarize",
        },
    ),
)


@comp_prep.step("load")
def load(dataset_path: str | None = None, debug: bool = False) -> dict[str, Any]:
    if dataset_path:
        if debug:
            print(f"[prep.load] reading from path: {dataset_path}")
        data = {"features": [[10, 20], [30, 40], [50, 60]], "labels": [1, 0, 1]}
    else:
        if debug:
            print("[prep.load] using in-memory mock data")
        data = {"features": [[1, 2], [3, 4], [5, 6]], "labels": [0, 1, 0]}
    return data


@comp_prep.step("normalize", depends_on=["load"])
def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    xs = raw["features"]
    s = sum(sum(row) for row in xs) or 1
    return {"features": [[v / s for v in row] for row in xs], "labels": raw["labels"]}


@comp_prep.step("summarize", depends_on=["normalize"])
def summarize(ds: dict[str, Any]) -> dict[str, Any]:
    n = len(ds["features"])
    d = len(ds["features"][0]) if n else 0
    return {"rows": n, "cols": d}


# ======================
# Component B: 切分数据
# ======================
comp_split = Component(
    name="split",
    contract=ComponentContract(
        inputs={
            "dataset": InputSpec(name="dataset", type="table"),
            "train_ratio": InputSpec(name="train_ratio", type="float", optional=False),
            "seed": InputSpec(name="seed", type="int", optional=True, default=42),
        },
        outputs={
            "train_ds": OutputSpec(name="train_ds", type="table"),
            "test_ds": OutputSpec(name="test_ds", type="table"),
        },
        input_to_step={
            "dataset": "ingest",
            "train_ratio": "ingest",
            "seed": "ingest",
        },
        output_from_step={
            "train_ds": ("do_split", "train_ds"),
            "test_ds": ("do_split", "test_ds"),
        },
    ),
)


@comp_split.step("ingest")
def ingest(dataset: dict, train_ratio: float, seed: int) -> tuple[dict, float, int]:
    return dataset, train_ratio, seed


@comp_split.step("do_split", depends_on=["ingest"])
def do_split(payload: tuple[dict, float, int]) -> dict[str, dict]:
    dataset, train_ratio, seed = payload
    n = len(dataset["features"])
    k = max(1, int(n * train_ratio))
    train = {"features": dataset["features"][:k], "labels": dataset["labels"][:k]}
    test = {"features": dataset["features"][k:], "labels": dataset["labels"][k:]}
    return {"train_ds": train, "test_ds": test}


# ======================
# Component C: 训练模型
# ======================
comp_train = Component(
    name="train",
    contract=ComponentContract(
        inputs={
            "train_ds": InputSpec(name="train_ds", type="table"),
            "learning_rate": InputSpec(
                name="learning_rate", type="float", optional=True, default=0.1
            ),
        },
        outputs={
            "model": OutputSpec(name="model", type="model"),
        },
        input_to_step={
            "train_ds": "ingest",
            "learning_rate": "ingest",
        },
        output_from_step={"model": "fit"},
    ),
)


@comp_train.step("ingest")
def t_ingest(train_ds: dict, learning_rate: float) -> tuple[dict, float]:
    return train_ds, learning_rate


@comp_train.step("fit", depends_on=["ingest"])
def fit(payload: tuple[dict, float]) -> dict[str, float]:
    train_ds, lr = payload
    total_features = sum(sum(r) for r in train_ds["features"]) or 1.0
    total_labels = sum(train_ds["labels"])
    return {"bias": float(total_labels), "scale": float(total_features), "lr": lr}


# ======================
# Component D: 评估
# ======================
comp_eval = Component(
    name="evaluate",
    contract=ComponentContract(
        inputs={
            "model": InputSpec(name="model", type="model"),
            "threshold": InputSpec(
                name="threshold", type="float", optional=True, default=0.05
            ),
            "debug": InputSpec(name="debug", type="bool", optional=True, default=False),
        },
        outputs={
            "score": OutputSpec(name="score", type="float"),
            "passed": OutputSpec(name="passed", type="bool"),
        },
        input_to_step={
            "model": "ingest",
            "threshold": "ingest",
            "debug": "ingest",
        },
        output_from_step={
            "score": "metric",
            "passed": "check",
        },
    ),
)


@comp_eval.step("ingest")
def e_ingest(model: dict, threshold: float, debug: bool) -> tuple[dict, float, bool]:
    return model, threshold, debug


@comp_eval.step("metric", depends_on=["ingest"])
def metric(payload: tuple[dict, float, bool]) -> float:
    model, _, debug = payload
    score = model["bias"] / (model["scale"] or 1.0)
    if debug:
        print(f"[evaluate.metric] score={score:.6f}")
    return float(score)


@comp_eval.step("check", depends_on=["metric", "ingest"])
def check(score: float, payload: tuple[dict, float, bool]) -> bool:
    _, threshold, _ = payload
    return bool(score >= threshold)


# ======================
# Pipeline: 组件级黑盒编排 + 顶层输入
# ======================
pipe = Pipeline(name="demo-with-pipeline-inputs")

# 顶层参数（PipelineInputSpec）
pipe.add_input(
    PipelineInputSpec(name="dataset_path", type="string", optional=True, default=None)
)
pipe.add_input(
    PipelineInputSpec(name="debug", type="bool", optional=True, default=False)
)
pipe.add_input(PipelineInputSpec(name="train_ratio", type="float", optional=False))
pipe.add_input(PipelineInputSpec(name="seed", type="int", optional=True, default=123))
pipe.add_input(
    PipelineInputSpec(name="threshold", type="float", optional=True, default=0.05)
)

# 添加节点并绑定输入
pipe.add_node(
    PipelineNode(
        name="prep",
        component=comp_prep,
        input_bindings={
            "dataset_path": PipelineInputRef(name="dataset_path"),
            "debug": PipelineInputRef(name="debug"),
        },
    )
)

pipe.add_node(
    PipelineNode(
        name="split",
        component=comp_split,
        input_bindings={
            "dataset": UpstreamOutputRef(node="prep", output="dataset"),
            "train_ratio": PipelineInputRef(name="train_ratio"),
            "seed": PipelineInputRef(name="seed"),
        },
    )
)

pipe.add_node(
    PipelineNode(
        name="train",
        component=comp_train,
        input_bindings={
            "train_ds": UpstreamOutputRef(node="split", output="train_ds"),
            "learning_rate": ConstantValue(value=0.2),
        },
    )
)

pipe.add_node(
    PipelineNode(
        name="evaluate",
        component=comp_eval,
        input_bindings={
            "model": UpstreamOutputRef(node="train", output="model"),
            "threshold": PipelineInputRef(name="threshold"),
            "debug": PipelineInputRef(name="debug"),
        },
    )
)

# 等价 link 用法（演示可选）
# pipe.link("prep", "dataset", "split", "dataset")
# pipe.link("split", "train_ds", "train", "train_ds")
# pipe.link("train", "model", "evaluate", "model")


if __name__ == "__main__":
    print("\n=== 1) 全图运行（提供必填 train_ratio，其余走默认） ===")
    out_all = pipe.run(
        pipeline_inputs={
            "train_ratio": 0.67,
        }
    )
    print("ALL:", out_all)  # {'evaluate': {'score': ..., 'passed': True/False}}

    print("\n=== 2) 修改顶层输入（传入 dataset_path / debug / threshold） ===")
    out_custom = pipe.run(
        pipeline_inputs={
            "dataset_path": "/data/my-dataset.csv",
            "debug": True,
            "train_ratio": 0.6,
            "seed": 999,
            "threshold": 0.01,
        }
    )
    print("CUSTOM:", out_custom)

    print("\n=== 3) 组件级子图执行：从 split 到 evaluate（注入上游 dataset） ===")
    # 先单独拿到 prep 的输出（或来自你自己的缓存/存储）
    prep_outputs = comp_prep.run(inputs={"dataset_path": None, "debug": False})
    ds = prep_outputs["dataset"]

    out_sub = pipe.run(
        pipeline_inputs={"train_ratio": 0.5},
        start_component="split",
        end_component="evaluate",
        provided_outputs={
            ("prep", "dataset"): ds,  # 注入上游 prep.dataset
        },
    )
    print("SUBGRAPH split→evaluate:", out_sub)

    print("\n=== 4) 组件内子图执行：只取 prep 的 normalize 输出（绕过 pipeline） ===")
    prep_only = comp_prep.run(
        start_step="normalize",
        end_step="normalize",
        inputs={"dataset_path": None, "debug": False},
    )
    print("prep.normalize only:", prep_only)

    print("\n=== 5) 演示必填参数缺失（应报错） ===")
    try:
        pipe.run()  # train_ratio 未提供且无默认 → 抛异常
    except Exception as e:
        print("EXPECTED ERROR:", repr(e))
