from miniml.component import Component
from miniml.io import InputSpec, OutputSpec
from miniml.pipeline import Pipeline, PipelineNode, PipelineInputSpec
from miniml.bindings import UpstreamOutputRef


# -------- Component A: data preparation --------
comp_a = Component(name="data_prep")


@comp_a.step("load")
def load():
    return {"features": [[1, 2], [3, 4], [5, 6]], "labels": [0, 1, 0]}


@comp_a.step("normalize", depends_on=["load"])
def normalize(data: dict):
    xs = data["features"]
    s = sum(sum(row) for row in xs) or 1
    return {"features": [[v / s for v in row] for row in xs], "labels": data["labels"]}


# 契约：A 无外部输入；输出用 normalize 的结果
comp_a.set_inputs({}, {})
comp_a.set_outputs(
    {"dataset": OutputSpec(name="dataset", type="table")},
    output_from_step={"dataset": "normalize"},
)


# -------- Component B: train --------
comp_b = Component(name="train")


@comp_b.step("ingest")  # 这个 step 将由 inputs 注入
def ingest(dataset: dict):
    return dataset


@comp_b.step("train", depends_on=["ingest"])
def train_step(dataset: dict):
    total_features = sum(sum(r) for r in dataset["features"])
    total_labels = sum(dataset["labels"])
    return {"bias": total_labels, "scale": total_features}


comp_b.set_inputs(
    {"dataset": InputSpec(name="dataset", type="table")},
    input_to_step={"dataset": "ingest"},  # Pipeline 会把 dataset 注入到 ingest
)
comp_b.set_outputs(
    {"model": OutputSpec(name="model", type="model")},
    output_from_step={"model": "train"},
)


# -------- Pipeline: stitch A -> B --------
pipe = Pipeline(name="demo")
pipe.add_input(
    PipelineInputSpec(name="debug_flag", type="bool", optional=True, default=False)
)

pipe.add_node(
    PipelineNode(
        name="prep",
        component=comp_a,
        input_bindings={},  # A 无输入
    )
)

pipe.add_node(
    PipelineNode(
        name="trainer",
        component=comp_b,
        input_bindings={
            # 将上游 A 的输出 dataset 绑定到 B 的输入 dataset
            "dataset": UpstreamOutputRef(node="prep", output="dataset"),
            # 也支持绑定 pipeline 顶层输入或常量，例如：
            # "debug": PipelineInputRef(name="debug_flag"),
            # "seed": ConstantValue(value=42),
        },
    )
)

if __name__ == "__main__":
    # 全图执行：返回 sink（trainer）的 outputs
    out_all = pipe.run()
    print("ALL:", out_all)

    # 组件级子图执行：只跑从 'prep' 到 'trainer'
    out_sub = pipe.run(start_component="prep", end_component="trainer")
    print("SUBGRAPH:", out_sub)

    # 组件内子图执行：例如只让 A 从 "normalize" 那步读输出
    # （注意：这通过修改 comp_a.run 的参数由你在实际调用时控制；
    #  在 Pipeline 内部我们默认黑盒调用 comp.run(inputs)，不传 start/end）
    a_only = comp_a.run(start_step="normalize", end_step="normalize")
    print("Component A normalize only:", a_only)
