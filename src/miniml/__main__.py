from miniml.component import Component

component = Component(name="simple_ml")


@component.step(name="load_data_1")
def load_data_1() -> dict:
    return {"features": [[1, 2], [3, 4], [5, 6]], "labels": [0, 1, 0]}


@component.step(name="load_data_2")
def load_data_2() -> dict:
    return {"features": [[4, 2], [7, 4], [10, 12]], "labels": [1, 0, 1]}


@component.step(name="train_model", depends_on=["load_data_1", "load_data_2"])
def train_model(data_1: dict, data_2: dict) -> dict:
    total_features = data_1["features"] + data_2["features"]
    total_labels = data_1["labels"] + data_2["labels"]

    return {"bias": total_labels, "scale": total_features}


@component.step(name="evaluate", depends_on=["train_model"])
def evaluate(model: dict) -> float:
    return sum([sum(i) for i in model["scale"]]) / (sum(model["bias"]))


@component.step(name="analyze", depends_on=["evaluate"])
def analyze(score: float) -> None:
    pass


@component.step(name="visualize", depends_on=["evaluate"])
def visualize(score: float) -> None:
    pass

if __name__ == "__main__":
    # 1) 跑完整 pipeline
    print("=== first run ===")
    result_full = component.run()
    print("full sinks:", result_full)

    # # 2) 只跑到 train_model
    # print("\n=== run until train_model ===")
    # result_train = component.run(end_step="train_model")
    # print("train_model result:", result_train)

    # # 3) 重跑 evaluate（假设你改了评估逻辑）
    print("\n=== rerun from evaluate ===")
    result_eval = component.run(start_step="analyze")
    print("evaluate result:", result_eval)
