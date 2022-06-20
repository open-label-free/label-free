from typing import Dict
from label_free.utils import load_object


class ExperimentConfig(Dict):
    ...


def train(experiment_config: ExperimentConfig):

    # #############
    # Load datasets
    # #############

    train_dataset = load_object(
        **experiment_config["train_dataset"]
    )
    val_dataset = load_object(
        **experiment_config["val_dataset"]
    )

    if "test_dataset" in experiment_config:

        test_dataset = load_object(
            **experiment_config["test_dataset"]
        )

    # ##########
    # Load model
    # ##########

    model = load_object(**experiment_config["model"])
    

    # ###############
    # Load experiment
    # ###############

    experiment_config["kwargs"]["train_dataset"] = model
    experiment_config["kwargs"]["val_dataset"] = model
    experiment_config["kwargs"]["test_dataset"] = model

    experiment_config["kwargs"]["model"] = model

    experiment = load_object(**experiment_config)

    experiment.train()
    experiment.test()
