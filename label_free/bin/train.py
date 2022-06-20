from typing import Dict
from label_free.utils import load_object


class ExperimentConfig(Dict):
    ...


def train(experiment_config: ExperimentConfig):

    # #############
    # Load datasets
    # #############

    experiment_config["kwargs"]["dataloader_train"] = load_object(
        **experiment_config["dataloader_train"]
    )
    experiment_config["kwargs"]["dataloader_validation"] = load_object(
        **experiment_config["dataloader_validation"]
    )

    if "dataset_test" in experiment_config:

        experiment_config["kwargs"]["dataloader_test"] = load_object(
            **experiment_config["dataloader_test"]
        )

    # ##########
    # Load model
    # ##########

    model = load_object(**experiment_config.model)
    experiment_config["kwargs"]["model"] = model

    # ###############
    # Load experiment
    # ###############

    experiment = load_object(**experiment_config)

    experiment.train()
