import sys
sys.path.append(".")
from typing import TYPE_CHECKING
from common.dataset import DatasetFactory, save_dataset
from common.logger import configure_logger, logger

if TYPE_CHECKING:
    from common.config.config import DatasetConfig


def preprocess_dataset(dataset_config: "DatasetConfig"):
    logger.info(f"Preprocessing dataset: {dataset_config.type} {dataset_config.split}")
    dataset = DatasetFactory.get_dataset(dataset_config)
    logger.info(f"Dataset loaded: {len(dataset)} items")
    save_dataset(dataset, dataset_config.save_path)
    logger.info(f"Dataset saved: {dataset_config.save_path}")


if __name__ == "__main__":
    from common.config import get_config

    app_config = get_config()
    configure_logger(app_config.logger_config.print_level)
    preprocess_dataset(app_config.dataset_config)
