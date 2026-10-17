import sys
sys.path.append(".")
from common.config import get_config
from common.logger import configure_logger
from stage1_value_finding.value_retrieval import ValueRetrievalRunner

if __name__ == "__main__":
    app_config = get_config()
    configure_logger(app_config.logger_config.print_level)
    runner = ValueRetrievalRunner.from_config(app_config)
    runner.run()
