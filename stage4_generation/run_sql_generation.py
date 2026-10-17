import sys
sys.path.append(".")
from common.config import get_config
from common.logger import configure_logger
from stage4_generation.sql_generation import SQLGenerationRunner

if __name__ == "__main__":
    app_config = get_config()
    configure_logger(app_config.logger_config.print_level)
    runner = SQLGenerationRunner.from_config(app_config)
    runner.run()
