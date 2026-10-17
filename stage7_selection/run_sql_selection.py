import sys
sys.path.append(".")

from common.config import get_config
from common.logger import configure_logger
from stage7_selection.sql_selection import SQLSelectionRunner


if __name__ == "__main__":
    app_config = get_config()
    configure_logger(app_config.logger_config.print_level)
    sql_selection_runner = SQLSelectionRunner.from_config(app_config)
    sql_selection_runner.run()
