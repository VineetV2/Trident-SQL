import sys
sys.path.append(".")

from common.config import get_config
from common.logger import configure_logger
from stage5_repair.sql_revision import SQLRevisionRunner


if __name__ == "__main__":
    app_config = get_config()
    configure_logger(app_config.logger_config.print_level)
    sql_reviser = SQLRevisionRunner.from_config(app_config)
    sql_reviser.run()
