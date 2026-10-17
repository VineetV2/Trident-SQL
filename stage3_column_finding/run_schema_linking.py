import sys
sys.path.append(".")
from common.config import get_config
from common.logger import configure_logger
from stage3_column_finding.schema_linking import SchemaLinkingRunner

if __name__ == "__main__":
    app_config = get_config()
    configure_logger(app_config.logger_config.print_level)
    runner = SchemaLinkingRunner.from_config(app_config)
    runner.run()
