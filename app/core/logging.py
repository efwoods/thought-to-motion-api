import logging

# Logging config
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.debug("This is a DEBUG message")
logger.info("This is an INFO message")
logger.error("This is an ERROR message")
