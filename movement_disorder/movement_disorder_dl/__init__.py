# 2025-02-12 Anthony Lee

import logging
# Logger for library - opt-in approach - https://realpython.com/python-logging-source-code/#library-vs-application-logging-what-is-nullhandler
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import data
from . import model
from . import tuner