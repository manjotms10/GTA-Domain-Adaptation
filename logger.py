import glob
import logging
import logging.handlers

LOG_FILENAME = 'project.out'

# Set up a specific logger with our desired output level
logger = logging.getLogger('GTA-DA')
logger.setLevel(logging.DEBUG)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(
              LOG_FILENAME, maxBytes=20, backupCount=5)

logger.addHandler(handler)

# Log some messages
for i in range(20):
    logger.debug('i = %d' % i)

# See what files are created
logfiles = glob.glob('%s*' % LOG_FILENAME)
