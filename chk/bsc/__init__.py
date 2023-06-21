import logging
import os

logger = logging.getLogger("IterCheckpoint")
formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s', '%H:%M:%S')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
fh = logging.FileHandler('IterCheckpoint.log', 'w')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.propagate = False
debug_flag = os.environ.get('ITERCHECKPOINT_DEBUG')
if debug_flag is not None and int(debug_flag) > 0:
    logger.setLevel(logging.DEBUG)
    logger.info("Enable debugging!")
else:
    logger.setLevel(logging.INFO)