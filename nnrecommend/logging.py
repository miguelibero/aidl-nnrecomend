import logging
import coloredlogs
import os
import sys


COLOR_FMT = '%(asctime)s %(name)s %(message)s'
NOCOLOR_FMT = '%(asctime)s %(name)s [%(levelname)s] %(message)s'


class RemoveStacktraceFilter(logging.Filter):
    def filter(self, record):
        if record.exc_info and len(record.exc_info) > 1:
            exc = record.exc_info[1]
            if not isinstance(exc, Exception):
                record.msg = "(%s) %s" % (type(exc).__name__, record.msg)
        record.exc_info = None
        record.stack_info = None
        record.exc_text = None
        return True


def get_logger(name):
    if not isinstance(name, str):
        clsname = None
        if not clsname and hasattr(name, '__name__'):
            clsname = name.__name__
        if not clsname and hasattr(name, '__class__'):
            clsname = name.__class__.__name__
        parts = [name.__module__]
        if clsname:
            parts.append(clsname)
        name = ".".join(parts)
    return logging.getLogger(name)


DISABLED_LOGGERS = ()


def setup_log(verbose, logfile=None):

    for name in DISABLED_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)

    logger = logging.getLogger()

    if logfile:
        file_formatter = logging.Formatter(NOCOLOR_FMT)
        logdir = os.path.dirname(logfile)
        if logdir:
            os.makedirs(logdir, exist_ok=True)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    supports_color = os.isatty(sys.stdout.fileno())
    if os.name == 'nt':
        supports_color = False
    if supports_color:
        stream_formatter = coloredlogs.ColoredFormatter(fmt=COLOR_FMT)
    else:
        stream_formatter = logging.Formatter(fmt=NOCOLOR_FMT)
    stream_handler = logging.StreamHandler()
    if not verbose:
        stream_handler.addFilter(RemoveStacktraceFilter())
    stream_handler.setFormatter(stream_formatter)
    lvl = logging.DEBUG if verbose else logging.INFO

    logger.logdebug = verbose
    stream_handler.setLevel(lvl)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    return logger