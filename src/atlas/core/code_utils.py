"""Utility functions for code manipulation."""

import functools
import logging
import os
import pathlib as pl
import subprocess as sb
import tempfile
import warnings
from logging import LogRecord

import qrcode
from packaging.version import Version
from packaging.version import parse as parse_version
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

from atlas import ATL_ROOT_DIR, __repo__, __version__

ATL_THEME = Theme(
    {
        # Level Styling (matches the rewritten names in ATLRichHandler)
        'logging.level.[ i ]': 'blue',
        'logging.level.[ ! ]': 'yellow',
        'logging.level.[...]': 'white dim',
        'logging.level.[ ✔ ]': 'green bold',
        'logging.level.[ x ]': 'red bold',
        # Message Highlighting
        'atl.path': 'magenta italic',
        'atl.number': 'cyan bold',
        'atl.success': 'green underline',
        'atl.failure': 'red bold underline',
    }
)


def display_qr_in_cli(data: str):
    """Generates and displays a QR code in the terminal.

    Parameters
    ----------
    data : str
        The data to encode in the QR code (e.g., a URL).
    """
    qr = qrcode.QRCode(version=1, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    qr.print_tty()


def save_qr_to_file(data: str, filename: str = 'qr_code.png'):
    """Generates a QR code and saves it as an image file."""
    qr = qrcode.QRCode(version=4, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white')
    img.save(filename)


class ATLHighlighter(RegexHighlighter):
    """
    Apply custom highlighting to log messages using Regex.

    This highlights patterns in the message body, such as file paths,
    numbers, or specific status keywords.
    """

    base_style = 'atl.'
    highlights = [
        r'(?P<path>[\w.\-/]+\.(py|log|txt|json|yaml))',  # File paths
        r'(?P<number>\b\d+\b)',  # Numbers
        r'(?P<success>Done|Success|Completed)',  # Success words
        r'(?P<failure>Error|Failed|Exception)',  # Failure words
    ]


class ATLRichHandler(RichHandler):
    """
    Custom RichHandler that rewrites logging level names locally.

    This allows us to display '[ i ]' instead of 'INFO' on the console
    without using logging.addLevelName() to change it globally (which
    would affect file logs and other libraries).
    """

    # Mapping standard level names/numbers to your custom icons
    _LEVEL_MAP = {
        'DEBUG': '[...]',
        'INFO': '[ i ]',
        'WARNING': '[ ! ]',
        'ERROR': '[ X ]',
        'CRITICAL': '[!!!]',
        'ATL_DEBUG': '    ',
        'SUCCESS': '[ ✔ ]',
        # Map AiiDA's REPORT level
        'REPORT': '[ i ]',
    }

    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercept the log record to modify the levelname before rendering.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be processed.
        """
        # We must create a copy of the record. If we modify the original
        # record, it will be modified for the FileHandler too, which
        # we do not want.
        record_copy = logging.makeLogRecord(record.__dict__)
        # Rewrite the level name if it exists in our map
        # Otherwise, keep the original (e.g., 'INFO')
        if record_copy.levelname in self._LEVEL_MAP:
            record_copy.levelname = self._LEVEL_MAP[record_copy.levelname]
        elif record_copy.levelno == 25:  # Handle custom integer levels
            record_copy.levelname = '[ ✔ ]'

        # Pass the modified copy to the parent RichHandler
        super().emit(record_copy)


def get_console_handler() -> tuple[RichHandler, Console]:
    """
    Sets up the custom console and handler with the ATL theme.

    Returns
    -------
    tuple[RichHandler, Console]
        The configured handler and console instance.
    """
    # Create Console with the highlighter
    console = Console(theme=ATL_THEME, highlighter=ATLHighlighter())

    # 3. Create the custom handler
    ch = ATLRichHandler(
        markup=True,
        show_path=False,
        log_time_format='[%m/%d/%y %H:%M:%S]',
        omit_repeated_times=False,
        console=console,
        level=11,  # Show everything above DEBUG(10)
    )

    formatter_con = logging.Formatter('%(message)s')
    ch.setFormatter(formatter_con)
    ch.set_name('atl_rich_handler')
    return ch, console


def logging_set_levels():
    logging.addLevelName(15, 'ATL_DEBUG')
    logging.addLevelName(25, 'SUCCESS')
    logging.addLevelName(19, '     ')


def create_handler_filters(handler: str):
    def handler_filter(record: LogRecord):
        return not (hasattr(record, 'block') and record.block == handler)

    return handler_filter


def init_logger(
    source: str, log_path=None, show_log_path: bool = True
) -> (logging.Logger, str):
    logger: logging.Logger = logging.getLogger(name='mdb')
    logger.setLevel(level=logging.DEBUG)

    logger.propagate = False

    # Console logger
    ch, console = get_console_handler()
    logger.addHandler(hdlr=ch)

    if not log_path:
        _, filename = tempfile.mkstemp(prefix=f'atl_{source}_', suffix='.log')
    else:
        log_path_dir = pl.Path(log_path)
        _, filename = tempfile.mkstemp(
            prefix=f'atl_{source}_', suffix='.log', dir=log_path_dir
        )

    fh = logging.FileHandler(filename=filename, mode='a+')
    fh.set_name(name='atl_file_handler')
    fh.addFilter(filter=create_handler_filters(handler='file'))
    fh.setLevel(level=logging.DEBUG)
    formatter_fil = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(shortmsg)s')
    fh.setFormatter(fmt=formatter_fil)
    logger.addHandler(hdlr=fh)

    logging_set_levels()

    if show_log_path:
        custom_print(string=f"Logging in '{filename}'", print_type='info')

    return logger, filename


class LevelNameFilter(logging.Filter):
    """Filters log records based on a list of allowed level names."""

    def __init__(self, levels_to_keep):
        super().__init__()
        self.levels_to_keep = set(levels_to_keep)

    def filter(self, record):
        return record.levelname in self.levels_to_keep


def custom_print(
    string: str,
    print_type: str = 'default',
    end='\n',
    extra_tab=False,
    logger=None,
    extras: dict = None,
):
    r"""Prints a string using different formatting styles for easier debugging.

    Parameters
    ----------
    string : str
        Text to be printed
    print_type : str, optional, `default=info`
        Style to use when printing. Available styles are:
            - `info/default`: prefixes [ i ] before the string.
            - `warning/warn`: prefixes [ ! ] before the string.
            - `debug/extra`: prefixes [...] before the string.
            - `done/ok`: prefixes [ ✔ ] before the string.
            - `error/problem`: prefixes [ X ] before the string.
            - `none/clean/clear/empty`: leaves an empty space before the string.
    end : str, optional, `default=\n`
        String appended after the last value, default a newline.
    extra_tab : bool, optional, `default=False`
        If True, adds an extra tab before the string.
    logger : logging.Logger, optional, `default=None`
        Logger to use for printing. If None, a new logger named 'mdb' is created

    Returns
    -------
    logging.Logger
        Logger used for printing the string
    """
    extra_tab_str = '\t' if extra_tab else ''

    # WARNING: Check if this is being repeated
    if not logger:
        logger = logging.getLogger('mdb')
        logger.propagate = False

    # Allows to use the custom print function without initializing
    # the logger first
    if logger.handlers == []:
        logger.setLevel(logging.DEBUG)
        ch, _ = get_console_handler()
        logger.addHandler(ch)
        logging_set_levels()

    # Default to empty dict if None
    extra_data = {'shortmsg': string, **(extras or {})}
    message = f'{extra_tab_str}{string}'

    match print_type:
        case 'info' | 'default':
            # Level 20: [ i ]
            logger.log(level=20, msg=message, extra=extra_data)
        case 'warn' | 'warning' | 'warn-soft' | 'warning-soft':
            # Level 30: [ ! ]
            logger.log(level=30, msg=message, extra=extra_data)
        case 'extra' | 'debug':
            # Level 10: [...]
            logger.log(level=10, msg=message, extra=extra_data)
        case 'none' | 'clean' | 'clear' | 'empty':
            # Level 15: ATL_DEBUG / Empty prefix
            logger.log(level=15, msg=message, extra=extra_data)
        case 'done' | 'ok':
            # Level 25: [ ✔ ]
            logger.log(level=25, msg=message, extra=extra_data)
        case 'error' | 'problem' | 'err':
            # Level 40: [ X ]
            logger.log(level=40, msg=message, extra=extra_data)
        case _:
            # Fallback for unknown types
            logger.log(level=20, msg=message, extra=extra_data)

    return logger


def atl_show_progress(
    iterable, total=None, interval=100, level_tag='[ i ]', prepend='MACE:'
):
    """
    Shows a rich progress bar with a custom format matching ATL logs.

    Parameters
    ----------
    iterable : iterable
        The iterable to wrap.
    total : int, optional
        The total number of items in the iterable.
    interval : int, optional
        The number of iterations between updates of the timestamp.
    level_tag : str, optional
        The logging level tag to display.
    prepend : str, optional
        The string to prepend to the progress bar.
    """
    import datetime

    progress = Progress(
        TextColumn('[{task.fields[timestamp]}]'),
        TextColumn('{task.fields[level_tag]} {task.fields[prepend]}'),
        BarColumn(),
        '[progress.percentage]{task.percentage:>3.0f}%',
        '({task.completed}/{task.total})',
        TimeElapsedColumn(),
        '<',
        TimeRemainingColumn(),
        console=Console(theme=ATL_THEME),
        transient=False,
        refresh_per_second=1,
    )
    with progress:
        task_id = progress.add_task(
            'progress',
            total=total,
            timestamp=datetime.datetime.now().strftime('%m/%d/%y %H:%M:%S'),
            level_tag=level_tag,
            prepend=prepend,
        )
        for i, item in enumerate(iterable):
            yield item
            if i % interval == 0:
                progress.update(
                    task_id,
                    advance=1,
                    timestamp=datetime.datetime.now().strftime('%m/%d/%y %H:%M:%S'),
                )
            else:
                progress.update(task_id, advance=1)


def deprecated(reason, since_ver=None):
    """
    Decorator to mark a function as deprecated.

    Parameters
    ----------
    reason : str
        Reason to print for the deprecation of the old function

    Example
    -------
    Use it as a decorator:

    >>> @deprecated(reason="Use to_cluster instead.", since_ver="0.6.2")
    >>> def to_atoms():
    >>>     pass

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_text = f'{func.__name__}() is deprecated: {reason}.'
            if since_ver:
                warn_text += f"\n(Deprecated since version: '{since_ver}')."
            warnings.warn(warn_text, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_config_path() -> pl.Path:
    """Get the path to ATLAS's the configuration directory."""
    # Try to get XDG_CONFIG_HOME, if it doesn't exist, return None
    config_path = os.environ.get('XDG_CONFIG_HOME', None)

    # Check if $HOME/.config exists and if it does, return the path
    if not config_path:
        config_folder = pl.Path().home() / '.config'
        if config_folder.exists():
            config_path = config_folder

    return pl.Path(config_path)


def init_config_dir(config_dir, config_file: str):
    """Create the configuration directory and the secrets file template."""
    # Create a 'mdb' directory inside the config directory
    config_dir = config_dir / 'mdb'
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create a 'config_file' file inside the 'mdb' directory
    try:
        file_path = config_dir / config_file

        with open(file_path, 'x') as f:
            f.write('{\n"API_KEY": ""\n}')

        # Limiting the permissions of the secrets file
        # to the owner only
        os.chmod(file_path, 0o700)

        return True, config_dir

    except FileExistsError:
        return False, config_dir


def get_cache_path() -> pl.Path:
    """Get the path to ATLAS's the cacheuration directory."""
    # Try to get XDG_cache_HOME, if it doesn't exist, return None
    cache_path = os.environ.get('XDG_CACHE_HOME', None)

    # Check if $HOME/.cache exists and if it does, return the path
    if not cache_path:
        cache_folder = pl.Path().home() / '.cache'
        if cache_folder.exists():
            cache_path = cache_folder

    return pl.Path(cache_path)


def init_cache_dir(cache_dir):
    """Create the mdb cache directory."""
    # Create an mdb folder inside the ~/.cache directory
    cache_dir = cache_dir / 'mdb'

    try:
        cache_dir.mkdir(parents=True, exist_ok=False)
        return cache_dir
    except FileExistsError:
        return None


def get_last_tagged_version():
    """
    Get the last tagged version from the GitHub repository (via SSH).
    If SSH access fails, fall back to checking the local Git repository.

    Returns
    -------
    str
        Last tagged version in the repository.
    str
        Hash of the current commit.
    """
    try:
        # Try fetching the latest tag from GitHub via SSH
        output = (
            sb.check_output(
                ['git', 'ls-remote', '--tags', __repo__],
                stderr=sb.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )

        # Extract tag names from the output
        tags = [
            line.split('/')[-1].replace('^{}', '').replace('v', '')
            for line in output.split('\n')
        ]
        # Get the newest tag. Use versioning.version.parse to handle the version sorting
        # Return '0.0.0' if no tags are found
        newest_tag = (
            sorted(tags, reverse=True, key=lambda v: parse_version(v))[0]
            if tags
            else '0.0.0'
        )

        # Get the latest commit hash from the remote
        hash_str = (
            sb.check_output(
                ['git', 'ls-remote', __repo__, 'HEAD'],
                timeout=5,
            )
            .decode()
            .split()[0]
        )

    except (sb.CalledProcessError, IndexError, sb.TimeoutExpired):
        # If SSH access fails, fall back to local repository
        custom_print(
            'Unable to fetch the latest version from remote: '
            f"'{__repo__}'. "
            'Falling back to local repository check.',
            'warn',
        )
        newest_tag, hash_str = get_last_tagged_version_local()

    return newest_tag, hash_str


def get_last_tagged_version_local(repo_dir_path: str = None):
    """
    Get the last tagged version from the local Git repository.

    Returns
    -------
    str
        Last tagged version in the local repository.
    str
        Hash of the current commit.
    """
    # Save the current directory path
    cwd = os.getcwd()

    # Use the default ATLAS root directory if no path is provided
    if not repo_dir_path or not pl.Path(repo_dir_path).exists():
        repo_dir_path = ATL_ROOT_DIR

    try:
        # Change to the ATLAS root directory
        os.chdir(repo_dir_path)

        # Split the output into a list of tags
        tags = get_list_of_tags()

        # Sort tags in descending order
        tags = sorted(tags, reverse=True)

        # Get the newest tag (first in the sorted list)
        newest_tag = tags[0] if tags else '0.0.0'

        # Get the current commit hash
        try:
            current_hash = (
                sb.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            )
        except sb.CalledProcessError:
            current_hash = 'unknown'

    finally:
        # Ensure we switch back to the original directory
        os.chdir(cwd)

    return newest_tag, current_hash


def get_list_of_tags(
    repo_path: str | pl.Path | None = None,
) -> list[Version]:
    """
    Get a list of tags from the git repository in the CWD.

    Parameters
    ----------
    repo_path : str, optional
        Path to the git repository. If None, uses the current working directory.

    Returns
    -------
    list[str]
        List of tags in the repository.
    """
    # Use the given repository path if provided
    if repo_path:
        os.chdir(repo_path)

    # Run the git fetch to get the last tagged version
    try:
        _ = sb.check_output(['git', 'fetch', '--tags', '--quiet'], timeout=5)
    except (sb.CalledProcessError, sb.TimeoutExpired):
        return '0.0.0', 'unknown'

    # Getting a sorted tag list
    output = (
        sb.check_output(['git', 'tag', '--merged', 'master', '--sort=-creatordate'])
        .decode()
        .strip()
    )

    # Split the output into a list of tags
    tags = output.split('\n') if output else []

    # Convert to version object
    tags = [Version(tag) for tag in tags]

    return tags


def get_atl_version_info():
    """
    Get the current version of ATLAS and the last tagged version in the repository.

    Returns
    -------
    Version
        Current version of ATLAS
    Version
        Last tagged version in the repository
    str
        Hash of the current commit
    """
    # Check the current version of ATLAS
    curr_version = Version(__version__)

    # Check the last tagged version in the repository
    ver, hash_str = get_last_tagged_version()
    if isinstance(ver, Version) or ver == 'unknown':
        last_tagged_version = ver
    else:
        last_tagged_version = Version(ver)

    return curr_version, last_tagged_version, hash_str


def check_atl_version(logger=None):
    """
    Check and print if the current version of ATLAS is up-to-date.

    Returns
    -------
    tuple[Version, Version]
        Current version of ATLAS and the last tagged version in the repository.
    """
    curr_version, last_tagged_version, hash_str = get_atl_version_info()

    print()
    new_logger = False
    if not logger:
        new_logger = True
        logger, _ = init_logger('atl_version_check', show_log_path=False)

    if hash_str == 'unknown' or last_tagged_version == 'unknown':
        custom_print('Unable to fetch the latest version!', 'error')
    elif curr_version < last_tagged_version:
        custom_print(
            (
                f'Current version of ATLAS ({curr_version}, {hash_str[:7]}) '
                'is outdated. Please update to the latest '
                f'version ({last_tagged_version}).'
            ),
            'warn',
        )
    elif curr_version > last_tagged_version:
        custom_print(
            (
                '[bold yellow]Current version of ATLAS '
                f'({curr_version}, {hash_str[:7]}) is an unrealeased version.'
            ),
            'warn',
        )
    else:
        custom_print(
            (
                f'Current version of ATLAS ({curr_version}, {hash_str[:7]}) '
                'is up-to-date.'
            ),
            'done',
        )

    print()

    # Clearing version check logger
    if new_logger:
        logger.handlers.clear()

    return curr_version, last_tagged_version
