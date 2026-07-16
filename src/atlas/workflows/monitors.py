"""Monitors for the ATLAS workflows."""

import tempfile

from aiida.orm import CalcJobNode
from aiida.transports import Transport

KEY_VERSION_ROOT: str = '0.1'
KEY_VERSION_CORE: str = '0.1'
KEY_VERSION_PLUGIN: str = '0.1'
__version__ = '0.1'


def output_monitor(node: CalcJobNode, transport: Transport) -> str:
    """Retrieve and inspect files in working directory of job to determine whether
    the job should be killed.

    :param node: The node representing the calculation job.
    :param transport: The transport that can be used to retrieve files from remote
        working directory.

    :returns: A string if the job should be killed, `None` otherwise.
    """
    output_monitor.__version__ = '0.1'
    with tempfile.NamedTemporaryFile('w+') as handle:
        transport.getfile(node.options.output_filename, handle.name)
        handle.seek(0)
        output = handle.read()

    hermitian_warning = 'WARNING: Sub-Space-Matrix'

    if hermitian_warning in output:
        return (
            'Detected a problem (Sub-Space-Matrix is not hermitian in DAV) in the'
            ' output file.'
        )
