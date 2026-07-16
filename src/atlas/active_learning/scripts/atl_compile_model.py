#!/usr/bin/env python3
"""Backend-agnostic model-compilation script.

Deployed as PortableCode / run in-container on the node where the model will be
used for inference. Reads ``compile_settings.json``, dispatches to
``backend.compile_model`` (see ``MLIPModelCompiler``), and writes
``compile_results.json`` for the parser. The compiled artifact is device- and
node-specific, which is why this runs on the inference computer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    """Compile the shipped model and write standardized results."""
    # ``/atl_data`` only exists in the containerized deployment; otherwise the
    # working dir is the sandbox itself.
    prepend_path = Path('/atl_data') if Path('/atl_data').exists() else Path('.')

    with open(prepend_path / 'compile_settings.json') as f:
        cfg = json.load(f)

    from atlas.active_learning.backends import get_backend

    backend = get_backend(cfg['backend'])
    model_path = prepend_path / cfg['model_filename']

    compiled_name = None
    try:
        compiled = backend.compile_model(
            model_path,
            device=cfg.get('device', 'cpu'),
            mode=cfg.get('mode', 'aotinductor'),
            target=cfg.get('target', 'ase'),
        )
        if compiled is not None:
            compiled = Path(compiled)
            # Make sure the artifact sits in the retrievable working dir.
            if compiled.resolve().parent != prepend_path.resolve():
                dest = prepend_path / compiled.name
                if compiled.resolve() != dest.resolve():
                    dest.write_bytes(compiled.read_bytes())
                compiled = dest
            compiled_name = compiled.name
    finally:
        with open(prepend_path / 'compile_results.json', 'w') as f:
            json.dump({'compiled_model': compiled_name}, f)

    if compiled_name is None:
        sys.stderr.write(
            f"backend '{cfg['backend']}'.compile_model produced no artifact "
            f"for '{cfg['model_filename']}' (device={cfg.get('device')}).\n"
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
