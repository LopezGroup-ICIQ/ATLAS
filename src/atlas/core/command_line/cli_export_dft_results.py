"""Export the results of a running (or finished) DFT batch from its AiiDA group.

``atl_run_dft_database`` refreshes its own extxyz file on every polling cycle,
but that file only exists inside the process that owns the run. This command
reads the same results straight out of the AiiDA group, so a partial dataset can
be collected from a second shell while the batch is still in flight, without
touching the running process.
"""

import argparse
import pathlib as pl


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            'Export finished VASP DFT results from one or more AiiDA groups '
            'to an extxyz file. Safe to run while the batch is still running.'
        )
    )
    parser.add_argument(
        '--group',
        '-g',
        required=True,
        action='append',
        help=(
            'Label of the AiiDA group holding the calculations. '
            'Repeat the flag to export several groups into one file.'
        ),
        metavar='LABEL',
    )
    parser.add_argument(
        '--output',
        '-o',
        default='dft_partial_results.xyz',
        help='Path of the extxyz file to write. Default: dft_partial_results.xyz',
        metavar='FILE',
    )
    parser.add_argument(
        '--db_file',
        '-i',
        default=None,
        help=(
            'Optional source database (.extxyz/.xyz) the batch was launched '
            'from. When given, the results are merged onto the original '
            'structures so their metadata is preserved, and the unlabelled '
            'structures are kept in the output.'
        ),
        metavar='FILE',
    )
    parser.add_argument(
        '--include-failed',
        action='store_true',
        help=(
            'Also export calculations that terminated with a non-zero exit '
            'status. Off by default.'
        ),
    )
    return parser.parse_args()


def _last_calcjob(node, orm):
    """Return the last CalcJobNode below a workchain node, or the node itself."""
    if isinstance(node, orm.CalcJobNode):
        return node

    calcjob_nodes = [
        child for child in node.called_descendants if isinstance(child, orm.CalcJobNode)
    ]
    return calcjob_nodes[-1] if calcjob_nodes else None


def export_dft_results(
    group_labels: list,
    output_path: 'str | pl.Path',
    db_file: 'str | pl.Path | None' = None,
    include_failed: bool = False,
) -> pl.Path:
    """
    Collect finished DFT results from AiiDA groups and write them to extxyz.

    Parameters
    ----------
    group_labels : list[str]
        Labels of the AiiDA groups to query.
    output_path : str or pathlib.Path
        Destination extxyz file.
    db_file : str or pathlib.Path, optional
        Source database the batch was launched from. When given, results are
        merged onto the original structures, matching on ``atl_id``.
    include_failed : bool, optional
        Whether to also export calculations with a non-zero exit status.

    Returns
    -------
    pathlib.Path
        The path the results were written to.
    """
    from aiida import orm
    from ase.io import read as ase_read

    from atlas.active_learning import active_learning_utils as atl_al_ut
    from atlas.active_learning import conversion as atl_conv
    from atlas.core import code_utils as atl_cut

    # Only finished calculations can be parsed. `exit_status` is left out when
    # failed calculations are requested, so errored runs are included too.
    filter_dict = {'attributes.process_state': 'finished'}
    if not include_failed:
        filter_dict['attributes.exit_status'] = 0

    # Reusing the AL query helper: it already falls back to VaspCalculation
    # nodes for groups written by older versions of ATLAS.
    result_nodes = atl_conv._gather_result_nodes_aiida(
        path=None, aiida_group_list=group_labels, filter_dict=filter_dict
    )

    # Loading the source database, if one was given, so the exported structures
    # keep the metadata they were submitted with.
    source_structures = None
    if db_file is not None:
        source_structures = ase_read(pl.Path(db_file), format='extxyz', index=':')
        source_by_id = {
            struct.info.get('atl_id'): idx
            for idx, struct in enumerate(source_structures)
        }

    collected = []
    skipped = 0
    for node in result_nodes:
        calcjob = _last_calcjob(node, orm)
        if calcjob is None:
            skipped += 1
            continue

        labelled = atl_al_ut.build_ref_labelled_atoms(calcjob)
        if labelled is None:
            skipped += 1
            continue

        if source_structures is None:
            collected.append(labelled)
            continue

        # Matching the node back to its source structure through the extra the
        # submission step stored on it.
        unique_id = node.base.extras.all.get('atl_calc_uuid')
        idx = source_by_id.get(unique_id)
        if idx is None:
            skipped += 1
            continue

        merged = source_structures[idx].copy()
        merged.info.update(
            {
                key: val
                for key, val in labelled.info.items()
                if key.startswith('REF_') or key in ('aiida_uuid',)
            }
        )
        merged.info['calc_type'] = 'vasp_dft'
        merged.info['calc_performed'] = True
        merged.arrays['REF_forces'] = labelled.arrays['REF_forces']
        merged.arrays['positions'] = labelled.arrays['positions']
        source_structures[idx] = merged
        collected.append(merged)

    images = source_structures if source_structures is not None else collected

    output_path = atl_cut.atomic_write_extxyz(output_path, images)

    atl_cut.custom_print(
        f'Exported {len(collected)} finished calculations '
        f'({skipped} skipped) to {output_path}.',
        'done',
    )

    return output_path


def run_export_dft_results():
    args = parse_arguments()

    from atlas.core import code_utils as atl_cut

    atl_cut.init_logger('export_dft_results', log_path='.')

    export_dft_results(
        group_labels=args.group,
        output_path=args.output,
        db_file=args.db_file,
        include_failed=args.include_failed,
    )


if __name__ == '__main__':
    run_export_dft_results()
