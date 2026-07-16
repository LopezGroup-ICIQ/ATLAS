"""Mock (instant, local) surrogates for the remote active-learning calcjobs.

Enabled through the ``[debug.mock]`` section of the settings TOML. These replace
the heavy remote CalcJobs (training, descriptors, MD, DFT) with in-process AiiDA
``calcfunction``s that produce nodes with the same outputs the workchain consumes,
so the loop logic can be exercised locally in seconds. Results are physically
meaningless and must never be used for science.
"""

from atlas.active_learning.mock.mock_calcs import (
    MOCK_STAGES,
    mock_descriptors,
    mock_dft_label,
    mock_md_process,
    mock_test_db_eval,
    mock_train_model,
    stage_is_mocked,
)

__all__ = [
    'MOCK_STAGES',
    'mock_dft_label',
    'mock_descriptors',
    'mock_md_process',
    'mock_test_db_eval',
    'mock_train_model',
    'stage_is_mocked',
]
