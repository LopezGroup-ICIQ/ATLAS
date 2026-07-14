"""Tests for the MLIP backend registry and protocol system."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from atlas.active_learning.backends import (
    _BACKEND_REGISTRY,
    get_backend,
    list_backends,
    register_backend,
)
from atlas.active_learning.backends._base import (
    MLIPCalculatorFactory,
    MLIPCommitteeEvaluator,
    MLIPDescriptorProvider,
    MLIPTrainer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_registry():
    """Save and restore the backend registry around each test that registers
    temporary backends, so real backends (mace, allegro) aren't clobbered.
    """
    saved = dict(_BACKEND_REGISTRY)
    yield
    _BACKEND_REGISTRY.clear()
    _BACKEND_REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# Registry mechanics
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for register_backend / get_backend / list_backends."""

    def test_list_backends_returns_sorted(self):
        names = list_backends()
        assert names == sorted(names)

    def test_builtin_backends_registered(self):
        names = list_backends()
        assert 'mace' in names
        assert 'allegro' in names

    def test_get_backend_returns_instance(self):
        backend = get_backend('mace')
        assert backend is not None
        assert type(backend).__name__ == 'MACEBackend'

    def test_get_backend_allegro_returns_instance(self):
        backend = get_backend('allegro')
        assert type(backend).__name__ == 'AllegroBackend'

    def test_get_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown MLIP backend 'nonexistent'"):
            get_backend('nonexistent')

    def test_get_backend_error_lists_available(self):
        with pytest.raises(ValueError, match='allegro') as exc_info:
            get_backend('nonexistent')
        assert 'mace' in str(exc_info.value)

    def test_get_backend_returns_new_instance_each_call(self):
        b1 = get_backend('mace')
        b2 = get_backend('mace')
        assert b1 is not b2

    @pytest.mark.usefixtures('_clean_registry')
    def test_register_custom_backend(self):
        @register_backend('test-dummy')
        class DummyBackend:
            pass

        assert 'test-dummy' in list_backends()
        backend = get_backend('test-dummy')
        assert type(backend).__name__ == 'DummyBackend'

    @pytest.mark.usefixtures('_clean_registry')
    def test_register_overwrites_existing(self):
        @register_backend('test-dup')
        class First:
            pass

        @register_backend('test-dup')
        class Second:
            pass

        backend = get_backend('test-dup')
        assert type(backend).__name__ == 'Second'

    @pytest.mark.usefixtures('_clean_registry')
    def test_register_preserves_class(self):
        @register_backend('test-identity')
        class MyBackend:
            x = 42

        assert MyBackend.x == 42


# ---------------------------------------------------------------------------
# Protocol conformance - MACE
# ---------------------------------------------------------------------------


class TestMACEProtocolConformance:
    """Verify MACEBackend satisfies all four protocols."""

    def test_is_mlip_trainer(self):
        backend = get_backend('mace')
        assert isinstance(backend, MLIPTrainer)

    def test_is_mlip_calculator_factory(self):
        backend = get_backend('mace')
        assert isinstance(backend, MLIPCalculatorFactory)

    def test_is_mlip_descriptor_provider(self):
        backend = get_backend('mace')
        assert isinstance(backend, MLIPDescriptorProvider)

    def test_is_mlip_committee_evaluator(self):
        backend = get_backend('mace')
        assert isinstance(backend, MLIPCommitteeEvaluator)


class TestMACEBackendProperties:
    """Test static properties of the MACE backend."""

    def test_calcjob_entry_point(self):
        backend = get_backend('mace')
        assert backend.calcjob_entry_point == 'mace-train'

    def test_parser_entry_point(self):
        backend = get_backend('mace')
        assert backend.parser_entry_point == 'mace-training-parser'

    def test_model_file_extension(self):
        backend = get_backend('mace')
        assert backend.model_file_extension == '.model'

    def test_supports_committee_training(self):
        backend = get_backend('mace')
        assert backend.supports_committee_training is True


# ---------------------------------------------------------------------------
# Protocol conformance - Allegro
# ---------------------------------------------------------------------------


class TestAllegroProtocolConformance:
    """Verify AllegroBackend satisfies the expected protocols
    and does NOT satisfy MLIPDescriptorProvider.
    """

    def test_is_mlip_trainer(self):
        backend = get_backend('allegro')
        assert isinstance(backend, MLIPTrainer)

    def test_is_mlip_calculator_factory(self):
        backend = get_backend('allegro')
        assert isinstance(backend, MLIPCalculatorFactory)

    def test_is_not_mlip_descriptor_provider(self):
        backend = get_backend('allegro')
        assert not isinstance(backend, MLIPDescriptorProvider)

    def test_is_mlip_committee_evaluator(self):
        backend = get_backend('allegro')
        assert isinstance(backend, MLIPCommitteeEvaluator)


class TestAllegroBackendProperties:
    """Test static properties of the Allegro backend."""

    def test_calcjob_entry_point(self):
        backend = get_backend('allegro')
        assert backend.calcjob_entry_point == 'allegro-train'

    def test_parser_entry_point(self):
        backend = get_backend('allegro')
        assert backend.parser_entry_point == 'allegro-training-parser'

    def test_model_file_extension(self):
        backend = get_backend('allegro')
        assert backend.model_file_extension == '.nequip.zip'

    def test_supports_committee_training(self):
        backend = get_backend('allegro')
        assert backend.supports_committee_training is True


# ---------------------------------------------------------------------------
# Protocol definitions - structural subtyping
# ---------------------------------------------------------------------------


class TestProtocolStructuralSubtyping:
    """Verify that arbitrary classes satisfying the protocol shape
    are recognized, and that incomplete classes are not.
    """

    def test_minimal_calculator_factory_satisfies_protocol(self):
        class Minimal:
            @property
            def model_file_extension(self) -> str:
                return '.pt'

            @property
            def lammps_pair_style(self) -> str:
                return 'minimal'

            def create_calculator(
                self, model_path, device='cpu', dtype='float32', **kw
            ):
                return MagicMock()

        assert isinstance(Minimal(), MLIPCalculatorFactory)

    def test_missing_method_does_not_satisfy_protocol(self):
        class Incomplete:
            @property
            def model_file_extension(self) -> str:
                return '.pt'

        assert not isinstance(Incomplete(), MLIPCalculatorFactory)

    def test_minimal_descriptor_provider_satisfies_protocol(self):
        class Minimal:
            def generate_descriptors(self, database, model_path, settings, **kw):
                return {}, np.array([]), []

        assert isinstance(Minimal(), MLIPDescriptorProvider)

    def test_minimal_committee_evaluator_satisfies_protocol(self):
        class Minimal:
            @property
            def supports_committee_training(self) -> bool:
                return False

            def evaluate_committee(
                self, structures, model_files, device='cpu', dtype='float32', **kw
            ):
                return {}

        assert isinstance(Minimal(), MLIPCommitteeEvaluator)


# ---------------------------------------------------------------------------
# Descriptor dispatch fallback
# ---------------------------------------------------------------------------


class TestDescriptorDispatch:
    """Test the generate_descriptors dispatcher in active_learning_utils,
    verifying the fallback logic for backends without descriptor support.
    """

    @pytest.fixture(autouse=True)
    def _require_tomllib(self):
        pytest.importorskip('tomllib', reason='Python 3.11+ required for tomllib')

    def test_allegro_descriptor_raises(self):
        from atlas.active_learning.active_learning_utils import generate_descriptors

        with pytest.raises(ValueError, match='does not support descriptor generation'):
            generate_descriptors(
                database=[],
                descriptor_type='allegro',
                descriptor_settings={},
            )

    def test_unknown_descriptor_type_raises(self):
        from atlas.active_learning.active_learning_utils import generate_descriptors

        with pytest.raises(ValueError, match='Unknown descriptor type'):
            generate_descriptors(
                database=[],
                descriptor_type='nonexistent',
                descriptor_settings={},
            )


# ---------------------------------------------------------------------------
# Allegro training config builder
# ---------------------------------------------------------------------------


class TestAllegroTrainingConfig:
    """Test the Allegro YAML config builder."""

    def test_build_config_basic(self):
        from atlas.active_learning.backends.allegro.training import (
            build_allegro_train_config,
        )

        settings = {
            'train_settings': {
                'r_max': 6.0,
                'num_layers': 2,
                'l_max': 2,
                'lr': 0.005,
                'batch_size': 5,
                'max_num_epochs': 100,
            }
        }

        config = build_allegro_train_config(
            settings_dict=settings,
            train_data_path='/data/train.xyz',
            model_name='model_0',
            iteration=3,
            db_size=500,
        )

        assert config['data']['split_dataset']['file_path'] == 'train.xyz'
        assert config['data']['ase_args']['format'] == 'extxyz'
        assert config['cutoff_radius'] == 6.0
        assert config['training_module']['optimizer']['lr'] == 0.005
        assert config['data']['train_dataloader']['batch_size'] == 5
        assert config['trainer']['max_epochs'] == 100
        assert config['training_module']['model']['num_layers'] == 2
        assert config['training_module']['model']['l_max'] == 2
        assert config['training_module']['model']['seed'] is not None

    def test_build_config_default_dtype(self):
        from atlas.active_learning.backends.allegro.training import (
            build_allegro_train_config,
        )

        config = build_allegro_train_config(
            settings_dict={'train_settings': {}},
            train_data_path='train.xyz',
            model_name='m0',
            iteration=0,
            db_size=10,
        )
        assert config['training_module']['model']['model_dtype'] == 'float32'

    def test_build_config_has_allegro_model_builder(self):
        from atlas.active_learning.backends.allegro.training import (
            build_allegro_train_config,
        )

        config = build_allegro_train_config(
            settings_dict={'train_settings': {}},
            train_data_path='train.xyz',
            model_name='m0',
            iteration=0,
            db_size=10,
        )
        assert (
            config['training_module']['model']['_target_']
            == 'allegro.model.AllegroModel'
        )

    def test_build_config_extra_keys_passed_through(self):
        from atlas.active_learning.backends.allegro.training import (
            build_allegro_train_config,
        )

        config = build_allegro_train_config(
            settings_dict={'train_settings': {'custom_key': 'custom_value'}},
            train_data_path='train.xyz',
            model_name='m0',
            iteration=0,
            db_size=10,
        )
        assert config['custom_key'] == 'custom_value'


# ---------------------------------------------------------------------------
# Backend cross-comparison
# ---------------------------------------------------------------------------


class TestBackendDifferences:
    """Verify key differences between MACE and Allegro backends."""

    def test_different_model_extensions(self):
        mace = get_backend('mace')
        allegro = get_backend('allegro')
        assert mace.model_file_extension != allegro.model_file_extension

    def test_different_entry_points(self):
        mace = get_backend('mace')
        allegro = get_backend('allegro')
        assert mace.calcjob_entry_point != allegro.calcjob_entry_point
        assert mace.parser_entry_point != allegro.parser_entry_point

    def test_mace_has_descriptors_allegro_does_not(self):
        mace = get_backend('mace')
        allegro = get_backend('allegro')
        assert isinstance(mace, MLIPDescriptorProvider)
        assert not isinstance(allegro, MLIPDescriptorProvider)

    def test_both_support_committee(self):
        mace = get_backend('mace')
        allegro = get_backend('allegro')
        assert mace.supports_committee_training is True
        assert allegro.supports_committee_training is True

    def test_training_data_uses_same_format(self):
        """Both backends reuse gen_mace_train_structure_list (extxyz),
        so prepare_training_data should be callable with the same args.
        """
        mace = get_backend('mace')
        allegro = get_backend('allegro')
        assert hasattr(mace, 'prepare_training_data')
        assert hasattr(allegro, 'prepare_training_data')
