"""Tests for the MLIP backend registry and protocol system."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from atlas.active_learning.backends import (
    _BACKEND_REGISTRY,
    find_inference_model,
    get_backend,
    list_backends,
    parse_model_spec,
    register_backend,
    resolve_model_spec,
    trainable_backends,
)
from atlas.active_learning.backends._base import (
    MLIPCalculatorFactory,
    MLIPCommitteeEvaluator,
    MLIPDescriptorProvider,
    MLIPPretrainedModel,
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


class TestPretrainedProtocolConformance:
    """Which backends serve published pretrained models."""

    def test_mace_is_pretrained_provider(self):
        assert isinstance(get_backend('mace'), MLIPPretrainedModel)

    def test_allegro_is_not_pretrained_provider(self):
        # Allegro serves no published foundation models; omitting the protocol
        # is how a backend says so.
        assert not isinstance(get_backend('allegro'), MLIPPretrainedModel)

    def test_trainable_backends(self):
        assert trainable_backends() == ['allegro', 'mace']


class TestParseModelSpec:
    """Splitting a '<backend>:<variant>' spec from a path or plain name."""

    def test_bare_backend(self):
        assert parse_model_spec('mace') == ('mace', None)

    def test_backend_with_variant(self):
        assert parse_model_spec('mace:mp-small') == ('mace', 'mp-small')

    def test_unregistered_prefix_is_not_a_spec(self):
        assert parse_model_spec('orb:orb-v2') is None

    def test_posix_path_is_not_a_spec(self):
        assert parse_model_spec('/models/curr_model.model') is None

    def test_windows_path_is_not_a_spec(self):
        # The guard is 'prefix is a registered backend', not 'contains a colon'.
        assert parse_model_spec('C:\\models\\curr_model.model') is None

    def test_soap_is_not_a_spec(self):
        assert parse_model_spec('soap') is None

    def test_non_string(self):
        assert parse_model_spec(None) is None


class TestResolveModelSpec:
    """Resolving a spec into a (backend, pretrained id) pair."""

    def test_bare_trainable_backend_resolves_from_disk(self):
        # A bare trainable backend must keep meaning "the model the AL loop
        # trained", never a foundation model.
        assert resolve_model_spec('mace') == ('mace', None)

    def test_variant_resolves_through_the_backend(self):
        assert resolve_model_spec('mace:mp-small') == ('mace', 'mace:mp-small')

    def test_benchmark_bare_variant_keeps_working(self):
        # The benchmark config's historical 'mace:small' means the mp family.
        assert resolve_model_spec('mace:small') == ('mace', 'mace:mp-small')

    def test_unparseable_spec_falls_back_to_default_backend(self):
        assert resolve_model_spec(None, 'mace') == ('mace', None)

    def test_variant_on_backend_without_pretrained_models_raises(self):
        with pytest.raises(ValueError, match='does not provide pretrained'):
            resolve_model_spec('allegro:something')


class TestMaceResolvePretrainedModel:
    """MACE foundation-model variant normalisation."""

    def test_bare_variant_defaults_to_mp_family(self):
        assert get_backend('mace').resolve_pretrained_model('small') == 'mace:mp-small'

    def test_explicit_mp_family(self):
        assert (
            get_backend('mace').resolve_pretrained_model('mp-small') == 'mace:mp-small'
        )

    def test_off_family_is_reachable(self):
        assert (
            get_backend('mace').resolve_pretrained_model('off-medium')
            == 'mace:off-medium'
        )

    def test_off23_alias_maps_to_off_family(self):
        # The benchmark config spells the off family 'off23-*'.
        assert (
            get_backend('mace').resolve_pretrained_model('off23-small')
            == 'mace:off-small'
        )

    def test_hyphenated_variant_is_not_split_as_a_family(self):
        assert (
            get_backend('mace').resolve_pretrained_model('medium-mpa-0')
            == 'mace:mp-medium-mpa-0'
        )

    def test_none_returns_default(self):
        assert get_backend('mace').resolve_pretrained_model(None) == 'mace:mp-medium'

    def test_unknown_variant_passes_through(self):
        # The accepted set depends on the installed mace-torch, so an unknown
        # variant warns and is passed to the framework rather than rejected.
        assert (
            get_backend('mace').resolve_pretrained_model('some-future-model')
            == 'mace:mp-some-future-model'
        )


class TestFindInferenceModel:
    """Choosing the model reference handed to create_calculator."""

    def test_explicit_pretrained_id_wins(self, tmp_path):
        (tmp_path / 'curr_model.model').write_text('x')
        result = find_inference_model(
            tmp_path,
            'curr_model',
            get_backend('mace'),
            pretrained_model='mace:mp-small',
        )
        assert result == 'mace:mp-small'

    def test_resolves_trained_model_path(self, tmp_path):
        (tmp_path / 'curr_model.model').write_text('x')
        result = find_inference_model(tmp_path, 'curr_model', get_backend('mace'))
        assert result == tmp_path / 'curr_model.model'

    def test_missing_trained_model_does_not_fall_back_to_foundation(self, tmp_path):
        """A missing trained model must surface as a missing path.

        Substituting the backend's default foundation model would run the MD on
        a different potential and produce plausible but wrong results.
        """
        result = find_inference_model(tmp_path, 'curr_model', get_backend('mace'))
        assert result == tmp_path / 'curr_model.model'
        assert not isinstance(result, str)
