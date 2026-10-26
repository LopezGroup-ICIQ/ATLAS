"""Initial database generation page (``atl_gen_init_db``)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from atlas.core.gui.pages.base import WorkflowPage
from atlas.core.gui.process.runner import (
    DetachedProcessMonitor,
    ProcessRunner,
    find_detached_process,
)
from atlas.core.gui.project import Project
from atlas.core.gui.widgets.boundary_panel import BoundaryPanel
from atlas.core.gui.widgets.db_manage_panel import DbManagePanel
from atlas.core.gui.widgets.prereq_banner import SuccessBanner
from atlas.core.gui.widgets.structures_panel import (
    StructuresPanel,
    StructuresTablePanel,
)
from atlas.core.gui.widgets.workflow_view import WorkflowStep

CANONICAL_CONFIG = 'database_generation_settings.toml'

DB_GEN_TABS: list[tuple[str, list[str]]] = [
    ('Database', ['database']),
    ('Phase Diagram', ['phase_diagram']),
    ('Generation', ['generation']),
    (
        'Modifications',
        [
            'deformation',
            'perturbation',
            'vacancies',
            'adsorbates',
            'targeted_modification',
        ],
    ),
    ('Filters', ['struct_filters']),
    ('Hull', ['concave_hull']),
]

_STEP_NAV: dict[str, tuple[int, str]] = {
    'database': (1, 'prototype'),
    'targeted_modification': (3, 'central_atom_octahedral'),
    'generation.bulk': (2, 'num_struct'),
    'generation.surface': (2, 'min_miller_index'),
    'generation.cluster': (2, 'size_range'),
    'min_num_atoms': (0, 'min_num_atoms'),
    'deformation': (3, 'lattice_frac_deform_max'),
    'perturbation': (3, 'perturbation_ang'),
    'vacancies': (3, 'min_vacancy_percentage'),
    'composition.size': (0, 'size'),
    'adsorbates': (3, 'adsorbate_species'),
    'struct_filters': (4, ''),
    'concave_hull': (5, 'gen_concave_hull'),
    'export': (0, 'export'),
}


def _compute_workflow(
    config: dict,
) -> tuple[list[WorkflowStep], int, int | None]:
    """Derive the pipeline steps and rough structure estimates from *config*.

    Returns ``(steps, num_phases, total_estimate)``.
    """
    db_dict = config.get('database', {})
    gen_dict = config.get('generation', {})
    gen_types = gen_dict.get('generate_type', [])
    if isinstance(gen_types, str):
        gen_types = [gen_types]
    phase_dict = config.get('phase_diagram', {})
    phases = phase_dict.get('phase', {}) or {}
    num_phases = max(len(phases), 1)

    steps: list[WorkflowStep] = []
    total = 0

    # -- pre-phase ----------------------------------------------------------

    steps.append(
        WorkflowStep(
            name='Gather Base Structures',
            description=f'{num_phases} phase(s) from Materials Project',
            estimated_count=num_phases,
            config_key='database',
            category='generation',
        )
    )
    total += num_phases

    target_mod = config.get('targeted_modification', {})
    cao = target_mod.get('central_atom_octahedral', {})
    if cao:
        limit = int(cao.get('limit_max_num_modifications', 0) or 0)
        steps.append(
            WorkflowStep(
                name='Targeted Modification',
                description='Central atom octahedral perturbation',
                estimated_count=limit or None,
                config_key='targeted_modification',
                category='modification',
            )
        )
        total += limit

    # -- per-phase ----------------------------------------------------------

    if 'bulk' in gen_types and gen_dict.get('bulk'):
        bulk = gen_dict['bulk']
        n_s = int(bulk.get('num_struct', 2) or 2)
        n_r = int(bulk.get('num_repeat', 2) or 2)
        per = n_s * n_r
        steps.append(
            WorkflowStep(
                name='Bulk Generation',
                description=f'{n_s} compositions × {n_r} repeats per phase',
                estimated_count=per,
                config_key='generation.bulk',
                category='generation',
                group='per_phase',
            )
        )
        total += per * num_phases

    if 'surface' in gen_types and gen_dict.get('surface'):
        surf = gen_dict['surface']
        cap = int(surf.get('max_number_supercells', 100) or 100)
        steps.append(
            WorkflowStep(
                name='Surface Generation',
                description=f'Capped at {cap:,} per phase',
                estimated_count=cap,
                config_key='generation.surface',
                category='generation',
                group='per_phase',
            )
        )
        total += cap * num_phases

    if 'cluster' in gen_types and gen_dict.get('cluster'):
        clust = gen_dict['cluster']
        sr = clust.get('size_range', [])
        n_sizes = len(sr) if isinstance(sr, list) else 0
        n_s = int(clust.get('num_struct', 2) or 2)
        n_r = int(clust.get('num_repeat', 2) or 2)
        per = max(n_sizes, 1) * n_s * n_r
        steps.append(
            WorkflowStep(
                name='Cluster Generation',
                description=f'{n_sizes} sizes × {n_s} × {n_r} per phase',
                estimated_count=per,
                config_key='generation.cluster',
                category='generation',
                group='per_phase',
            )
        )
        total += per * num_phases

    min_at = db_dict.get('min_num_atoms', 0) or 0
    max_at = db_dict.get('max_num_atoms', 999) or 999
    steps.append(
        WorkflowStep(
            name='Atom Count Filter',
            description=f'Keep {min_at}–{max_at} atoms',
            is_filter=True,
            config_key='min_num_atoms',
            category='filter',
            group='per_phase',
        )
    )

    if config.get('deformation'):
        d = config['deformation']
        limit = int(d.get('limit_max_num_deformations', 100) or 100)
        steps.append(
            WorkflowStep(
                name='Lattice Deformation',
                description=f'Up to {limit:,} per phase',
                estimated_count=limit,
                config_key='deformation',
                category='modification',
                group='per_phase',
            )
        )
        total += limit * num_phases

    if config.get('perturbation'):
        p = config['perturbation']
        limit = int(p.get('limit_max_num_perturbs', 100) or 100)
        steps.append(
            WorkflowStep(
                name='Random Perturbation',
                description=f'Up to {limit:,} per phase',
                estimated_count=limit,
                config_key='perturbation',
                category='modification',
                group='per_phase',
            )
        )
        total += limit * num_phases

    if config.get('vacancies'):
        v = config['vacancies']
        limit = int(v.get('limit_max_num_vacancies', 100) or 100)
        steps.append(
            WorkflowStep(
                name='Vacancy Generation',
                description=f'Up to {limit:,} per phase',
                estimated_count=limit,
                config_key='vacancies',
                category='modification',
                group='per_phase',
            )
        )
        total += limit * num_phases

    comp = db_dict.get('composition', {}) or {}
    global_size = comp.get('size')
    has_per_phase_limit = any(
        pd.get('limit_max_num_structures')
        for pd in phases.values()
        if isinstance(pd, dict)
    )
    if global_size or has_per_phase_limit:
        if global_size:
            ppl = max(1, int(global_size) // num_phases)
            desc = f'Target ~{ppl:,} per phase'
        else:
            desc = 'Per-phase limits applied'
        steps.append(
            WorkflowStep(
                name='Per-Phase Limit',
                description=desc,
                is_filter=True,
                config_key='composition.size',
                category='filter',
                group='per_phase',
            )
        )
        if global_size:
            total = min(total, int(global_size))

    # -- post-phase ---------------------------------------------------------

    if config.get('adsorbates'):
        ads = config['adsorbates']
        limit = int(ads.get('limit_max_num_perturbs', 100) or 100)
        steps.append(
            WorkflowStep(
                name='Adsorbate Addition',
                description=f'Up to {limit:,} adsorbate structures',
                estimated_count=limit,
                config_key='adsorbates',
                category='modification',
                group='post_phase',
            )
        )
        total += limit

    if config.get('struct_filters'):
        steps.append(
            WorkflowStep(
                name='Structure Filtering',
                description='User-defined structure filters',
                is_filter=True,
                config_key='struct_filters',
                category='filter',
                group='post_phase',
            )
        )

    elements: set[str] = set()
    el_list = phase_dict.get('element_list', [])
    if isinstance(el_list, list):
        elements.update(el_list)
    for pd in phases.values():
        if isinstance(pd, dict):
            for k in pd.get('composition') or {}:
                if isinstance(k, str) and k != 'offset':
                    elements.add(k)
    n_el = max(len(elements), 1)
    steps.append(
        WorkflowStep(
            name='Isolated Atom Addition',
            description=f'{n_el} unique element(s)',
            estimated_count=n_el,
            config_key='',
            category='output',
            group='post_phase',
        )
    )
    total += n_el

    hull = config.get('concave_hull', {}) or {}
    steps.append(
        WorkflowStep(
            name='Boundary Generation',
            description='Concave hull / morphological closing',
            is_active=bool(hull.get('gen_concave_hull')),
            config_key='concave_hull',
            category='output',
            group='post_phase',
        )
    )

    export = db_dict.get('export', {}) or {}
    do_export = bool(export.get('export'))
    steps.append(
        WorkflowStep(
            name='Database Export',
            description=export.get('format', 'extxyz') if do_export else '',
            is_active=do_export,
            config_key='export',
            category='output',
            group='post_phase',
        )
    )

    return steps, num_phases, total if total > 0 else None


class InitDbPage(WorkflowPage):
    """Configure and launch initial database generation."""

    DISPLAY_NAME = 'Initial DB'
    NAVIGATION_KEY = 'init_db'

    def __init__(
        self,
        project: Project,
        schema_data,
        application_font,
        log,
        navigate,
        notification=None,
        parent=None,
    ):
        super().__init__(
            project,
            schema_data,
            application_font,
            log,
            navigate,
            notification,
            parent,
        )

        self.config_panel = self._build_config_panel()
        self.config_panel.data_changed.connect(self._refresh_workflow)
        self.config_panel.workflow_step_clicked.connect(self._on_workflow_nav)
        self.run_button = QPushButton(
            self._themed_icon('play_arrow'), ' Run Database Generation'
        )
        self.run_button.clicked.connect(self.run)
        self.cancel_button = self._make_cancel_button()
        self.config_panel.add_action_button(self.run_button)
        self.config_panel.add_action_button(self.cancel_button)

        self.structures_panel = StructuresPanel(project)
        self.table_panel = StructuresTablePanel(project)
        self.boundary_panel = BoundaryPanel(project)
        # The latent-space generation pane and the Config tab edit the same
        # configuration, so each surface picks up the other's saves.
        self.boundary_panel.config_updated.connect(
            self.config_panel.reload_active_snapshot
        )
        self.config_panel.save_succeeded.connect(
            self.boundary_panel.reload_generation_settings
        )
        self.boundary_panel.log_message.connect(self._log)
        self.boundary_panel.database_missing.connect(
            lambda: self.tabs.setCurrentIndex(0)
        )
        # Traversal and regeneration drive the shared status-bar progress bar.
        self.boundary_panel.busy_changed.connect(self.process_running_changed.emit)
        self.boundary_panel.progress_changed.connect(self.progress_changed.emit)
        self._manage_panel = DbManagePanel(project)
        self._manage_panel.database_deleted.connect(self._on_db_deleted)

        self.tabs = QTabWidget()
        self.tabs.setObjectName('pageTab')
        self.tabs.addTab(self.config_panel, 'Config')
        self.tabs.addTab(self._manage_panel, 'Manage')
        self.tabs.addTab(self.structures_panel, 'Outputs, Database')
        self.tabs.addTab(self.table_panel, 'Table View')
        self.tabs.addTab(self.boundary_panel, 'Outputs, Boundary')
        self._mark_output_tabs(2)

        self.success_banner = SuccessBanner(
            message='Database generated successfully!',
            actions=[
                ('View Outputs', self._go_to_outputs),
                ('Go to DFT Labelling →', lambda: self._navigate('dft')),
            ],
        )
        self.success_banner.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.success_banner)
        layout.addWidget(self.tabs)

        self._refresh_workflow()

    # ----------------------------------------------------- workflow diagram

    def _refresh_workflow(self) -> None:
        config = self.config_panel.collect_data()
        steps, num_phases, total = _compute_workflow(config)
        self.config_panel.set_workflow_steps(steps, num_phases, total)

    def _on_workflow_nav(self, config_key: str) -> None:
        nav = _STEP_NAV.get(config_key)
        if nav is None:
            return
        tab_idx, field_key = nav
        if self.config_panel._form_tabs is not None:
            self.config_panel._form_tabs.setCurrentIndex(tab_idx)
        if field_key:
            self.config_panel.focus_field(field_key)

    def _build_config_panel(self):
        from atlas.core.gui.widgets.config_panel import ConfigPanel

        return ConfigPanel(
            schema_data=self._schema_data,
            section_key='database_generation',
            sub_section_tabs=DB_GEN_TABS,
            project=self.project,
            application_font=self._application_font,
        )

    def run(self) -> None:
        if not self.config_panel.save_to_project(label='run'):
            self._log('❌ Cannot run: configuration could not be saved.')
            return

        parsed, err = self.config_panel.parsed_config()
        if err or not parsed:
            self._log(f'❌ Cannot run: {err or "empty configuration"}')
            return
        phases = (parsed.get('phase_diagram') or {}).get('phase')
        if not phases:
            self._log(
                '❌ Cannot run: no phases defined in the phase diagram.\n'
                '   Add at least one phase (with a prototype and composition) '
                'before generating the database.'
            )
            return

        success, errors, warnings = self.config_panel._run_validator(parsed)
        if not success:
            self._log('❌ Cannot run: configuration validation failed:')
            for e in errors:
                self._log(f'   • {e}')
            self._log('Fix the errors above before generating the database.')
            return
        if warnings:
            for w in warnings:
                self._log(f'⚠ {w}')

        if (
            QMessageBox.question(
                self,
                'Confirm Run',
                'Generate the initial database with the current configuration?',
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        self.success_banner.hide()
        command = ['atl_gen_init_db', 'generate', '-c', CANONICAL_CONFIG]
        self._set_running(self.run_button, self.cancel_button)
        self.worker = ProcessRunner(
            command,
            cwd=self.project.cwd(),
            log_file=self.project.log_path('atl_gen_init_db'),
            detached=True,
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(self._on_finished)
        self.worker.start()

    def on_shown(self) -> None:
        self._try_reconnect_detached()
        self.structures_panel.refresh()
        self.table_panel.refresh()
        self.boundary_panel.refresh()
        self._manage_panel.refresh()
        self._update_success_banner()

    def set_theme(self, theme_name: str) -> None:
        super().set_theme(theme_name)
        # Output panels are not covered by the base class; the boundary
        # panel's side-panel strip carries themed icons of its own.
        if hasattr(self, 'boundary_panel'):
            self.boundary_panel.set_theme(theme_name)

    def _try_reconnect_detached(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        logs_dir = self.project.dir / 'logs'
        info = find_detached_process(logs_dir, 'atl_gen_init_db')
        if info is None:
            return
        self._log(f'🔄 Found running process (PID {info["pid"]}), reconnecting…')
        self._set_running(self.run_button, self.cancel_button)
        self.worker = DetachedProcessMonitor(
            pid=info['pid'],
            log_file=info['log_file'],
            pid_file=info['pid_file'],
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(self._on_finished)
        self.worker.start()

    def _update_success_banner(self) -> None:
        state = self.project.workflow_state()
        init_done = state['stages']['init_db']['status'] in ('done', 'partial')
        if init_done and not self.success_banner.isVisible():
            pass

    def _go_to_outputs(self) -> None:
        self.tabs.setCurrentWidget(self.structures_panel)

    def _on_db_deleted(self) -> None:
        self._log('🗑 Database deleted.')
        self.structures_panel.refresh()
        self.table_panel.refresh()
        self.boundary_panel.refresh()
        self.success_banner.hide()
        self.workflow_state_changed.emit()

    def _on_finished(self, return_code: int) -> None:
        self._log(f'\n✅ atl_gen_init_db finished with exit code: {return_code}\n')
        self._set_idle(self.run_button, self.cancel_button, 'Run Database Generation')
        self._notification('Initial Database Generation', return_code == 0)
        if return_code == 0:
            try:
                count = self.project.refresh_structures_index()
            except Exception as exc:
                self._log(f'⚠ Failed to index structures: {exc}')
                return
            self._log(f'📦 Indexed {count} structures into the project.')
            self.structures_panel.refresh()
            self.table_panel.refresh()
            self.boundary_panel.refresh()
            self.success_banner.show()
            self.workflow_state_changed.emit()
