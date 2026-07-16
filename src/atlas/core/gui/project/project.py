"""The ``Project`` class, runtime API for an ATLAS GUI project bundle.

Layout
------
The project bundle consists of an SQLite file and a canonical artifact directory:

* ``<parent>/<name>.atlasproj`` - SQLite file with index, settings history, summaries
* ``<parent>/<name>.atlas/`` - artifact directory containing:
  - ``database_generation_settings.toml``
  - ``active_learning_settings.toml``
  - ``dft_settings.toml``
  - ``databases/<database_name>.xz``
  - ``models/<al_run_id>/...``
  - ``reports/<al_run_id>/...``
  - ``logs/<command>.<ts>.log``

Configs live at the top of the ``.atlas/`` directory so that CLI tools
invoked with that directory as CWD find them via their existing default
filename lookups (e.g. ``cli_active_learning.py`` looks for
``active_learning_settings.toml``).
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path
from typing import Any

import tomli
import tomli_w

from atlas.core import ATL_DATA_DIR
from atlas.core.gui.project import migrations, schema

ATLASPROJ_SUFFIX = '.atlasproj'
ATLASDIR_SUFFIX = '.atlas'

# Mapping from schema section key -> on-disk canonical TOML filename.
# The non-trivial mappings preserve CLI default-filename lookup behaviour.
CONFIG_FILENAMES: dict[str, str] = {
    'database_generation': 'database_generation_settings.toml',
    'dft': 'dft_settings.toml',
    'dft_benchmark': 'dft_benchmark_settings.toml',
    'active_learning': 'active_learning_settings.toml',
}

# Subdirectories created inside the .atlas/ sibling.
SUBDIRS = ('databases', 'models', 'reports', 'logs')

# Source templates copied into a freshly created project.
TEMPLATE_DIR = Path(ATL_DATA_DIR) / 'input_files'


class ProjectError(Exception):
    """Raised on project create / open / migration failures."""


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    # NB: iterating a sqlite3.Row yields values, not column names, explicit
    # `.keys()` is required here.  Ruff's SIM118 autofix is wrong for Row.
    return {k: row[k] for k in row.keys()}  # noqa: SIM118


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S')


class Project:
    """In-memory handle for an open ATLAS project."""

    def __init__(self, path: Path, conn: sqlite3.Connection):
        self.path: Path = Path(path).resolve()
        self.conn: sqlite3.Connection = conn
        self.conn.row_factory = sqlite3.Row

    # =============================================================== create/open

    @classmethod
    def create(cls, parent_dir: Path | str, name: str) -> Project:
        """Create a new project bundle. Raises if it would overwrite existing data."""
        parent = Path(parent_dir).resolve()
        if not parent.is_dir():
            raise ProjectError(f'Parent directory does not exist: {parent}')

        if not name or any(ch in name for ch in r'\/:*?"<>|'):
            raise ProjectError(f'Invalid project name: {name!r}')

        atlasproj = parent / f'{name}{ATLASPROJ_SUFFIX}'
        atlasdir = parent / f'{name}{ATLASDIR_SUFFIX}'
        if atlasproj.exists() or atlasdir.exists():
            raise ProjectError(f'Project already exists at {atlasproj} or {atlasdir}')

        atlasdir.mkdir()
        for sub in SUBDIRS:
            (atlasdir / sub).mkdir()

        conn = sqlite3.connect(str(atlasproj))
        schema.initialise(conn)

        meta = {
            'name': name,
            'created_at': _now_iso(),
            'last_opened_at': _now_iso(),
            'schema_version': str(schema.SCHEMA_VERSION),
        }
        conn.executemany(
            'INSERT OR REPLACE INTO project_meta (key, value) VALUES (?, ?)',
            list(meta.items()),
        )
        conn.commit()

        project = cls(atlasproj, conn)
        project._seed_templates()
        return project

    @classmethod
    def open(cls, atlasproj_path: Path | str) -> Project:
        """Open an existing project. Applies any pending migrations."""
        path = Path(atlasproj_path).resolve()
        if path.suffix != ATLASPROJ_SUFFIX or not path.exists():
            raise ProjectError(f'Not an ATLAS project file: {path}')

        # Sibling dir must also exist for the project to be usable.
        sibling = path.with_suffix(ATLASDIR_SUFFIX)
        if not sibling.is_dir():
            raise ProjectError(f'Missing project directory: {sibling}')

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        # Make sure all tables exist (handles half-initialised files).
        for statement in schema.DDL:
            conn.execute(statement)
        conn.commit()

        migrations.apply_migrations(conn)

        project = cls(path, conn)
        project.set_meta('last_opened_at', _now_iso())
        return project

    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self.conn.close()

    # ============================================================== filesystem

    @property
    def dir(self) -> Path:
        return self.path.with_suffix(ATLASDIR_SUFFIX)

    def cwd(self) -> Path:
        """Working directory for CLI subprocesses launched by the GUI."""
        return self.dir

    def config_path(self, section: str) -> Path:
        filename = CONFIG_FILENAMES.get(section, f'{section}.toml')
        return self.dir / filename

    def log_path(self, command: str) -> Path:
        ts = time.strftime('%Y%m%dT%H%M%S')
        return self.dir / 'logs' / f'{command}.{ts}.log'

    def _seed_templates(self) -> None:
        """Seed each section's config from the packaged CLI templates."""
        for section, filename in CONFIG_FILENAMES.items():
            source = TEMPLATE_DIR / filename
            if not source.exists():
                continue
            try:
                toml_text = source.read_text(encoding='utf-8')
            except OSError:
                continue
            # save_config_snapshot rewrites (e.g. forces database_path),
            # records the snapshot, and writes the canonical on-disk file.
            self.save_config_snapshot(
                section,
                toml_text,
                label='template',
                activate=True,
            )

    # =========================================================== meta key/value

    def meta(self, key: str, default: str | None = None) -> str | None:
        cur = self.conn.execute('SELECT value FROM project_meta WHERE key = ?', (key,))
        row = cur.fetchone()
        return row['value'] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            'INSERT OR REPLACE INTO project_meta (key, value) VALUES (?, ?)',
            (key, value),
        )
        self.conn.commit()

    @property
    def name(self) -> str:
        return self.meta('name') or self.path.stem

    # =============================================================== configs

    def save_config_snapshot(
        self,
        section: str,
        toml_text: str,
        label: str | None = None,
        activate: bool = True,
    ) -> int:
        """Insert a snapshot row; optionally activate it and mirror to disk."""
        toml_text = self._project_rewrite_toml(section, toml_text)

        created_at = _now_iso()
        if label is None:
            label = f'auto-{created_at}'

        with self.conn:
            if activate:
                self.conn.execute(
                    'UPDATE config_snapshots SET is_active = 0 WHERE section = ?',
                    (section,),
                )
            cur = self.conn.execute(
                """
                INSERT INTO config_snapshots
                    (section, label, toml, created_at, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (section, label, toml_text, created_at, 1 if activate else 0),
            )
            snapshot_id = cur.lastrowid

        if activate:
            path = self.config_path(section)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(toml_text, encoding='utf-8')

        return snapshot_id

    def active_config(self, section: str) -> tuple[int, str] | None:
        """Return ``(snapshot_id, toml)`` for the active snapshot of ``section``."""
        cur = self.conn.execute(
            """
            SELECT id, toml FROM config_snapshots
            WHERE section = ? AND is_active = 1
            ORDER BY id DESC LIMIT 1
            """,
            (section,),
        )
        row = cur.fetchone()
        return (row['id'], row['toml']) if row else None

    def list_config_snapshots(self, section: str) -> list[dict]:
        cur = self.conn.execute(
            """
            SELECT id, section, label, created_at, is_active
            FROM config_snapshots
            WHERE section = ?
            ORDER BY id DESC
            """,
            (section,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    def _project_rewrite_toml(self, section: str, toml_text: str) -> str:
        """Apply project-aware rewrites before storing a TOML.

        For ``database_generation`` we pin ``database.database_path`` so the
        ``atl_gen_init_db`` output lands under the project's ``databases/``
        subdirectory regardless of what the user typed.
        """
        if section != 'database_generation':
            return toml_text
        try:
            data = tomli.loads(toml_text)
        except tomli.TOMLDecodeError:
            return toml_text
        database = data.setdefault('database', {})
        if isinstance(database, dict):
            database['database_path'] = 'databases'
        return tomli_w.dumps(data)

    # ============================================================ initial DB

    def init_db_name(self) -> str:
        snapshot = self.active_config('database_generation')
        if snapshot is None:
            return 'init_db'
        try:
            data = tomli.loads(snapshot[1])
        except tomli.TOMLDecodeError:
            return 'init_db'
        return (data.get('database') or {}).get('database_name') or 'init_db'

    def init_db_path(self) -> Path:
        exact = self.dir / 'databases' / f'{self.init_db_name()}.xz'
        if exact.exists():
            return exact
        # ATLAS appends a timestamp to the database name; find the latest match
        db_dir = self.dir / 'databases'
        if db_dir.is_dir():
            prefix = self.init_db_name()
            candidates = sorted(
                db_dir.glob(f'{prefix}*.xz'),
                key=lambda p: p.stat().st_mtime,
            )
            if candidates:
                return candidates[-1]
            # No prefix match, fall back to newest .xz
            all_xz = sorted(
                db_dir.glob('*.xz'),
                key=lambda p: p.stat().st_mtime,
            )
            if all_xz:
                return all_xz[-1]
        return exact

    def clear_structures_index(self) -> None:
        """Remove all rows from the structures table."""
        with self.conn:
            self.conn.execute('DELETE FROM structures')
        self._structures_index_mtime = None

    def refresh_structures_index(self, force: bool = False) -> int:
        """Walk the on-disk initial DB and upsert summary rows. Returns count.

        Skips the expensive .xz decompression when the file hasn't changed
        since the last successful index (based on mtime).  Pass *force=True*
        to re-scan unconditionally.
        """
        path = self.init_db_path()
        if not path.exists():
            return 0

        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return 0

        last_mtime = getattr(self, '_structures_index_mtime', None)
        if not force and last_mtime is not None and last_mtime == current_mtime:
            cur = self.conn.execute('SELECT COUNT(*) FROM structures')
            return cur.fetchone()[0]

        if not force:
            cur = self.conn.execute('SELECT COUNT(*) FROM structures')
            if cur.fetchone()[0] > 0 and last_mtime is None:
                self._structures_index_mtime = current_mtime
                return self.conn.execute('SELECT COUNT(*) FROM structures').fetchone()[
                    0
                ]

        from atlas.core.initial_db import InitialDatabase

        loaded = InitialDatabase.load_database(path)
        df = getattr(loaded, 'df', loaded)
        if df is None or getattr(df, 'empty', True):
            return 0

        rows = []
        now = _now_iso()
        for _, item in df.iterrows():
            atl_id = _coerce_str(
                item.get('atl_id') or item.get('unique_id') or item.get('aiida_uuid')
            )
            if not atl_id:
                continue
            # Compute n_atoms from the structure object
            struct = item.get('structure')
            n_atoms = len(struct.species) if struct is not None else None

            # Extract phase name (column stores Phase objects, not strings)
            phase_val = item.get('phase')
            phase_name = (
                phase_val.name if hasattr(phase_val, 'name') else _coerce_str(phase_val)
            )

            rows.append(
                (
                    atl_id,
                    _coerce_str(item.get('formula')),
                    _coerce_int(n_atoms),
                    phase_name,
                    _derive_struct_type(item),
                    _derive_modifications_json(item),
                    1 if bool(item.get('calc_performed')) else 0,
                    _coerce_float(item.get('calc_energy')),
                    now,
                )
            )

        if not rows:
            return 0

        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO structures
                    (atl_id, formula, n_atoms, phase, struct_type,
                     modifications, calc_performed, calc_energy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(atl_id) DO UPDATE SET
                    formula        = excluded.formula,
                    n_atoms        = excluded.n_atoms,
                    phase          = excluded.phase,
                    struct_type    = excluded.struct_type,
                    modifications  = excluded.modifications,
                    calc_performed = excluded.calc_performed,
                    calc_energy    = excluded.calc_energy
                """,
                rows,
            )
        self._structures_index_mtime = current_mtime
        return len(rows)

    def list_structures(self, limit: int | None = None) -> list[dict]:
        query = 'SELECT * FROM structures ORDER BY created_at'
        if limit is not None:
            query += f' LIMIT {int(limit)}'
        cur = self.conn.execute(query)
        return [_row_to_dict(r) for r in cur.fetchall()]

    def structure_counts(self) -> dict[str, int]:
        cur = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN calc_performed = 1 THEN 1 ELSE 0 END) AS labelled
            FROM structures
            """
        )
        row = cur.fetchone()
        total = int(row['total'] or 0)
        labelled = int(row['labelled'] or 0)
        return {'total': total, 'labelled': labelled, 'unlabelled': total - labelled}

    def structure_breakdown(self, column: str) -> list[tuple[str, int]]:
        """Group structures by *column* and return ``[(value, count), …]``."""
        allowed = {'phase', 'struct_type', 'formula'}
        if column not in allowed:
            raise ValueError(f'column must be one of {allowed}')
        cur = self.conn.execute(
            f'SELECT {column}, COUNT(*) AS cnt FROM structures '  # noqa: S608
            f'GROUP BY {column} ORDER BY cnt DESC',
        )
        return [(r[0] or 'unknown', int(r[1])) for r in cur.fetchall()]

    def modifications_breakdown(self) -> list[tuple[str, int]]:
        """Count how many structures have each modification flag."""
        rows = self.conn.execute(
            'SELECT modifications FROM structures WHERE modifications IS NOT NULL',
        ).fetchall()
        import json

        counts: dict[str, int] = {}
        for (raw,) in rows:
            for mod in json.loads(raw):
                counts[mod] = counts.get(mod, 0) + 1
        return sorted(counts.items(), key=lambda x: -x[1])

    def n_atoms_list(self) -> list[int]:
        """Return all non-null ``n_atoms`` values for histogram plotting."""
        cur = self.conn.execute(
            'SELECT n_atoms FROM structures WHERE n_atoms IS NOT NULL',
        )
        return [int(r[0]) for r in cur.fetchall()]

    def remove_structures_from_index(self, atl_ids: list[str]) -> None:
        """Remove structures from the SQLite index (main-thread only)."""
        if not atl_ids:
            return
        placeholders = ','.join('?' for _ in atl_ids)
        with self.conn:
            self.conn.execute(
                f'UPDATE dft_runs SET atl_id = NULL WHERE atl_id IN ({placeholders})',  # noqa: S608
                atl_ids,
            )
            self.conn.execute(
                f'DELETE FROM structures WHERE atl_id IN ({placeholders})',  # noqa: S608
                atl_ids,
            )
        self._structures_index_mtime = None

    # =============================================================== AL runs

    def record_al_submission(
        self,
        base_workchain_pk: int | None,
        config_snapshot_id: int | None,
        status: str = 'submitted',
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO al_runs
                    (base_workchain_pk, config_snapshot_id, started_at, status)
                VALUES (?, ?, ?, ?)
                """,
                (base_workchain_pk, config_snapshot_id, _now_iso(), status),
            )
            return cur.lastrowid

    def list_al_runs(self) -> list[dict]:
        cur = self.conn.execute('SELECT * FROM al_runs ORDER BY id DESC')
        return [_row_to_dict(r) for r in cur.fetchall()]

    def update_al_run_status(
        self,
        run_id: int,
        status: str,
        finished: bool = False,
    ) -> None:
        if finished:
            with self.conn:
                self.conn.execute(
                    'UPDATE al_runs SET status = ?, finished_at = ? WHERE id = ?',
                    (status, _now_iso(), run_id),
                )
        else:
            with self.conn:
                self.conn.execute(
                    'UPDATE al_runs SET status = ? WHERE id = ?',
                    (status, run_id),
                )

    def al_run_counts(self) -> dict[str, int]:
        cur = self.conn.execute(
            'SELECT status, COUNT(*) AS n FROM al_runs GROUP BY status'
        )
        return {row['status'] or 'unknown': int(row['n']) for row in cur.fetchall()}

    def upsert_al_iterations(
        self,
        run_id: int,
        iterations: list[dict],
    ) -> int:
        """Insert or update iteration rows parsed from a log file.

        Each dict in *iterations* should have keys matching
        ``al_iterations`` columns (``iteration``, ``db_size``,
        ``rmse_e``, ``rmse_f``, …).  Returns the number of rows upserted.
        """
        if not iterations:
            return 0
        now = _now_iso()
        rows = []
        for it in iterations:
            rows.append(
                (
                    run_id,
                    it.get('iteration', 0),
                    _coerce_int(it.get('db_size')),
                    _coerce_float(it.get('rmse_e')),
                    _coerce_float(it.get('rmse_f')),
                    _coerce_float(it.get('mae_e')),
                    _coerce_float(it.get('mae_f')),
                    it.get('model_path'),
                    it.get('latent_space_plot_path'),
                    now,
                )
            )
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO al_iterations
                    (run_id, iteration, db_size, rmse_e, rmse_f,
                     mae_e, mae_f, model_path, latent_space_plot_path,
                     refreshed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, iteration) DO UPDATE SET
                    db_size    = excluded.db_size,
                    rmse_e     = excluded.rmse_e,
                    rmse_f     = excluded.rmse_f,
                    mae_e      = excluded.mae_e,
                    mae_f      = excluded.mae_f,
                    model_path = excluded.model_path,
                    latent_space_plot_path = excluded.latent_space_plot_path,
                    refreshed_at = excluded.refreshed_at
                """,
                rows,
            )
        return len(rows)

    def list_al_iterations(self, run_id: int) -> list[dict]:
        cur = self.conn.execute(
            'SELECT * FROM al_iterations WHERE run_id = ? ORDER BY iteration',
            (run_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    def latest_al_run_id(self) -> int | None:
        cur = self.conn.execute(
            'SELECT id FROM al_runs ORDER BY id DESC LIMIT 1',
        )
        row = cur.fetchone()
        return row['id'] if row else None

    def refresh_al_run(self, run_id: int) -> None:
        """Query AiiDA for a single AL workchain and update its status."""
        from atlas.core.gui.project.aiida_sync import sync_al_runs

        sync_al_runs(self)

    # ============================================================== DFT runs

    def record_dft_submission(
        self,
        calc_uuid: str,
        aiida_pk: int | None,
        atl_id: str | None,
        queue: str | None,
        status: str = 'submitted',
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO dft_runs
                    (calc_uuid, aiida_pk, atl_id, queue, status, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (calc_uuid, aiida_pk, atl_id, queue, status, _now_iso()),
            )

    def dft_run_counts(self) -> dict[str, int]:
        cur = self.conn.execute(
            'SELECT status, COUNT(*) AS n FROM dft_runs GROUP BY status'
        )
        return {row['status'] or 'unknown': int(row['n']) for row in cur.fetchall()}

    def list_dft_runs(self) -> list[dict]:
        cur = self.conn.execute('SELECT * FROM dft_runs ORDER BY submitted_at DESC')
        return [_row_to_dict(r) for r in cur.fetchall()]

    def refresh_dft_runs(self) -> None:
        """Query AiiDA for tracked DFT calculations and update statuses."""
        from atlas.core.gui.project.aiida_sync import sync_dft_runs

        sync_dft_runs(self)

    # =========================================================== workflow state

    def workflow_state(self) -> dict:
        """Return a per-stage workflow status used by the Overview tracker.

        Stages: ``init_db``, ``dft``, ``al``, ``reports``. Each carries a
        ``status`` of ``empty`` / ``partial`` / ``running`` / ``done`` and a
        short human-readable ``metric``. ``next_recommended`` names the first
        stage that is not yet ``done`` (or ``None`` if everything is done).
        """
        counts = self.structure_counts()
        total = counts['total']
        labelled = counts['labelled']
        has_init_db_file = self.init_db_path().exists()

        if total > 0:
            init_status = 'done'
            init_metric = f'{total:,} structures indexed'
        elif has_init_db_file:
            # .xz exists but the index hasn't been refreshed yet.
            init_status = 'partial'
            init_metric = 'Database file present, refresh index to see counts'
        else:
            init_status = 'empty'
            init_metric = 'No database generated yet'

        if total == 0:
            dft_status = 'empty'
            dft_metric = 'Needs an initial database'
        elif labelled == 0:
            dft_status = 'empty'
            dft_metric = f'0 / {total:,} structures labelled'
        elif labelled < total:
            dft_status = 'partial'
            dft_metric = f'{labelled:,} / {total:,} structures labelled'
        else:
            dft_status = 'done'
            dft_metric = f'All {total:,} structures labelled'

        al_runs = self.list_al_runs()
        if not al_runs:
            al_status = 'empty'
            al_metric = 'No runs started yet'
        else:
            active = [
                r
                for r in al_runs
                if (r.get('status') or '') in ('submitted', 'running')
            ]
            completed = [r for r in al_runs if (r.get('status') or '') == 'completed']
            if active:
                al_status = 'running'
                al_metric = f'{len(active)} active / {len(al_runs)} total'
            elif completed:
                al_status = 'done'
                al_metric = f'{len(completed)} completed / {len(al_runs)} total'
            else:
                al_status = 'partial'
                al_metric = f'{len(al_runs)} submitted'

        # Reports follow AL state.
        if al_status == 'done':
            reports_status = 'done'
            reports_metric = 'Results available'
        elif al_status == 'running':
            reports_status = 'partial'
            reports_metric = 'Will populate as AL iterations finish'
        else:
            reports_status = 'empty'
            reports_metric = 'No completed AL runs yet'

        stages = {
            'init_db': {'status': init_status, 'metric': init_metric},
            'dft': {'status': dft_status, 'metric': dft_metric},
            'al': {'status': al_status, 'metric': al_metric},
            'reports': {'status': reports_status, 'metric': reports_metric},
        }

        next_recommended = None
        for key in ('init_db', 'dft', 'al', 'reports'):
            if stages[key]['status'] in ('empty', 'partial'):
                next_recommended = key
                break

        return {'stages': stages, 'next_recommended': next_recommended}


# ============================================================ helpers


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive
        return None
    return text or None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_struct_type(item) -> str | None:
    explicit = _coerce_str(item.get('struct_type'))
    if explicit:
        return explicit
    for flag in ('bulk', 'surface', 'cluster', 'isolated_atom'):
        val = item.get(flag)
        try:
            if val is not None and val == val and bool(val):
                return flag
        except (TypeError, ValueError):
            continue
    return None


def _derive_modifications_json(item) -> str:
    mods = [
        flag
        for flag in ('perturb', 'deformation', 'vacancy', 'replacement')
        if bool(item.get(flag))
    ]
    import json

    return json.dumps(mods)
