"""The `jobs.kind` migration (Finding 2, P1) applies base -> head on a fresh
temp SQLite, additively.

Mirrors `tests/test_readalong_epub_requested_migration.py`'s two-layer
pattern: constructing `DatabaseService(tmp_path)` runs Alembic's full
`command.upgrade(cfg, "head")` chain from base (the safe way to exercise a
migration in this repo -- CLAUDE.md warns against running Alembic from the
repo root, which would rewrite the URL onto the live `/data` bind mount), and
a second class exercises the migration module's own upgrade/downgrade in
isolation against a bare `jobs` table.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import sqlalchemy as sa

from src.db.database_service import DatabaseService


class TestJobKindMigrationAppliesToHead(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "job_kind_migration.db")
        self.db_service = DatabaseService(self.db_path)

    def tearDown(self) -> None:
        self.db_service.db_manager.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_column_exists_not_nullable_defaults_to_alignment(self) -> None:
        inspector = sa.inspect(self.db_service.db_manager.engine)
        columns = {c["name"]: c for c in inspector.get_columns("jobs")}

        self.assertIn("kind", columns)
        self.assertFalse(columns["kind"]["nullable"])

    def test_orm_model_matches_the_migrated_schema(self) -> None:
        """A model that drifts from the migration is exactly the class of bug
        this test catches: a freshly created DB (Base.metadata.create_all)
        and a migrated one would silently disagree on column shape."""
        from src.db.models import Base

        migrated = sa.inspect(self.db_service.db_manager.engine)
        mig_cols = {c["name"]: str(c["type"]) for c in migrated.get_columns("jobs")}

        model_engine = sa.create_engine("sqlite:///:memory:")
        try:
            Base.metadata.create_all(model_engine)
            built = sa.inspect(model_engine)
            built_cols = {c["name"]: str(c["type"]) for c in built.get_columns("jobs")}
        finally:
            model_engine.dispose()

        self.assertEqual(mig_cols.get("kind"), built_cols.get("kind"))

    def test_preexisting_job_row_backfills_to_alignment_kind(self) -> None:
        """A row inserted before the `kind` column existed must come back as
        the pre-existing undifferentiated kind, not NULL or empty -- it truly
        was alignment-repair/retry tracking, the only kind that existed."""
        with self.db_service.db_manager.engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO books (abs_id, abs_title) VALUES ('mig-book', 'Migration Book')"
            ))
            conn.execute(sa.text(
                "INSERT INTO jobs (abs_id, last_attempt, retry_count, progress) "
                "VALUES ('mig-book', 123.0, 0, 0.5)"
            ))

        from src.db.models import JOB_KIND_ALIGNMENT

        job = self.db_service.get_latest_job("mig-book")
        self.assertEqual(job.kind, JOB_KIND_ALIGNMENT)

    def test_saved_job_defaults_to_alignment_kind_through_the_full_stack(self) -> None:
        from src.db.models import Book, Job, JOB_KIND_ALIGNMENT

        self.db_service.save_book(Book(abs_id="kind-check", abs_title="Kind Check", status="active"))
        self.db_service.save_job(Job(abs_id="kind-check", last_attempt=1.0, retry_count=0, progress=0.0))

        job = self.db_service.get_latest_job("kind-check")
        self.assertEqual(job.kind, JOB_KIND_ALIGNMENT)


class TestJobKindMigrationModuleDirectly(unittest.TestCase):
    """Exercises the migration's own upgrade/downgrade against a pre-existing
    bare `jobs` table, isolated from the rest of the chain."""

    def setUp(self) -> None:
        import importlib.util

        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        migration_path = (
            Path(__file__).resolve().parents[1]
            / "alembic/versions/3f6c8a1d9e42_add_job_kind.py"
        )
        spec = importlib.util.spec_from_file_location("job_kind_migration", migration_path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

        self.engine = sa.create_engine("sqlite:///:memory:")
        with self.engine.begin() as conn:
            conn.execute(sa.text(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, abs_id VARCHAR(255), "
                "last_attempt FLOAT, retry_count INTEGER, last_error TEXT, progress FLOAT)"
            ))
        self._MigrationContext = MigrationContext
        self._Operations = Operations

    def tearDown(self) -> None:
        self.engine.dispose()

    def _run(self, fn) -> None:
        with self.engine.begin() as conn:
            ctx = self._MigrationContext.configure(conn)
            operations = self._Operations(ctx)
            old_op = self.mod.op
            try:
                self.mod.op = operations
                fn()
            finally:
                self.mod.op = old_op

    def _columns(self):
        return {c["name"]: c for c in sa.inspect(self.engine).get_columns("jobs")}

    def test_upgrade_adds_not_nullable_column_with_default(self) -> None:
        self._run(self.mod.upgrade)
        columns = self._columns()
        self.assertIn("kind", columns)
        self.assertFalse(columns["kind"]["nullable"])

    def test_upgrade_backfills_existing_rows_to_alignment(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO jobs (abs_id, last_attempt, retry_count, progress) "
                "VALUES ('pre-existing', 1.0, 0, 0.0)"
            ))
        self._run(self.mod.upgrade)
        with self.engine.begin() as conn:
            value = conn.execute(sa.text(
                "SELECT kind FROM jobs WHERE abs_id = 'pre-existing'"
            )).scalar()
        self.assertEqual(value, "alignment")

    def test_upgrade_is_idempotent(self) -> None:
        self._run(self.mod.upgrade)
        self._run(self.mod.upgrade)
        columns = self._columns()
        self.assertIn("kind", columns)

    def test_downgrade_drops_column(self) -> None:
        self._run(self.mod.upgrade)
        self._run(self.mod.downgrade)
        columns = self._columns()
        self.assertNotIn("kind", columns)

    def test_downgrade_is_idempotent(self) -> None:
        self._run(self.mod.upgrade)
        self._run(self.mod.downgrade)
        self._run(self.mod.downgrade)
        columns = self._columns()
        self.assertNotIn("kind", columns)


if __name__ == "__main__":
    unittest.main()
