"""The `jobs.stage` migration (staged progress reporting for read-along
generation) applies base -> head on a fresh temp SQLite, additively.

Mirrors `tests/test_job_kind_migration.py`'s two-layer pattern: constructing
`DatabaseService(tmp_path)` runs Alembic's full `command.upgrade(cfg, "head")`
chain from base (the safe way to exercise a migration in this repo --
CLAUDE.md warns against running Alembic from the repo root, which would
rewrite the URL onto the live `/data` bind mount), and a second class
exercises the migration module's own upgrade/downgrade in isolation against a
bare `jobs` table.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import sqlalchemy as sa

from src.db.database_service import DatabaseService


class TestJobStageMigrationAppliesToHead(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "job_stage_migration.db")
        self.db_service = DatabaseService(self.db_path)

    def tearDown(self) -> None:
        self.db_service.db_manager.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_column_exists_and_is_nullable(self) -> None:
        inspector = sa.inspect(self.db_service.db_manager.engine)
        columns = {c["name"]: c for c in inspector.get_columns("jobs")}

        self.assertIn("stage", columns)
        self.assertTrue(columns["stage"]["nullable"])

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

        self.assertEqual(mig_cols.get("stage"), built_cols.get("stage"))

    def test_preexisting_job_row_backfills_to_null_stage(self) -> None:
        """A row inserted before the `stage` column existed must come back as
        None, not some placeholder -- there is truly no stage recorded for
        it (it predates staged progress reporting entirely)."""
        with self.db_service.db_manager.engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO books (abs_id, abs_title) VALUES ('mig-book', 'Migration Book')"
            ))
            conn.execute(sa.text(
                "INSERT INTO jobs (abs_id, last_attempt, retry_count, progress, kind) "
                "VALUES ('mig-book', 123.0, 0, 0.5, 'alignment')"
            ))

        job = self.db_service.get_latest_job("mig-book")
        self.assertIsNone(job.stage)

    def test_saved_job_with_stage_round_trips_through_the_full_stack(self) -> None:
        from src.db.models import Book, Job, JOB_KIND_READALONG

        self.db_service.save_book(Book(abs_id="stage-check", abs_title="Stage Check", status="active"))
        self.db_service.save_job(Job(
            abs_id="stage-check", last_attempt=1.0, retry_count=0, progress=0.2,
            kind=JOB_KIND_READALONG, stage="transcoding_audio",
        ))

        job = self.db_service.get_latest_job("stage-check", kind=JOB_KIND_READALONG)
        self.assertEqual(job.stage, "transcoding_audio")


class TestJobStageMigrationModuleDirectly(unittest.TestCase):
    """Exercises the migration's own upgrade/downgrade against a pre-existing
    bare `jobs` table, isolated from the rest of the chain."""

    def setUp(self) -> None:
        import importlib.util

        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        migration_path = (
            Path(__file__).resolve().parents[1]
            / "alembic/versions/ccf04b7f7b3e_add_job_stage.py"
        )
        spec = importlib.util.spec_from_file_location("job_stage_migration", migration_path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

        self.engine = sa.create_engine("sqlite:///:memory:")
        with self.engine.begin() as conn:
            conn.execute(sa.text(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, abs_id VARCHAR(255), "
                "last_attempt FLOAT, retry_count INTEGER, last_error TEXT, progress FLOAT, "
                "kind VARCHAR(50))"
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

    def test_upgrade_adds_nullable_column(self) -> None:
        self._run(self.mod.upgrade)
        columns = self._columns()
        self.assertIn("stage", columns)
        self.assertTrue(columns["stage"]["nullable"])

    def test_upgrade_leaves_existing_rows_null(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO jobs (abs_id, last_attempt, retry_count, progress, kind) "
                "VALUES ('pre-existing', 1.0, 0, 0.0, 'alignment')"
            ))
        self._run(self.mod.upgrade)
        with self.engine.begin() as conn:
            value = conn.execute(sa.text(
                "SELECT stage FROM jobs WHERE abs_id = 'pre-existing'"
            )).scalar()
        self.assertIsNone(value)

    def test_upgrade_is_idempotent(self) -> None:
        self._run(self.mod.upgrade)
        self._run(self.mod.upgrade)
        columns = self._columns()
        self.assertIn("stage", columns)

    def test_downgrade_drops_column(self) -> None:
        self._run(self.mod.upgrade)
        self._run(self.mod.downgrade)
        columns = self._columns()
        self.assertNotIn("stage", columns)

    def test_downgrade_is_idempotent(self) -> None:
        self._run(self.mod.upgrade)
        self._run(self.mod.downgrade)
        self._run(self.mod.downgrade)
        columns = self._columns()
        self.assertNotIn("stage", columns)


if __name__ == "__main__":
    unittest.main()
