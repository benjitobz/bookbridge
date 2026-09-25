"""The
`books.readalong_epub_requested` migration applies base -> head on a fresh
temp SQLite, additively.

Mirrors `tests/test_alignment_quality_migration.py`'s two-layer pattern:
constructing `DatabaseService(tmp_path)` runs Alembic's full
`command.upgrade(cfg, "head")` chain from base (the safe way to exercise a
migration in this repo -- CLAUDE.md warns against running Alembic from the
repo root, which would rewrite the URL onto the live `/data` bind mount), and
a second class exercises the migration module's own upgrade/downgrade in
isolation against a bare `books` table.
"""

import unittest

import sqlalchemy as sa

from src.db.database_service import DatabaseService


class TestReadalongEpubRequestedMigrationAppliesToHead(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "readalong_intent_migration.db")
        self.db_service = DatabaseService(self.db_path)

    def tearDown(self):
        self.db_service.db_manager.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_column_exists_not_nullable_defaults_false(self):
        inspector = sa.inspect(self.db_service.db_manager.engine)
        columns = {c["name"]: c for c in inspector.get_columns("books")}

        self.assertIn("readalong_epub_requested", columns)
        column = columns["readalong_epub_requested"]
        self.assertFalse(column["nullable"])

    def test_orm_model_matches_the_migrated_schema(self):
        """A model that drifts from the migration is exactly the class of bug
        this test catches: a freshly created DB (Base.metadata.create_all) and
        a migrated one would silently disagree on column shape."""
        from src.db.models import Base

        migrated = sa.inspect(self.db_service.db_manager.engine)
        mig_cols = {c["name"]: str(c["type"]) for c in migrated.get_columns("books")}

        model_engine = sa.create_engine("sqlite:///:memory:")
        try:
            Base.metadata.create_all(model_engine)
            built = sa.inspect(model_engine)
            built_cols = {c["name"]: str(c["type"]) for c in built.get_columns("books")}
        finally:
            model_engine.dispose()

        self.assertEqual(
            mig_cols.get("readalong_epub_requested"),
            built_cols.get("readalong_epub_requested"),
        )

    def test_saved_book_defaults_to_false_through_the_full_stack(self):
        from src.db.models import Book

        self.db_service.save_book(Book(abs_id="mig-check", abs_title="Migration Check", status="active"))
        book = self.db_service.get_book("mig-check")
        self.assertFalse(book.readalong_epub_requested)


class TestReadalongEpubRequestedMigrationModuleDirectly(unittest.TestCase):
    """Exercises the migration's own upgrade/downgrade against a pre-existing
    bare `books` table, isolated from the rest of the chain."""

    def setUp(self):
        import importlib.util
        from pathlib import Path

        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        migration_path = (
            Path(__file__).resolve().parents[1]
            / "alembic/versions/8eef09f34e4a_add_readalong_epub_requested.py"
        )
        spec = importlib.util.spec_from_file_location("readalong_intent_migration", migration_path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

        self.engine = sa.create_engine("sqlite:///:memory:")
        with self.engine.begin() as conn:
            conn.execute(sa.text(
                "CREATE TABLE books (abs_id VARCHAR(255) PRIMARY KEY, abs_title VARCHAR(500))"
            ))
        self._MigrationContext = MigrationContext
        self._Operations = Operations

    def tearDown(self):
        self.engine.dispose()

    def _run(self, fn):
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
        return {c["name"]: c for c in sa.inspect(self.engine).get_columns("books")}

    def test_upgrade_adds_not_nullable_column_with_default(self):
        self._run(self.mod.upgrade)
        columns = self._columns()
        self.assertIn("readalong_epub_requested", columns)
        self.assertFalse(columns["readalong_epub_requested"]["nullable"])

    def test_upgrade_backfills_existing_rows_to_false(self):
        with self.engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO books (abs_id, abs_title) VALUES ('pre-existing', 'Pre-existing Book')"
            ))
        self._run(self.mod.upgrade)
        with self.engine.begin() as conn:
            value = conn.execute(sa.text(
                "SELECT readalong_epub_requested FROM books WHERE abs_id = 'pre-existing'"
            )).scalar()
        self.assertEqual(value, 0)

    def test_upgrade_is_idempotent(self):
        self._run(self.mod.upgrade)
        self._run(self.mod.upgrade)
        columns = self._columns()
        self.assertIn("readalong_epub_requested", columns)

    def test_downgrade_drops_column(self):
        self._run(self.mod.upgrade)
        self._run(self.mod.downgrade)
        columns = self._columns()
        self.assertNotIn("readalong_epub_requested", columns)

    def test_downgrade_is_idempotent(self):
        self._run(self.mod.upgrade)
        self._run(self.mod.downgrade)
        self._run(self.mod.downgrade)
        columns = self._columns()
        self.assertNotIn("readalong_epub_requested", columns)


if __name__ == "__main__":
    unittest.main()
