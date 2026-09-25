"""add kind to jobs

Revision ID: 3f6c8a1d9e42
Revises: 8eef09f34e4a
Create Date: 2026-09-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3f6c8a1d9e42"
down_revision: Union[str, Sequence[str], None] = "8eef09f34e4a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match src.db.models.JOB_KIND_ALIGNMENT -- every job row created before
# this column existed (alignment-build/retry tracking and Forge & Match) is
# retroactively that kind, so existing rows migrate sanely with no behavior
# change for callers that don't yet distinguish kinds.
_EXISTING_JOB_KIND = "alignment"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("jobs")}
    if "kind" not in columns:
        op.add_column(
            "jobs",
            sa.Column(
                "kind",
                sa.String(length=50),
                nullable=False,
                server_default=_EXISTING_JOB_KIND,
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("jobs")}
    if "kind" in columns:
        op.drop_column("jobs", "kind")
