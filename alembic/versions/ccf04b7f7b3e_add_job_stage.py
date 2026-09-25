"""add stage to jobs

Revision ID: ccf04b7f7b3e
Revises: 3f6c8a1d9e42
Create Date: 2026-09-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ccf04b7f7b3e"
down_revision: Union[str, Sequence[str], None] = "3f6c8a1d9e42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("jobs")}
    if "stage" not in columns:
        op.add_column(
            "jobs",
            sa.Column("stage", sa.String(length=50), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("jobs")}
    if "stage" in columns:
        op.drop_column("jobs", "stage")
