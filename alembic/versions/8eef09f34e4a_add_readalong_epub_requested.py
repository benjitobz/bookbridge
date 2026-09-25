"""add readalong_epub_requested to books

Revision ID: 8eef09f34e4a
Revises: b4e7d2a8c163
Create Date: 2026-09-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8eef09f34e4a"
down_revision: Union[str, Sequence[str], None] = "b4e7d2a8c163"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("books")}
    if "readalong_epub_requested" not in columns:
        op.add_column(
            "books",
            sa.Column(
                "readalong_epub_requested",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("books")}
    if "readalong_epub_requested" in columns:
        op.drop_column("books", "readalong_epub_requested")
