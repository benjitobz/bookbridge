"""add koreader_book_status (per-device sidecar reading status)

Revision ID: b4e7d2a8c163
Revises: a6c3e9f1b7d4
Create Date: 2026-09-18
"""

from alembic import op
import sqlalchemy as sa


revision = "b4e7d2a8c163"
down_revision = "a6c3e9f1b7d4"
branch_labels = None
depends_on = None

TABLE = "koreader_book_status"


def _tables(inspector) -> set[str]:
    return set(inspector.get_table_names())


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE in _tables(inspector):
        return

    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("md5", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("device", sa.String(length=128), nullable=True),
        sa.Column("device_id", sa.String(length=128), nullable=True),
        sa.Column("device_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("modified", sa.String(length=10), nullable=True),
        sa.Column("received_at", sa.Float(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "md5", "user_id", "device_key", name="uq_koreader_book_status_md5_user_device"
        ),
    )
    op.create_index("ix_koreader_book_status_md5", TABLE, ["md5"])
    op.create_index("ix_koreader_book_status_user_id", TABLE, ["user_id"])
    op.create_index("ix_koreader_book_status_device_key", TABLE, ["device_key"])
    op.create_index("ix_koreader_book_status_last_updated", TABLE, ["last_updated"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in _tables(inspector):
        return
    op.drop_index("ix_koreader_book_status_last_updated", table_name=TABLE)
    op.drop_index("ix_koreader_book_status_device_key", table_name=TABLE)
    op.drop_index("ix_koreader_book_status_user_id", table_name=TABLE)
    op.drop_index("ix_koreader_book_status_md5", table_name=TABLE)
    op.drop_table(TABLE)
