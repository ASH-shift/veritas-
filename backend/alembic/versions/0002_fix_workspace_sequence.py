"""Advance the workspaces id sequence past the seeded demo workspace

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-24

0001 inserts the demo workspace with an explicit id=1, which does not advance
the SERIAL sequence. The first POST /auth/register then tried id=1 again and
failed with a duplicate primary key. This sets the sequence to MAX(id).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "SELECT setval(pg_get_serial_sequence('workspaces', 'id'), "
        "(SELECT COALESCE(MAX(id), 1) FROM workspaces))"
    )


def downgrade() -> None:
    # Nothing to undo: the sequence position is not schema.
    pass
