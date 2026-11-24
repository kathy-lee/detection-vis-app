"""files.path and files.config relative to DATASET_ROOTDIR

The dataset root is a machine setting (the repo's .env), so the table stores paths below
it and the code joins the root back on when it reads a row
(detection_vis_backend.paths.resolve_stored_path / resolve_stored_config). Values outside
the root, names that are not paths (NuScenes keeps its version in config) and empty
values are left as they are.

Downgrade is not an exact inverse for values that were relative before the upgrade
(e.g. the './1.bag' placeholder rows, RaDICaL's 'radical/radarcfg'): it cannot tell them
from the ones upgrade made relative, so it puts the root in front of them too. For an
exact restore, load the pg_dump taken before upgrading.

Revision ID: 3d1f6a2b9c47
Revises: 752367e53636
Create Date: 2026-11-24

"""
import os

import sqlalchemy as sa
from alembic import op

from detection_vis_backend.paths import dataset_root


# revision identifiers, used by Alembic.
revision = '3d1f6a2b9c47'
down_revision = '752367e53636'
branch_labels = None
depends_on = None


def to_relative(value, root):
    """value without `root/` in front; anything else is returned as it is."""
    prefix = root.rstrip("/") + "/"
    if value and value.startswith(prefix):
        return value[len(prefix):]
    return value


def to_absolute(value, root, is_config):
    """`root/value` for a relative value. A config is joined only when that names an
    existing file, so names such as 'v1.0-mini' stay names."""
    if not value or os.path.isabs(value):
        return value
    joined = os.path.join(root, value)
    if is_config and not os.path.exists(joined):
        return value
    return joined


def _rewrite(convert_path, convert_config):
    conn = op.get_bind()
    files = sa.table("files", sa.column("id", sa.Integer), sa.column("path", sa.String),
                     sa.column("config", sa.String))
    for row in conn.execute(sa.select(files.c.id, files.c.path, files.c.config)).fetchall():
        path, config = convert_path(row.path), convert_config(row.config)
        if (path, config) != (row.path, row.config):
            conn.execute(files.update().where(files.c.id == row.id).values(path=path, config=config))


def upgrade() -> None:
    root = dataset_root()
    _rewrite(lambda v: to_relative(v, root), lambda v: to_relative(v, root))


def downgrade() -> None:
    root = dataset_root()
    _rewrite(lambda v: to_absolute(v, root, is_config=False), lambda v: to_absolute(v, root, is_config=True))
