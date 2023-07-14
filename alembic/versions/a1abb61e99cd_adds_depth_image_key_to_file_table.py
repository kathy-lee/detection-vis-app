"""adds depth_image key to file table

Revision ID: a1abb61e99cd
Revises: f2b735d809cb
Create Date: 2023-07-09 20:55:28.406508

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1abb61e99cd'
down_revision = 'f2b735d809cb'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('files', sa.Column('depth_image', sa.Boolean(), nullable=True))
    op.create_index(op.f('ix_files_depth_image'), 'files', ['depth_image'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_files_depth_image'), table_name='files')
    op.drop_column('files', 'depth_image')
    # ### end Alembic commands ###