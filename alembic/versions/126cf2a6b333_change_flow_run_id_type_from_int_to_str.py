"""change flow_run_id type from int to str

Revision ID: 126cf2a6b333
Revises: e7cfd895a4f6
Create Date: 2023-08-06 16:01:20.122463

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '126cf2a6b333'
down_revision = 'e7cfd895a4f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('models', 'flow_run_id',
                    type_=sa.String(),
                    existing_type=sa.Integer())
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('models', 'flow_run_id',
                    type_=sa.Integer(),
                    existing_type=sa.String())
    # ### end Alembic commands ###