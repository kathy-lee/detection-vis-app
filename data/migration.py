import os
from sqlalchemy import create_engine
from database import Base
from urllib.parse import quote

from models import User

#DATABASE_URL = "postgresql://user:password@localhost:5432/dbname".format()
# DATABASE_URL = "postgresql://{}:{}@{}:{}/{}?sslmode=disable".format(
#         quote(os.environ["DB_USER"]),
#         quote(os.environ["DB_PSWD"]),
#         os.environ["DB_HOST"],
#         os.environ["DB_PORT"],
#         os.environ["DB_NAME"],
#     )

# engine = create_engine(DATABASE_URL)

# # Run the migrations
# Base.metadata.create_all(engine)
