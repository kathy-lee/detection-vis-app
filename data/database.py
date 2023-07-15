import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote



#DATABASE_URL = "postgresql://user:password@localhost:5432/dbname"
#DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
# DATABASE_URL = "postgresql://{}:{}@{}:{}/{}?sslmode=disable".format(
#         quote(os.environ["DB_USER"]),
#         quote(os.environ["DB_PSWD"]),
#         os.environ["DB_HOST"],
#         os.environ["DB_PORT"],
#         os.environ["DB_NAME"],
#     )
database_url = os.getenv('DATABASE_URL')

engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Create a new session
# db = SessionLocal()

# # Define the user data
# user_data = UserCreate(email="test@example.com", password="testpassword")

# # Call the create_user function to insert the user
# create_user(db=db, user=user_data)

# # Close the session
# db.close()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @app.post("/users/", response_model=User)
# def create_user(user: User, db: Session = Depends(get_db)):
#     db.add(user)
#     db.commit()
#     db.refresh(user)
#     return user


# from sqlalchemy import create_engine
# from model import Base, User

# engine = create_engine('sqlite:///example.db')  # Use an SQLite database file named example.db
# Base.metadata.create_all(engine)