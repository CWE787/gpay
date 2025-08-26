from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Template(Base):
    __tablename__ = "templates"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # np.float32 bytes
    dim = Column(Integer, nullable=False)
    num_frames = Column(Integer, default=0)
    model = Column(String, default="gabor_lbp_v1")
    quality = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Engine/session factory
_engine = None
_Session = None

def init_db(path: str = "sqlite:///palm_pay.db"):
    global _engine, _Session
    _engine = create_engine(path)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine)
    return _Session