from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey

Base = declarative_base()


class Definition(Base):
    __tablename__ = "DEFINITIONS"
    id_ = Column(Integer, primary_key=True, index=True)
    word_id = Column(Integer)
    meaning_id = Column(Integer)
    definition_id = Column(Integer)
    lang = Column(String)
    value = Column(String)
    word = Column(String)


class Synonym(Base):
    __tablename__ = "SYNONYMS"
    id_ = Column(Integer, primary_key=True, index=True)
    head_id = Column(Integer, ForeignKey("DEFINITIONS.word_id"), nullable=False)
    tail_id = Column(Integer, ForeignKey("DEFINITIONS.word_id"), nullable=False)
    rel_type = Column(String, nullable=False)
