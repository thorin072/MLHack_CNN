import sqlalchemy as al
from sqlalchemy.ext.declarative import declarative_base

# Create your models here.
Base = declarative_base()


class Dishes(Base):
    __tablename__ = 'dishes'
    id = al.Column("id", al.Integer, primary_key=True)
    name = al.Column("name", al.String)

    def __init__(self, name):
        self.name = name


class Products(Base):
    __tablename__ = 'products'
    id = al.Column("id", al.Integer, primary_key=True)
    name = al.Column("name", al.String)


proteins = al.Column("proteins", al.Integer)
fat = al.Column("fat", al.Integer)
carbohydrates = al.Column("carbohydrates", al.String)
kilocalories = al.Column("kilocalories", al.Integer)


def __init__(self, name, proteins, fat, carbohydrates, kilocalories):
    self.name = name
    self.proteins = proteins
    self.fat = fat
    self.carbohydrates = carbohydrates
    self.kilocalories = kilocalories


class Catalogs(Base):
    __tablename__ = 'catalogs'
    id = al.Column("id", al.Integer, primary_key=True)
    product_id = al.Column("product_id", al.Integer)
    dish_id = al.Column("dish_id", al.Integer)
    weight = al.Column("weight", al.Integer)

    def __init__(self, product_id, dish_id, weight):
        self.product_id = product_id
        self.dish_id = dish_id
        self.weight = weight