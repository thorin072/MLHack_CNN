from django.shortcuts import render
from sqlalchemy.orm import Session
from django.conf import settings
from sqlalchemy.sql import text
from .models import Dishes
from .models import Products
from .models import Catalogs
from django.http import HttpResponse
import json
from django.views.decorators.csrf import csrf_exempt

engine = settings.ENGINE

@csrf_exempt
def get_information(request):
    request_data = json.loads(request.body)
    session = Session(bind=engine)
    #
    # тут нужно найти id блюда по картинки
    id = 1  # к примеру
    #

    response = [{
        'authentication': True,
        'dish_name': instance_d.name,
        'proteins_sum': instance_d.proteins_sum,
        'fat_sum': instance_d.fat_sum,
        'carbohydrates_sum': instance_d.carbohydrates_sum,
        'kilocalories_sum': instance_d.kilocalories_sum,
        'products': [
            {
                'id': instance_p.id,
                'product_name': instance_p.name,
                'proteins': instance_p.proteins,
                'fat': instance_p.fat,
                'carbohydrates': instance_p.carbohydrates,
                'kilocalories': instance_p.kilocalories,
                'weight': instance_p.weight
            } for instance_p in session.query(Products).filter(Products.dish_id == id)
        ]
    }
    for instance_d in session.execute("select d.name as name, sum(proteins) as proteins_sum, sum(fat) as fat_sum \
													, sum(carbohydrates) as carbohydrates_sum, sum(kilocalories) as kilocalories_sum \
													from dishes as d \
														inner join catalogs as c \
															on d.id = c.dish_id" \
                                      "                    and d.id = :x \
														inner join products as p \
															on c.product_id = p.id \
											group by d.name", x = id)
    ]

    response = json.dumps(response, ensure_ascii=False)
    return HttpResponse(response)





