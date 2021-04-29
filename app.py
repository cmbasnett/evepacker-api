from flask import Flask, request, jsonify
from collections import namedtuple
from decimal import Decimal, InvalidOperation
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
from flask_cors import CORS, cross_origin
import json
import re
import sqlite3
from flask_redis import FlaskRedis

app = Flask(__name__)
redis_client = FlaskRedis(app)
CORS(app)


def populate_redis_cache():
    db = sqlite3.connect('sqlite-latest.sqlite')
    cursor = db.cursor()
    query = f'SELECT typeName, typeID, volume FROM invTypes;'
    cursor.execute(query)
    print('Populating redis cache...')
    for name, type_id, volume in cursor.fetchall():
        redis_client.set(name, json.dumps({'typeId': type_id, 'volume': volume}))


class Item:
    def __init__(self):
        self.name = ''
        self.price = Decimal()
        self.volume = Decimal()
        self.quantity = 0
        self.typeid = 0
        self.is_split = False

    @property
    def price_per_volume(self):
        return self.price / self.volume

    @property
    def volume_per_unit(self):
        return self.volume / self.quantity

    @property
    def price_per_unit(self):
        return self.price / self.quantity

    @property
    def units_per_volume(self):
        return self.quantity / self.volume


def parse_price(s) -> Decimal:
    try:
        return Decimal(s[:-4].replace(',', ''))
    except InvalidOperation:
        return Decimal()


def parse_volume(s) -> Decimal:
    try:
        return Decimal(s[:-3].replace(',', ''))
    except InvalidOperation:
        return Decimal()


def parse_quantity(s) -> int:
    try:
        return int(s.replace(',', ''))
    except ValueError:
        return 1


def parse_items(blob):
    lines = blob.split('\n')
    quantity_column_index = None
    volume_column_index = None
    price_column_index = None
    line = lines[0]
    values = line.split('\t')
    for column_index, value in enumerate(values):
        if value.endswith('m3'):
            volume_column_index = column_index
        elif value.endswith('ISK'):
            price_column_index = column_index
        elif re.match(r'\d+', value) is not None:
            quantity_column_index = column_index
    items = []
    for line in lines:
        values = line.split('\t')
        item = Item()
        item.name = values[0]
        db_item = redis_client.get(item.name)
        if db_item is not None:
            db_item = json.loads(db_item)
        item.typeid = db_item.get('typeId', None) if db_item is not None else None
        item.quantity = parse_quantity(values[quantity_column_index])
        item.price = parse_price(values[price_column_index]) if price_column_index is not None else 0
        item.volume = Decimal(db_item['volume']) * item.quantity if volume_column_index is None else parse_volume(values[volume_column_index])
        items.append(item)
    return items


class Packing:
    def __init__(self):
        self.items = []
        self.volume = Decimal()
        self.price = Decimal()


def pack_items(items, volume: Decimal, should_allow_splitting: False, value_limit: Decimal('inf')):
    # The solver only works with integers, so we need to multiply the volumes
    # so all significant figures are represented.
    value = value_limit
    if should_allow_splitting:
        # fractional knapsack problem
        packed_items = []
        items.sort(key=lambda x: x.price_per_volume, reverse=True)
        for item in items:
            if volume < 0:
                break
            if item.volume < volume and item.price < value:
                packed_items.append(item)
                volume -= item.volume
                value -= item.price
            else:
                quantity = int(volume / item.volume_per_unit)
                if value.is_finite():
                    quantity = min(quantity, int(value / item.price_per_unit))
                if quantity < 1:
                    continue
                else:
                    volume_per_unit = item.volume_per_unit
                    price_per_unit = item.price_per_unit
                    new_item = Item()
                    new_item.name = item.name
                    new_item.quantity = quantity
                    new_item.volume = volume_per_unit * quantity
                    new_item.price = price_per_unit * quantity
                    new_item.typeid = item.typeid
                    new_item.is_split = True
                    packed_items.append(new_item)
                    volume -= new_item.volume
                    value -= new_item.price

    else:
        values = list(map(lambda x: x.price, items))
        weights = [list(map(lambda x: x.volume * 100, items))]
        capacities = [float(volume * 100)]

        solver = KnapsackSolver(
            KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            'Evepacker'
        )

        solver.Init(values, weights, capacities)
        solver.Solve()
        packed_items = []

        for i in range(len(values)):
            if solver.BestSolutionContains(i):
                if values[i] > value:
                    continue
                packed_items.append(i)
                value -= values[i]

        packed_items = [items[p] for p in packed_items]

    return packed_items


def get_total_volume(items):
    return float(sum(map(lambda x: x.volume, items)))


def get_total_price(items):
    return float(sum(map(lambda x: x.price, items)))


@app.route('/api/pack', methods=['POST'])
@cross_origin()
def pack():
    volume = Decimal(request.json['volume'].replace(',', ''))
    should_allow_splitting = bool(request.json.get('should_allow_splitting', True))
    value_limit = Decimal(request.json.get('value_limit', Decimal('inf')))
    items = parse_items(request.json['blob'])
    packed_items = pack_items(items, volume, should_allow_splitting, value_limit)
    return jsonify({
        'price': get_total_price(packed_items),
        'volume': get_total_volume(packed_items),
        'total_price': get_total_price(items),
        'total_volume': get_total_volume(items),
        'items': list(map(lambda x: {
            'name': x.name,
            'typeid': x.typeid,
            'quantity': x.quantity,
            'is_split': x.is_split,
            'volume': str(x.volume),
            'price': str(x.price)}, packed_items))
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# populate_redis_cache()
