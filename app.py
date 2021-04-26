from flask import Flask, request, jsonify
from collections import namedtuple
from decimal import Decimal, InvalidOperation
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
from flask_cors import CORS, cross_origin
import json
import re
import sqlite3

app = Flask(__name__)
CORS(app)


db = None
name_type_ids = dict()


def build_typid_map(db):
    global name_type_ids
    cursor = db.cursor()
    query = f'SELECT typeName, typeID FROM invTypes;'
    cursor.execute(query)
    name_type_ids = {x[0]: x[1] for x in cursor.fetchall()}


def get_typeid_from_name(db, name):
    cursor = db.cursor()
    query = f'SELECT typeId FROM invTypes WHERE typeName = "{name}";'
    cursor.execute(query)
    row = cursor.fetchone()
    if row is None:
        return None
    return row[0]


def get_typeids_from_names(db, names):
    cursor = db.cursor()
    names = ','.join(map(lambda x: f'\"{x}\"', names))
    query = f'SELECT typeId FROM invTypes WHERE typeName IN ({names});'
    cursor.execute(query)
    type_ids = cursor.fetchall()
    return type_ids


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
        item.price = parse_price(values[price_column_index])
        item.volume = parse_volume(values[volume_column_index])
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


def pack_items(items, volume: Decimal, should_allow_splitting: False):
    # The solver only works with integers, so we need to multiply the volumes
    # so all significant figures are represented.
    if should_allow_splitting:
        # fractional knapsack problem
        packed_items = []
        items.sort(key=lambda x: x.price_per_volume, reverse=True)
        for item in items:
            if volume < 0:
                break
            if item.volume < volume:
                packed_items.append(item)
                volume -= item.volume
            else:
                quantity = int(volume / item.volume_per_unit)
                if quantity < 1:
                    continue
                else:
                    vpu = item.volume_per_unit
                    ppu = item.price_per_unit
                    new_item = Item()
                    new_item.name = item.name
                    new_item.quantity = quantity
                    new_item.volume = vpu * quantity
                    new_item.price = ppu * quantity
                    new_item.typeid = item.typeid
                    new_item.is_split = True
                    packed_items.append(new_item)
                    volume -= new_item.volume

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
        packed_weights = []
        total_weight = 0

        for i in range(len(values)):
            if solver.BestSolutionContains(i):
                packed_items.append(i)
                packed_weights.append(weights[0][i])
                total_weight += weights[0][i]

        packed_items = [x for x in range(0, len(weights[0])) if solver.BestSolutionContains(x)]
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
    items = parse_items(request.json['blob'])
    packed_items = pack_items(items, volume, should_allow_splitting)
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
