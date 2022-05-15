import json
from json import JSONDecodeError

import numpy as np

from som_tsp.error import FileReadError, ValidationError
from som_tsp.solver import Instances


def read_instance(filename: str) -> Instances:
    try:
        with open(filename) as file:
            instance = json.load(file)
    except FileNotFoundError as error:
        raise FileReadError(f"File not found '{filename}'")
    except JSONDecodeError as error:
        raise FileReadError(error.msg)

    return np.array(with_validation(instance))


def with_validation(raw_instance: dict | list) -> list[tuple[float, float]]:
    if type(raw_instance) is not list:
        raise ValidationError("Instance is not a list")

    instance = []
    for i, city in enumerate(raw_instance):
        if type(city) is not list:
            raise ValidationError(f"City {i} is not a list")
        if len(city) != 2:
            raise ValidationError(f"City {i} needs to consist out of two coordinates")
        if not isinstance(city[0], int) and not isinstance(city[0], float):
            raise ValidationError(f"City {i}'s x-coordinate needs to be numeric")
        if not isinstance(city[1], int) and not isinstance(city[1], float):
            raise ValidationError(f"City {i}'s y-coordinate needs to be numeric")

        instance.append((float(city[0]), float(city[1])))
    return instance
