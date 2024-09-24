from typing import Union


class Parameter:

    def __init__(self, data: Union[str, int, float, bool, None]):
        self._data = data

    @property
    def value(self):
        return self._data

