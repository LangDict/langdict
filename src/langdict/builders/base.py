


class Builder:
    """Builder interface"""

    def __init__(self):
        pass

    @classmethod
    def build(cls):
        raise NotImplementedError(
            f"Builder [{type(cls).__name__}] is missing the required \"build\" function"
        )
