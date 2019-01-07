class GridParam:
    """
    Formal parameter for GridSearch

    Parameters
    ----------
    value : scalar
        Centered value for range
    step_size : scalar
        Step between each value
    max_value : scalar
        End of range
    min_value : scalar, default 0
        Start of range
    length : int
        Length of the range
    """
    def __init__(self, value, step_size, max_value, min_value=0, length=5):
        self.value = value
        self.step_size = step_size
        self.max = max_value
        self.min = min_value
        self.length = length

    def get_range(self):
        """
        Range computation

        Return
        ----------
        list :
            Sorted list of parameters
        """
        params = [self.value]
        n_vals = (self.length // 2)
        self.length -= 1
        i = 0
        while i < n_vals:
            value = max(self.value - (i + 1) * self.step_size, self.min)
            params = [value] + params
            i += 1
        i = 0
        while i < self.length - n_vals:
            value = min(self.value + (i + 1) * self.step_size, self.max)
            params = params + [value]
            i += 1
        return sorted(list(set(params)))
