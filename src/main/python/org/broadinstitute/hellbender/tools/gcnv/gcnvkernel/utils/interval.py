import numpy as np
from abc import abstractmethod
from typing import Dict


class IntervalAnnotation:
    def __init__(self, raw_value):
        self.parsed_value = self.parse(raw_value)

    def get_value(self):
        return self.parsed_value

    @staticmethod
    @abstractmethod
    def parse(raw_value):
        pass

    @staticmethod
    @abstractmethod
    def get_key() -> str:
        pass


class Interval:
    """ This class represents a genomic interval along with annotations
    """
    def __init__(self, contig: str, start: int, stop: int, name: str):
        self.contig: str = str(contig)
        self.start: int = int(start)
        self.stop: int = int(stop)
        if name is None:
            self.name: str = contig + "_" + str(start) + "_" + str(stop)
        else:
            self.name: str = str(name)
        self._key = (self.contig, self.start, self.stop)
        self.annotations = dict()

    def add_annotation(self, key: str, annotation: IntervalAnnotation):
        self.annotations[key] = annotation

    def get_annotation(self, key: str):
        return self.annotations[key].get_value()

    def get_padded(self, padding: int):
        assert padding >= 0, "padding must be >= 0"
        return Interval(self.contig, self.start - padding, self.stop + padding,
                        self.name + "_symm_pad_" + str(padding))

    def overlaps_with(self, other):
        assert isinstance(other, Interval), "the other object is not an interval!"
        if self.contig != other.contig:
            return False
        else:
            if other.start <= self.stop <= other.stop or other.start <= self.start <= other.stop:
                return True
            else:
                return False

    def get_midpoint(self):
        return 0.5 * (self.start + self.stop)

    def distance(self, other):
        if self.contig != other.contig:
            return np.inf
        else:
            return np.abs(self.get_midpoint() - other.get_midpoint())

    def __eq__(self, other):
        return self._key == other._key

    def __lt__(self, other):
        return self._key.__lt__(other._key)

    def __hash__(self):
        return hash(self._key)

    def __str__(self):
        return str(self._key)

    def __repr__(self):
        return self.__str__()


class GCContentAnnotation(IntervalAnnotation):
    """ This class represents GC content annotation that can be added to an Interval
    """
    def __init__(self, gc_content):
        super().__init__(gc_content)

    @staticmethod
    def parse(raw_value):
        gc_content = float(raw_value)
        if not 0.0 <= gc_content <= 1.0:
            raise ValueError("GC content ({0}) must be a float in range [0, 1]".format(gc_content))
        return gc_content

    @staticmethod
    def get_key():
        return "GC_CONTENT"

interval_annotations_dict: Dict[str, IntervalAnnotation] = {
    GCContentAnnotation.get_key(): GCContentAnnotation
}

interval_annotations_dtypes: Dict[str, object] = {
    GCContentAnnotation.get_key(): float
}
