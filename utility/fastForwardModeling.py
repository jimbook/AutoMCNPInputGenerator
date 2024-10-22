import numpy as np

from utility import stratumModel
from utility import outputReader
from utility import coordinate

class fastForwardModeling(object):
    def __init__(self, weight:outputReader.MCNP_WWOUTReader, stand:outputReader.MCNP_OutputReader, zone: stratumModel.DetectionZone):
        self.weight = weight
        self._data = None
        self.stand = stand
        self.zone = zone
        self.boundary = stratumModel.AnnulusBoundary(zone)
        self._normalCoe = None

    @property
    def normalCoe(self) -> float:
        if self._normalCoe is None:
            s = 0.
            for i in range(self.weight.data.shape[0]):
                d = self.weight.data[i]
                if d != 0.:
                    p = self.weight.points[i]
                    point = coordinate.Point(*p,coord=self.boundary.zone.coord)
                    if self.boundary.is_in_it(point):
                        s += 1. / d
            standGamma = self.stand.data[0]
            r = standGamma / s
            self._normalCoe = r
        return self._normalCoe

    def stratumGamma(self, stratum:stratumModel.Stratum) -> tuple[float, float]:
        s = 0.
        for i in range(self.weight.data.shape[0]):
            d = self.weight.data[i]
            if d != 0.:
                p = self.weight.points[i]
                point = coordinate.Point(*p)
                if stratum.boundary.is_in_it(point):
                    s += 1. / d
        return s * self.normalCoe, self.stand.error[0]

    def getInIt(self, stratum:stratumModel.Stratum) -> np.array:
        result = np.empty_like(self.weight.data)
        for i in range(self.weight.points.shape[0]):
            p = self.weight.points[i]
            point = coordinate.Point(*p, coord=stratum.dZone.coord)
            if stratum.boundary.is_in_it(point):
                result[i] = 1.
            else:
                result[i] = 0.
        return result

    @property
    def data(self) -> np.array:
        if self._data is None:
            checkZero =(self.weight.data != 0.)
            reciprocal = 1. / self.weight.data[checkZero]
            a = np.zeros_like(self.weight.data)
            a[checkZero] = reciprocal
            self._data = a
        return self._data