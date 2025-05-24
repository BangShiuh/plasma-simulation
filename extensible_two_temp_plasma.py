import cantera as ct

class ExtensibleTwoTempPlasmaData(ct.ExtensibleRateData):
    __slots__ = ("T", "Te")

    def __init__(self):
        self.T = None
        self.Te = None

    def update(self, gas):
        T = gas.T
        # Assuming electron temperature is stored as a custom property
        # You may need to adjust this based on how Te is stored in your gas object
        Te = getattr(gas, 'Te', T)  # fallback to gas temp if not available

        if self.T != T or self.Te != Te:
            self.T = T
            self.Te = Te
            return True
        else:
            return False

@ct.extension(name="extensible-two-temp-plasma", data=ExtensibleTwoTempPlasmaData)
class TwoTempPlasmaRate(ct.ExtensibleRate):
    __slots__ = ("A", "b")

    def set_parameters(self, params, units):
        self.A = params.convert_rate_coeff("A", units)
        self.b = params["b"]

    def get_parameters(self, params):
        params.set_quantity("A", self.A, self.conversion_units)
        params["b"] = self.b

    def validate(self, equation, soln):
        if self.A < 0:
            raise ValueError(f"Found negative 'A' for reaction {equation}")

    def eval(self, data):
        # rate = A * (Te / Tgas)^b
        return self.A * (data.Te / data.T) ** self.b


plasma = ct.Solution("data/helium-oxygen-hydrogen-plasma.yaml")
print(plasma.Te)