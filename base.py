"""

Base-classes for ivim model and fits.

All the models in the reconst module follow the same template: a Model object
is used to represent the abstract properties of the model, that are independent
of the specifics of the data . These properties are reused whenver fitting a
particular set of data (different voxels, for example).


"""


class IvimModel(object):
    """ Abstract class for ivim models
    """

    def __init__(self, img, bvals):
        """Initialization of the abstract class for ivim model

        Parameters
        ----------
        img     : img data
        bvals   : bvalue array

        """
        self.img = img
        self.bvals = bvals

    def fit(self, data, mask=None, **kwargs):
        return IvimFit(self, data)


class IvimFit(object):
    """ Abstract class which holds the fit result of Ivim

    For example that could be holding img, S0, D .... 
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
