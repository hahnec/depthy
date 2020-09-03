from tests.test_structure_tensor import StructTensorTestCase
from tests.test_epi_depth import EpiDepthTestCase
from tests.test_cli import CliTestCase
from tests.test_plt import PlotDepthTestCase


test_classes = [StructTensorTestCase, EpiDepthTestCase, CliTestCase, PlotDepthTestCase]

for test_class in test_classes:
    obj = test_class()
    obj.setUp()
    obj.test_all()
    del obj
