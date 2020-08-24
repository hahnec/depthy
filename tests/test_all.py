from tests.test_structure_tensor import StructTensorTestCase
from tests.test_epi_depth import EpiDepthTestCase
from tests.test_cli import CliTestCase


test_classes = [StructTensorTestCase, EpiDepthTestCase, CliTestCase]

for test_class in test_classes:
    obj = test_class()
    obj.setUp()
    obj.test_all()
    del obj
