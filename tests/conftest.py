import pytest

from promptcorrect.segmenters.dummy import DummySegmenter
from promptcorrect.testing import make_synthetic_objects


@pytest.fixture
def objects():
    return make_synthetic_objects(n_images=3, objects_per_image=2, seed=0)


@pytest.fixture
def seg(objects):
    s = DummySegmenter(objects)
    s.set_image(1)
    return s
