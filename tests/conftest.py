import logging
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
