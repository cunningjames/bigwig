#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from bigwig.skeleton import fib

__author__ = "James Cunningham"
__copyright__ = "James Cunningham"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
