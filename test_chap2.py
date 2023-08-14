import numpy as np
import pytest

np.seterr(divide='ignore')

# Exercise 2
@pytest.mark.parametrize("test_input, expected", [
    (np.array([0,0,1]), 1.0),
    (np.array([1,2,3]), np.linalg.norm(np.array([1,2,3]))),
    (np.array([1,2,3,4]), np.linalg.norm(np.array([1,2,3,4])))
    
])
def test_normOfVect(test_input, expected):
    from my_notebook import normOfVect
    assert normOfVect(test_input) == expected

# Exercise 3
@pytest.mark.parametrize("test_input, expected", [
    (np.array([0,1,0]), 1.0),
    (np.array([0,3,0]), 1.0),
    (np.array([1,2,3,3]), 1.0),
    
])
def test_createUnitVector(test_input, expected):
    from my_notebook import createUnitVector
    assert np.linalg.norm(createUnitVector(test_input)) == expected

# Exercise 4
@pytest.mark.parametrize("test_input, mag ,expected", [
    (np.array([0,1,0]), 4 , np.array([0,4,0])),
    (np.array([1,2,4]), 4,  4 * np.array([1,2,4]) / np.linalg.norm(np.array([1,2,4]))),
    (np.array([1,2,3,3]), 1000, 1000 * np.array([1,2,3,3]) / np.linalg.norm(np.array([1,2,3,3]))),
    
])
def test_createUnitVector(test_input, mag ,expected):
    from my_notebook import createMagVector
    assert (createMagVector(test_input, mag) == expected).all()

# Exercise 5
