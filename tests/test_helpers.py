import json
import math
import numpy as np
import pytest

from types import SimpleNamespace

from xaicompare.helpers import make_json_safe


# --------- Tests ----------

@pytest.mark.parametrize(
    "value,expected_type,expected_value",
    [
        (np.int8(1), int, 1),
        (np.int32(-7), int, -7),
        (np.int64(1234567890123), int, 1234567890123),
        (np.float32(1.5), float, 1.5),
        (np.float64(-0.25), float, -0.25),
        (np.bool_(True), bool, True),
        (np.bool_(False), bool, False),
    ],
)
def test_numpy_scalars(value, expected_type, expected_value):
    out = make_json_safe(value)
    assert isinstance(out, expected_type)
    # floats: handle potential float32 vs float rounding
    if expected_type is float and isinstance(value, (np.floating,)):
        assert math.isclose(out, expected_value, rel_tol=0, abs_tol=1e-12)
    else:
        assert out == expected_value


def test_none_passthrough():
    assert make_json_safe(None) is None


def test_numpy_array_simple():
    arr = np.array([[1, 2], [3, 4]])
    out = make_json_safe(arr)
    assert out == [[1, 2], [3, 4]]  # tolist()
    # Ensure json serializable
    json.dumps(out)


def test_numpy_array_object_dtype():
    arr = np.array([np.int64(5), np.float32(2.5), np.bool_(True)], dtype=object)
    out = make_json_safe(arr)
    # tolist() preserves numpy scalars; function should recurse only for ndarray itself
    # Here, since the function only does tolist() for ndarray and NOT recursive, the elements
    # would still be numpy scalars if dtype=object. But since np.ndarray case returns tolist(),
    # and numpy scalars become Python types when cast to list? Not always. So we ensure safety:
    # We can pass the array as part of a larger structure (list/tuple/dict) to test deeper conversion.
    # For this test, confirm basic list conversion.
    assert isinstance(out, list)
    # However, those elements may still be numpy scalars. Let's make sure it can still be dumped:
    # json.dumps on numpy scalars fails, so this would fail if elements are not converted.
    # The current function does NOT recursively convert elements for ndarray.
    # Therefore, to ensure the function's current behavior, we must assert it produces a list
    # with raw values (tolist usually converts numpy scalars to Python types, which is OK).
    json.dumps(out)  # should not raise


def test_list_and_tuple_nested():
    data = [np.int64(3), (np.float64(1.25), [np.bool_(True), None])]
    out = make_json_safe(data)
    # tuple becomes list; nested conversions applied
    assert out == [3, [1.25, [True, None]]]
    json.dumps(out)


def test_dict_with_numpy_values_and_nonstring_keys():
    data = {
        np.int64(7): np.float32(3.5),
        (1, 2): [np.bool_(False), np.array([1, 2, 3])],
        "plain": "ok",
    }
    out = make_json_safe(data)
    # Keys become strings
    assert set(out.keys()) == {"7", "(1, 2)", "plain"}
    assert out["7"] == 3.5
    assert out["(1, 2)"] == [False, [1, 2, 3]]
    assert out["plain"] == "ok"
    json.dumps(out)


def test_already_json_serializable_passthrough():
    data = {"a": [1, 2, 3], "b": {"x": True, "y": None}}
    out = make_json_safe(data)
    # keys already strings, values plain
    assert out == data
    json.dumps(out)


def test_fallback_to_str_for_non_serializable_set():
    data = {"s": {1, 2, 3}}  # set is not JSON-serializable
    out = make_json_safe(data)
    assert "s" in out
    # Fallback is str(set), order not guaranteed; just ensure it's a string and mentions elements
    assert isinstance(out["s"], str)
    for elem in ["1", "2", "3"]:
        assert elem in out["s"]
    json.dumps(out)


def test_fallback_to_str_for_bytes():
    data = b"abc"
    out = make_json_safe(data)
    # bytes are not JSON-serializable by default; expect "b'abc'"
    assert isinstance(out, str)
    assert out.startswith("b'")
    assert "abc" in out
    json.dumps(out)



def test_custom_object_fallback_to_str():
    obj = SimpleNamespace(a=1, b=2)
    out = make_json_safe(obj)
    assert isinstance(out, str)
    # The exact format is implementation-specific; check stable parts:
    assert "a=1" in out and "b=2" in out
    json.dumps(out)  # should not raise



def test_roundtrip_large_structure():
    complex_data = {
        "ints": [np.int64(i) for i in range(5)],
        "floats": (np.float32(0.1), np.float64(0.2)),
        "bools": [np.bool_(True), np.bool_(False)],
        "array": np.arange(6).reshape(2, 3),
        "nested": [{"k": (np.int32(10), [np.array([1, 2]), {"z": np.bool_(True)}])}],
        None: "becomes 'None' key",
    }
    out = make_json_safe(complex_data)
    # Key None becomes "None"
    assert "None" in out
    # Tuple converted to list
    assert isinstance(out["floats"], list)
    # NumPy array converted to list of lists
    assert out["array"] == [[0, 1, 2], [3, 4, 5]]
    # Deep nested conversions ok
    assert out["nested"][0]["k"][1][0] == [1, 2]
    json.dumps(out)  # should not raise