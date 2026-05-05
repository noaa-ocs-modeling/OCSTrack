
from ocstrack.Model.model import natural_sort_key

def test_natural_sort_key():
    """
    Test that the natural_sort_key function correctly sorts filenames
    containing numbers.
    """
    unsorted_list = ["file10.nc", "file2.nc", "file1.nc"]
    expected_list = ["file1.nc", "file2.nc", "file10.nc"]

    sorted_list = sorted(unsorted_list, key=natural_sort_key)

    assert sorted_list == expected_list

def test_natural_sort_key_with_different_prefixes():
    """
    Test that sorting is correct with mixed prefixes or names.
    """
    unsorted_list = ["z_output_10.dat", "z_output_1.dat", "a_output_5.dat"]
    expected_list = ["a_output_5.dat", "z_output_1.dat", "z_output_10.dat"]

    sorted_list = sorted(unsorted_list, key=natural_sort_key)

    assert sorted_list == expected_list
