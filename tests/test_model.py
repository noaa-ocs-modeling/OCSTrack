
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


import numpy as np

from ocstrack.Model.model import stretching, set_depth, _parse_gr3_mesh


class TestStretching:
    def test_all_vstretching_variants_return_arrays(self):
        """stretching returns (s, C) arrays for each Vstretching mode."""
        for vstretch in (1, 2, 3, 4):
            s, c = stretching(vstretch, theta_s=7.0, theta_b=4.0,
                              hc=50.0, N=10, kgrid=0)
            assert len(s) == len(c)
            assert len(s) > 0

    def test_kgrid_one_uses_w_points(self):
        """stretching returns arrays for both RHO-points and W-points grids."""
        s0, c0 = stretching(2, 7.0, 4.0, 50.0, 10, kgrid=0)
        s1, c1 = stretching(2, 7.0, 4.0, 50.0, 10, kgrid=1)
        assert len(s0) == len(c0)
        assert len(s1) == len(c1)


class TestSetDepth:
    def test_returns_expected_shape(self):
        """set_depth returns a 3D array (eta, xi, N) of depths."""
        h = np.ones((5, 4)) * 100.0
        zeta = np.zeros((5, 4))
        z = set_depth(Vtransform=2, Vstretching=4, theta_s=7.0, theta_b=4.0,
                      hc=50.0, N=10, igrid=1, h=h, zeta=zeta)
        assert z.shape == (5, 4, 10)
        # Depths should be negative (below surface)
        assert np.all(z <= 0)


class TestParseGr3Mesh:
    def test_parses_nodes(self, tmp_path):
        """_parse_gr3_mesh reads node lon/lat/depth from an hgrid.gr3 file."""
        gr3 = tmp_path / "hgrid.gr3"
        # header: name line, then "<n_elements> <n_nodes>"
        lines = ["test mesh", "0 3"]
        # node_id lon lat depth
        lines += ["1 -70.0 40.0 10.0",
                  "2 -71.0 41.0 20.0",
                  "3 -72.0 42.0 30.0"]
        gr3.write_text("\n".join(lines) + "\n")

        lons, lats, depths = _parse_gr3_mesh(str(gr3))
        assert list(lons) == [-70.0, -71.0, -72.0]
        assert list(lats) == [40.0, 41.0, 42.0]
        assert list(depths) == [10.0, 20.0, 30.0]

