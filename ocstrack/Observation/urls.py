"""Satellite data URLS"""

import re


def _canonical_sat_key(key: str) -> str:
    """
    Canonicalize a satellite key by stripping all non-alphanumeric
    characters and lower-casing.

    This lets users pass any punctuation/case variant of a satellite name
    and still match the correct entry. For example, ``'jason-3'``,
    ``'jason3'``, ``'Jason_3'`` and ``'JASON 3'`` all canonicalize to
    ``'jason3'``.

    Parameters
    ----------
    key : str
        A user-supplied satellite key.

    Returns
    -------
    str
        The canonicalized key (alphanumeric, lower-case).
    """
    return re.sub(r'[^0-9a-z]', '', str(key).lower())


def resolve_sat_key(key: str, lookup: dict) -> str:
    """
    Resolve a user-supplied satellite key against a lookup dictionary,
    tolerating punctuation and case differences.

    An exact match is tried first; failing that, both the input and the
    dictionary keys are canonicalized (see :func:`_canonical_sat_key`) and
    compared. This means CoastWatch-style keys (e.g. ``'jason3'``) and
    CCI-style keys (e.g. ``'jason-3'``) are interchangeable for the caller.

    Parameters
    ----------
    key : str
        The user-supplied satellite key.
    lookup : dict
        The mapping of valid satellite keys (e.g. ``URL_TEMPLATES`` for
        CoastWatch or ``CCI_ALTIMETERS`` for CCI).

    Returns
    -------
    str
        The matching key *as it appears in* ``lookup``.

    Raises
    ------
    ValueError
        If no exact or canonical match is found.
    """
    if key in lookup:
        return key

    canon = _canonical_sat_key(key)
    for valid_key in lookup:
        if _canonical_sat_key(valid_key) == canon:
            return valid_key

    valid = sorted(lookup.keys())
    raise ValueError(
        f"Unknown satellite key: '{key}'.\n"
        f"Valid keys (any punctuation/case variant is accepted): {valid}"
    )


URL_TEMPLATES = {
    'sentinel3a': 'https://www.star.nesdis.noaa.gov/data/pub0010/lsa/johnk/coastwatch/3a/3a_',
    'sentinel3b': 'https://www.star.nesdis.noaa.gov/data/pub0010/lsa/johnk/coastwatch/3b/3b_',
    'sentinel6a': 'https://www.star.nesdis.noaa.gov/data/pub0010/lsa/johnk/coastwatch/6a/6a_',
    'jason2':     'https://www.star.nesdis.noaa.gov/data/pub0010/lsa/johnk/coastwatch/j2/j2_',
    'jason3':     'https://www.star.nesdis.noaa.gov/data/pub0010/lsa/johnk/coastwatch/j3/j3_',
    'cryosat2':   'https://www.star.nesdis.noaa.gov/data/pub0010/lsa/johnk/coastwatch/c2/c2_',
    'saral':      'https://www.star.nesdis.noaa.gov/data/pub0010/lsa/johnk/coastwatch/sa/sa_',
    'swot':       'https://www.star.nesdis.noaa.gov/data/pub0010/lsa/johnk/coastwatch/sw/sw_',
}

# Base URL for Argo data from Ifremer
# We will append /<region>/<year>/<month>/
ARGO_BASE_URL = "https://data-argo.ifremer.fr/geo"

# ESA CCI Sea State v5 - IFREMER FTP server
# Credentials are required. Register at: https://eftp.ifremer.fr
CCI_FTP_HOST = "eftp.ifremer.fr"
CCI_FTP_VERSION = "5"
CCI_FTP_BASE_PATH = "/products/v{version}/data/satellite"

# CCI altimeter satellite keys and their FTP directory names
CCI_ALTIMETERS = {
    'cfosat':                    'cfosat',
    'cryosat-2':                 'cryosat-2',
    'envisat':                   'envisat',
    'ers-1':                     'ers-1',
    'ers-2':                     'ers-2',
    'gfo':                       'gfo',
    'jason-1':                   'jason-1',
    'jason-2':                   'jason-2',
    'jason-3':                   'jason-3',
    'saral':                     'saral',
    'sentinel-3a':               'sentinel-3_a',
    'sentinel-3b':               'sentinel-3_b',
    'sentinel-6a':               'sentinel-6_a',
    'swot':                      'swot',
    'topex-poseidon_poseidon':   'topex-poseidon_poseidon',
    'topex-poseidon_topex':      'topex-poseidon_topex',
}

# CCI SAR satellite keys and their FTP directory names
CCI_SARS = {
    'envisat-sar':    'envisat',
    'sentinel-1a':   'sentinel-1a',
    'sentinel-1b':   'sentinel-1b',
    'sentinel-1c':   'sentinel-1c',
}

# Variables to retain from raw CCI files in the merged output
CCI_KEEP_VARS = [
    'swh',
    'swh_adjusted',
    'swh_with_8m_offset_correction',
    'swh_quality_level',
    'swh_uncertainty',
    'bathymetry',
    'distance_to_coast',
]
