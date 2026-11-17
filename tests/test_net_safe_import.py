"""PLI 5.6.3: HTTP(S)-only URL opening (pli.net) and imports limited to known packages (pli.safe_import)."""

import urllib.error

import pytest

from pli import net, safe_import


@pytest.mark.parametrize("url", ["file:///etc/hosts", "ftp://example.invalid/x", "data:text/plain,hello"])
def test_other_schemes_are_refused(url):
    with pytest.raises(urllib.error.URLError):
        net.urlopen(url, timeout=2)


def test_http_and_https_handlers_are_there():
    names = {type(h).__name__ for h in net._OPENER.handlers}
    assert {"HTTPHandler", "HTTPSHandler", "HTTPRedirectHandler"}   # ProxyHandler is kept only when a proxy is set <= names
    assert not names & {"FileHandler", "FTPHandler", "DataHandler"}


def test_a_known_package_is_imported_and_others_refused():
    assert safe_import.import_module("pli.net") is net
    assert safe_import.import_module("json", ("json",)).__name__ == "json"
    for name in ("os", "subprocess", "plix.y", "", None):
        with pytest.raises(ImportError):
            safe_import.import_module(name)
