"""Tests for the fetch_sources module."""

from src.ingestion.fetch_sources import guess_extension, slugify


class TestSlugify:
    def test_slugify_basic(self):
        assert slugify("Test Source") == "Test-Source"

    def test_slugify_url(self):
        assert slugify("https://donidcr.gov.np/page") == "donidcr.gov.np-page"

    def test_slugify_unicode(self):
        result = slugify("नेपाल सेवा")
        assert isinstance(result, str)


class TestGuessExtension:
    def test_html(self):
        assert guess_extension("text/html") == ".html"

    def test_pdf(self):
        assert guess_extension("application/pdf") == ".pdf"

    def test_unknown(self):
        assert guess_extension(None) == ".bin"
