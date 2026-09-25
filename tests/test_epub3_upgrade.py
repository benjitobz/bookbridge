"""Unit tests for the EPUB 2 -> EPUB 3 upgrade
(``src/services/epub3_upgrade.py``, a port of Storyteller's own upgrade --
see that module's docstring for the licence attribution and what was ported
vs. written fresh).

Cases mirror what Storyteller's own ``upgrade.test.ts``/``upgrade.ts`` cover
(identifiers, title, languages, authors with ``opf:role``, legacy
``<meta name=...>``, cover migration, guide -> landmarks, manifest
properties detection, nav href collision, NCX missing/empty), adapted to
this module's function-level (not whole-file-epubcheck) test shape since
this repository has no epubcheck binary available.
"""
import hashlib
import tempfile
import zipfile
from pathlib import Path

import pytest
from lxml import etree

from src.services.epub3_upgrade import (
    Landmark,
    NavigationItem,
    _find_opf_path,
    _looks_like_valid_epub3,
    _resolve_href_to_opf_dir,
    build_nav_document,
    choose_nav_href,
    collect_manifest_properties,
    extract_guide_landmarks,
    extract_ncx_toc,
    fix_font_mime_types,
    remove_guide,
    remove_invalid_dc_attrs,
    remove_spine_toc_ref,
    set_last_modified,
    set_package_version,
    upgrade_authors,
    upgrade_cover,
    upgrade_date,
    upgrade_identifiers,
    upgrade_languages,
    upgrade_meta,
    upgrade_package_metadata,
    upgrade_title,
    upgrade_epub2_to_epub3,
)

_OPF_NS = "http://www.idpf.org/2007/opf"
_DC_NS = "http://purl.org/dc/elements/1.1/"


def _pkg(metadata_xml: str = "", manifest_xml: str = "", spine_xml: str = "<spine/>",
         guide_xml: str = "", version: str = "2.0") -> etree._Element:
    xml = (
        f'<package xmlns="{_OPF_NS}" xmlns:opf="{_OPF_NS}" version="{version}" '
        'unique-identifier="id">'
        f'<metadata xmlns:dc="{_DC_NS}">{metadata_xml}</metadata>'
        f"<manifest>{manifest_xml}</manifest>"
        f"{spine_xml}"
        f"{guide_xml}"
        "</package>"
    )
    return etree.fromstring(xml.encode("utf-8"))


def _find(pkg, path):
    return pkg.find(path.format(ns=_OPF_NS))


def _metadata(pkg):
    return _find(pkg, "{{{ns}}}metadata")


def _manifest(pkg):
    return _find(pkg, "{{{ns}}}manifest")


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

class TestUpgradeIdentifiers:
    def test_urn_identifier_splits_scheme_and_value(self):
        pkg = _pkg(metadata_xml='<dc:identifier id="bid">urn:isbn:1234567890</dc:identifier>')
        upgrade_identifiers(pkg)
        ident = _metadata(pkg).find("{*}identifier")
        assert ident.text == "isbn:1234567890"
        assert ident.get("id") == "bid"
        assert len(ident.attrib) == 1

    def test_opf_scheme_attribute_prefixed_onto_value(self):
        pkg = _pkg(metadata_xml='<dc:identifier id="bid" opf:scheme="ISBN">1234567890</dc:identifier>')
        upgrade_identifiers(pkg)
        ident = _metadata(pkg).find("{*}identifier")
        assert ident.text == "ISBN:1234567890"

    def test_bare_scheme_attribute_without_opf_prefix_still_honored(self):
        """Some real-world (Calibre) writers omit the ``opf:`` prefix --
        this codebase's own series_metadata.py documents the same quirk."""
        pkg = _pkg(metadata_xml='<dc:identifier id="bid" scheme="ISBN">1234567890</dc:identifier>')
        upgrade_identifiers(pkg)
        ident = _metadata(pkg).find("{*}identifier")
        assert ident.text == "ISBN:1234567890"

    def test_uri_scheme_not_prefixed(self):
        pkg = _pkg(metadata_xml='<dc:identifier id="bid" opf:scheme="URI">http://example.com/book</dc:identifier>')
        upgrade_identifiers(pkg)
        ident = _metadata(pkg).find("{*}identifier")
        assert ident.text == "http://example.com/book"

    def test_publication_identifier_text_is_preserved(self):
        pkg = _pkg(metadata_xml='<dc:identifier id="id">urn:uuid:publication</dc:identifier>')
        upgrade_identifiers(pkg)
        ident = _metadata(pkg).find("{*}identifier")
        assert ident.text == "urn:uuid:publication"

    def test_anonymous_identifier_still_normalizes(self):
        pkg = _pkg(metadata_xml='<dc:identifier opf:scheme="ISBN">1234567890</dc:identifier>')
        pkg.attrib.pop("unique-identifier")
        upgrade_identifiers(pkg)
        ident = _metadata(pkg).find("{*}identifier")
        assert ident.text == "ISBN:1234567890"


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

class TestUpgradeTitle:
    def test_empty_titles_removed_first_kept_gets_title_type_meta(self):
        pkg = _pkg(metadata_xml="<dc:title>Real Title</dc:title><dc:title>   </dc:title>")
        upgrade_title(pkg)
        metadata = _metadata(pkg)
        titles = metadata.findall("{*}title")
        assert len(titles) == 1
        assert titles[0].text == "Real Title"
        title_id = titles[0].get("id")
        assert title_id
        metas = metadata.findall("{*}meta")
        assert any(
            m.get("refines") == f"#{title_id}" and m.get("property") == "title-type" and m.text == "main"
            for m in metas
        )

    def test_no_titles_at_all_is_a_noop(self):
        pkg = _pkg(metadata_xml="")
        upgrade_title(pkg)
        assert _metadata(pkg).findall("{*}title") == []
        assert _metadata(pkg).findall("{*}meta") == []


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

class TestUpgradeLanguages:
    def test_existing_language_kept_stripped_of_extra_attrs(self):
        pkg = _pkg(metadata_xml='<dc:language opf:garbage="x">en</dc:language>')
        upgrade_languages(pkg)
        lang = _metadata(pkg).find("{*}language")
        assert lang.text == "en"
        assert dict(lang.attrib) == {}

    def test_missing_language_defaults_to_und(self):
        pkg = _pkg(metadata_xml="")
        upgrade_languages(pkg)
        lang = _metadata(pkg).find("{*}language")
        assert lang is not None
        assert lang.text == "und"


# ---------------------------------------------------------------------------
# Authors (creator/contributor with opf:role / opf:file-as)
# ---------------------------------------------------------------------------

class TestUpgradeAuthors:
    def test_role_and_file_as_become_refines_metas(self):
        pkg = _pkg(metadata_xml='<dc:creator opf:role="aut" opf:file-as="Doe, Jane">Jane Doe</dc:creator>')
        upgrade_authors(pkg)
        metadata = _metadata(pkg)
        creator = metadata.find("{*}creator")
        assert creator.text == "Jane Doe"
        creator_id = creator.get("id")
        assert creator_id
        assert len(creator.attrib) == 1  # only id left
        metas = metadata.findall("{*}meta")
        assert any(
            m.get("refines") == f"#{creator_id}" and m.get("property") == "role"
            and m.get("scheme") == "marc:relators" and m.text == "aut"
            for m in metas
        )
        assert any(
            m.get("refines") == f"#{creator_id}" and m.get("property") == "file-as" and m.text == "Doe, Jane"
            for m in metas
        )

    def test_bare_role_attribute_without_opf_prefix_still_migrated(self):
        pkg = _pkg(metadata_xml='<dc:contributor role="edt">Bare Editor</dc:contributor>')
        upgrade_authors(pkg)
        metadata = _metadata(pkg)
        contributor = metadata.find("{*}contributor")
        contributor_id = contributor.get("id")
        assert contributor_id
        assert any(
            m.get("property") == "role" and m.text == "edt" and m.get("refines") == f"#{contributor_id}"
            for m in metadata.findall("{*}meta")
        )

    def test_no_role_or_file_as_leaves_no_new_id_or_meta(self):
        pkg = _pkg(metadata_xml="<dc:creator>Plain Author</dc:creator>")
        upgrade_authors(pkg)
        metadata = _metadata(pkg)
        creator = metadata.find("{*}creator")
        assert creator.get("id") is None
        assert metadata.findall("{*}meta") == []


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

class TestUpgradeDate:
    def test_only_first_nonempty_date_kept(self):
        pkg = _pkg(metadata_xml="<dc:date>2020-01-01</dc:date><dc:date>2021-01-01</dc:date>")
        upgrade_date(pkg)
        dates = _metadata(pkg).findall("{*}date")
        assert len(dates) == 1
        assert dates[0].text == "2020-01-01"

    def test_empty_date_dropped(self):
        pkg = _pkg(metadata_xml="<dc:date></dc:date>")
        upgrade_date(pkg)
        assert _metadata(pkg).findall("{*}date") == []


# ---------------------------------------------------------------------------
# Legacy <meta name=...> rewriting
# ---------------------------------------------------------------------------

class TestUpgradeMeta:
    def test_fixed_layout_true_becomes_pre_paginated(self):
        pkg = _pkg(metadata_xml='<meta name="fixed-layout" content="true"/>')
        upgrade_meta(pkg)
        meta = _metadata(pkg).find("{*}meta")
        assert meta.get("property") == "rendition:layout"
        assert meta.get("name") is None
        assert meta.text == "pre-paginated"

    def test_fixed_layout_false_becomes_reflowable(self):
        pkg = _pkg(metadata_xml='<meta name="fixed-layout" content="false"/>')
        upgrade_meta(pkg)
        meta = _metadata(pkg).find("{*}meta")
        assert meta.text == "reflowable"

    def test_orientation_lock_maps_known_values(self):
        pkg = _pkg(metadata_xml='<meta name="orientation-lock" content="Landscape"/>')
        upgrade_meta(pkg)
        meta = _metadata(pkg).find("{*}meta")
        assert meta.get("property") == "rendition:orientation"
        assert meta.text == "landscape"

    def test_orientation_lock_unknown_value_falls_back_to_auto(self):
        pkg = _pkg(metadata_xml='<meta name="orientation-lock" content="sideways"/>')
        upgrade_meta(pkg)
        meta = _metadata(pkg).find("{*}meta")
        assert meta.text == "auto"

    def test_rendition_prefixed_name_recognized(self):
        pkg = _pkg(metadata_xml='<meta name="rendition:spread" content="both"/>')
        upgrade_meta(pkg)
        meta = _metadata(pkg).find("{*}meta")
        assert meta.get("property") == "rendition:spread"
        assert meta.text == "both"

    def test_unrelated_meta_left_untouched(self):
        pkg = _pkg(metadata_xml='<meta name="calibre:series" content="My Series"/>')
        upgrade_meta(pkg)
        meta = _metadata(pkg).find("{*}meta")
        assert meta.get("name") == "calibre:series"
        assert meta.get("content") == "My Series"
        assert meta.get("property") is None


# ---------------------------------------------------------------------------
# Cover migration
# ---------------------------------------------------------------------------

class TestUpgradeCover:
    def test_cover_meta_migrates_properties_to_image_item(self):
        pkg = _pkg(
            metadata_xml='<meta name="cover" content="cover-img"/>',
            manifest_xml='<item id="cover-img" href="cover.jpg" media-type="image/jpeg"/>',
        )
        upgrade_cover(pkg)
        item = _manifest(pkg).find("{*}item")
        assert item.get("properties") == "cover-image"

    def test_cover_meta_pointing_at_xhtml_wrapper_is_ignored(self):
        pkg = _pkg(
            metadata_xml='<meta name="cover" content="cover-page"/>',
            manifest_xml='<item id="cover-page" href="cover.xhtml" media-type="application/xhtml+xml"/>',
        )
        upgrade_cover(pkg)
        item = _manifest(pkg).find("{*}item")
        assert item.get("properties") is None

    def test_existing_properties_preserved_alongside_cover_image(self):
        pkg = _pkg(
            metadata_xml='<meta name="cover" content="cover-img"/>',
            manifest_xml='<item id="cover-img" href="cover.jpg" media-type="image/jpeg" properties="mathml"/>',
        )
        upgrade_cover(pkg)
        item = _manifest(pkg).find("{*}item")
        assert set(item.get("properties").split()) == {"cover-image", "mathml"}


# ---------------------------------------------------------------------------
# remove_invalid_dc_attrs / set_last_modified / set_package_version
# ---------------------------------------------------------------------------

class TestRemoveInvalidDcAttrs:
    def test_strips_opf_attrs_from_untouched_dc_elements(self):
        pkg = _pkg(metadata_xml='<dc:publisher opf:extra="junk" id="pub">Acme</dc:publisher>')
        remove_invalid_dc_attrs(pkg)
        publisher = _metadata(pkg).find("{*}publisher")
        assert publisher.get("id") == "pub"
        assert publisher.get("extra") is None
        assert len(publisher.attrib) == 1

    def test_non_dc_elements_untouched(self):
        pkg = _pkg(metadata_xml='<meta name="cover" content="x" extra="keep"/>')
        remove_invalid_dc_attrs(pkg)
        meta = _metadata(pkg).find("{*}meta")
        assert meta.get("extra") == "keep"


class TestSetLastModified:
    def test_adds_dcterms_modified(self):
        pkg = _pkg(metadata_xml="")
        set_last_modified(pkg)
        metas = _metadata(pkg).findall("{*}meta")
        assert any(m.get("property") == "dcterms:modified" and m.text for m in metas)

    def test_replaces_existing_dcterms_modified_rather_than_duplicating(self):
        pkg = _pkg(metadata_xml='<meta property="dcterms:modified">2000-01-01T00:00:00Z</meta>')
        set_last_modified(pkg)
        metas = _metadata(pkg).findall("{*}meta")
        modified = [m for m in metas if m.get("property") == "dcterms:modified"]
        assert len(modified) == 1
        assert modified[0].text != "2000-01-01T00:00:00Z"


def test_upgrade_package_metadata_runs_every_step_in_order():
    """Smoke test that the composed function performs each individual
    upgrade -- not a re-test of each one's own behavior."""
    pkg = _pkg(metadata_xml=(
        '<dc:identifier id="bid">urn:isbn:111</dc:identifier>'
        "<dc:title>A Book</dc:title>"
    ))
    upgrade_package_metadata(pkg)
    metadata = _metadata(pkg)
    assert metadata.find("{*}identifier").text == "isbn:111"
    assert metadata.find("{*}language") is not None  # defaulted
    assert any(m.get("property") == "dcterms:modified" for m in metadata.findall("{*}meta"))


class TestSetPackageVersion:
    def test_sets_version_attribute(self):
        pkg = _pkg()
        set_package_version(pkg, "3.0")
        assert pkg.get("version") == "3.0"


# ---------------------------------------------------------------------------
# Guide -> landmarks / spine toc / guide removal
# ---------------------------------------------------------------------------

class TestGuideLandmarks:
    def test_known_guide_types_mapped(self):
        pkg = _pkg(guide_xml=(
            "<guide>"
            '<reference type="cover" title="Cover" href="cover.xhtml"/>'
            '<reference type="toc" title="Contents" href="toc.xhtml"/>'
            "</guide>"
        ))
        landmarks = extract_guide_landmarks(pkg)
        assert landmarks == [
            Landmark(href="cover.xhtml", title="Cover", type="cover"),
            Landmark(href="toc.xhtml", title="Contents", type="toc"),
        ]

    def test_unknown_guide_type_dropped(self):
        pkg = _pkg(guide_xml='<guide><reference type="totally-unknown" title="X" href="x.xhtml"/></guide>')
        assert extract_guide_landmarks(pkg) == []

    def test_notes_type_kept_with_empty_epub_type(self):
        """Ported verbatim from Storyteller: "notes" is a recognized guide
        type with NO EPUB 3 landmark equivalent -- kept here, filtered out
        later at nav-build time."""
        pkg = _pkg(guide_xml='<guide><reference type="notes" title="Notes" href="notes.xhtml"/></guide>')
        landmarks = extract_guide_landmarks(pkg)
        assert landmarks == [Landmark(href="notes.xhtml", title="Notes", type="")]

    def test_no_guide_element_returns_empty(self):
        pkg = _pkg()
        assert extract_guide_landmarks(pkg) == []

    def test_remove_guide_removes_element(self):
        pkg = _pkg(guide_xml='<guide><reference type="cover" href="c.xhtml"/></guide>')
        remove_guide(pkg)
        assert pkg.find("{*}guide") is None

    def test_remove_guide_noop_when_absent(self):
        pkg = _pkg()
        remove_guide(pkg)  # must not raise
        assert pkg.find("{*}guide") is None


class TestRemoveSpineTocRef:
    def test_removes_toc_attribute(self):
        pkg = _pkg(spine_xml='<spine toc="ncx"/>')
        remove_spine_toc_ref(pkg)
        assert "toc" not in pkg.find("{*}spine").attrib

    def test_noop_when_no_toc_attribute(self):
        pkg = _pkg(spine_xml="<spine/>")
        remove_spine_toc_ref(pkg)  # must not raise
        assert "toc" not in pkg.find("{*}spine").attrib


# ---------------------------------------------------------------------------
# Manifest properties detection (svg/scripted/mathml/switch)
# ---------------------------------------------------------------------------

class TestCollectManifestProperties:
    def _zip_with(self, tmp_path: Path, contents: dict) -> Path:
        path = tmp_path / "book.epub"
        with zipfile.ZipFile(path, "w") as z:
            for name, data in contents.items():
                z.writestr(name, data)
        return path

    def test_svg_and_script_detected(self, tmp_path):
        pkg = _pkg(manifest_xml=(
            '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>'
        ))
        epub_path = self._zip_with(tmp_path, {
            "ch1.xhtml": b"<html><body><svg></svg></body></html>",
            "ch2.xhtml": b"<html><body><script>1</script></body></html>",
        })
        with zipfile.ZipFile(epub_path) as zf:
            collect_manifest_properties(pkg, "", zf)
        items = {item.get("id"): item for item in _manifest(pkg).findall("{*}item")}
        assert items["ch1"].get("properties") == "svg"
        assert items["ch2"].get("properties") == "scripted"

    def test_mathml_and_epub_switch_detected(self, tmp_path):
        pkg = _pkg(manifest_xml='<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>')
        epub_path = self._zip_with(tmp_path, {
            "ch1.xhtml": b"<html><body><math></math><epub:switch></epub:switch></body></html>",
        })
        with zipfile.ZipFile(epub_path) as zf:
            collect_manifest_properties(pkg, "", zf)
        item = _manifest(pkg).find("{*}item")
        assert set(item.get("properties").split()) == {"mathml", "switch"}

    def test_plain_xhtml_gets_no_properties(self, tmp_path):
        pkg = _pkg(manifest_xml='<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>')
        epub_path = self._zip_with(tmp_path, {"ch1.xhtml": b"<html><body><p>Plain.</p></body></html>"})
        with zipfile.ZipFile(epub_path) as zf:
            collect_manifest_properties(pkg, "", zf)
        item = _manifest(pkg).find("{*}item")
        assert item.get("properties") is None

    def test_non_xhtml_manifest_items_skipped(self, tmp_path):
        pkg = _pkg(manifest_xml='<item id="img" href="cover.jpg" media-type="image/jpeg"/>')
        epub_path = self._zip_with(tmp_path, {"cover.jpg": b"\xff\xd8\xff"})
        with zipfile.ZipFile(epub_path) as zf:
            collect_manifest_properties(pkg, "", zf)  # must not raise / not try to parse jpeg
        item = _manifest(pkg).find("{*}item")
        assert item.get("properties") is None

    def test_existing_properties_preserved(self, tmp_path):
        pkg = _pkg(manifest_xml=(
            '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        ))
        epub_path = self._zip_with(tmp_path, {"ch1.xhtml": b"<html><body><svg></svg></body></html>"})
        with zipfile.ZipFile(epub_path) as zf:
            collect_manifest_properties(pkg, "", zf)
        item = _manifest(pkg).find("{*}item")
        assert set(item.get("properties").split()) == {"nav", "svg"}


# ---------------------------------------------------------------------------
# Font MIME type fix
# ---------------------------------------------------------------------------

class TestFixFontMimeTypes:
    def test_legacy_ttf_mime_corrected(self):
        pkg = _pkg(manifest_xml='<item id="f" href="fonts/a.ttf" media-type="application/x-font-ttf"/>')
        fix_font_mime_types(pkg)
        assert _manifest(pkg).find("{*}item").get("media-type") == "font/ttf"

    def test_already_correct_mime_left_alone(self):
        pkg = _pkg(manifest_xml='<item id="f" href="fonts/a.woff2" media-type="font/woff2"/>')
        fix_font_mime_types(pkg)
        assert _manifest(pkg).find("{*}item").get("media-type") == "font/woff2"

    def test_non_font_item_untouched(self):
        pkg = _pkg(manifest_xml='<item id="c" href="ch1.xhtml" media-type="application/xhtml+xml"/>')
        fix_font_mime_types(pkg)
        assert _manifest(pkg).find("{*}item").get("media-type") == "application/xhtml+xml"


# ---------------------------------------------------------------------------
# Nav href collision avoidance
# ---------------------------------------------------------------------------

class TestChooseNavHref:
    def test_default_nav_xhtml_when_no_collision(self):
        pkg = _pkg(manifest_xml='<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>')
        assert choose_nav_href(pkg) == "nav.xhtml"

    def test_collision_bumps_to_nav1(self):
        pkg = _pkg(manifest_xml='<item id="existing" href="nav.xhtml" media-type="application/xhtml+xml"/>')
        assert choose_nav_href(pkg) == "nav1.xhtml"

    def test_multiple_collisions_keep_bumping(self):
        pkg = _pkg(manifest_xml=(
            '<item id="a" href="nav.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="b" href="nav1.xhtml" media-type="application/xhtml+xml"/>'
        ))
        assert choose_nav_href(pkg) == "nav2.xhtml"


# ---------------------------------------------------------------------------
# NCX parsing: missing / empty / nested
# ---------------------------------------------------------------------------

class TestExtractNcxToc:
    def _epub_with_ncx(self, tmp_path: Path, ncx_bytes: bytes, opf_extra_manifest: str = "") -> Path:
        path = tmp_path / "book.epub"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("toc.ncx", ncx_bytes)
        return path

    def test_no_ncx_manifest_item_returns_empty(self, tmp_path):
        pkg = _pkg(manifest_xml='<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>')
        epub_path = tmp_path / "book.epub"
        with zipfile.ZipFile(epub_path, "w") as z:
            z.writestr("ch1.xhtml", b"<html/>")
        with zipfile.ZipFile(epub_path) as zf:
            assert extract_ncx_toc(pkg, "", zf) == []

    def test_ncx_item_present_but_missing_from_archive_returns_empty(self, tmp_path):
        pkg = _pkg(manifest_xml='<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
        epub_path = tmp_path / "book.epub"
        with zipfile.ZipFile(epub_path, "w") as z:
            z.writestr("placeholder", b"x")
        with zipfile.ZipFile(epub_path) as zf:
            assert extract_ncx_toc(pkg, "", zf) == []

    def test_ncx_with_empty_navmap_returns_empty(self, tmp_path):
        pkg = _pkg(manifest_xml='<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
        ncx = b'<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap/></ncx>'
        epub_path = self._epub_with_ncx(tmp_path, ncx)
        with zipfile.ZipFile(epub_path) as zf:
            assert extract_ncx_toc(pkg, "", zf) == []

    def test_ncx_malformed_xml_returns_empty_not_raises(self, tmp_path):
        pkg = _pkg(manifest_xml='<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
        epub_path = self._epub_with_ncx(tmp_path, b"<ncx><navMap>not closed")
        with zipfile.ZipFile(epub_path) as zf:
            assert extract_ncx_toc(pkg, "", zf) == []

    def test_nested_navpoints_and_fragment_hrefs(self, tmp_path):
        pkg = _pkg(
            spine_xml='<spine toc="ncx"/>',
            manifest_xml='<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        )
        ncx = (
            b'<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">'
            b"<navMap>"
            b'<navPoint id="n1"><navLabel><text>Chapter 1</text></navLabel>'
            b'<content src="ch1.xhtml"/></navPoint>'
            b'<navPoint id="n2"><navLabel><text>Chapter 2</text></navLabel>'
            b'<content src="ch2.xhtml#frag"/>'
            b'<navPoint id="n2a"><navLabel><text>Section</text></navLabel>'
            b'<content src="ch2.xhtml#s1"/></navPoint>'
            b"</navPoint>"
            b"</navMap></ncx>"
        )
        epub_path = self._epub_with_ncx(tmp_path, ncx)
        with zipfile.ZipFile(epub_path) as zf:
            entries = extract_ncx_toc(pkg, "", zf)
        assert len(entries) == 2
        assert entries[0] == NavigationItem(title="Chapter 1", href="ch1.xhtml", children=[])
        assert entries[1].title == "Chapter 2"
        assert entries[1].href == "ch2.xhtml#frag"
        assert len(entries[1].children) == 1
        assert entries[1].children[0] == NavigationItem(title="Section", href="ch2.xhtml#s1", children=[])

    def test_navpoint_with_no_label_falls_back_to_numeric_title(self, tmp_path):
        pkg = _pkg(manifest_xml='<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
        ncx = (
            b'<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">'
            b'<navMap><navPoint id="n1"><content src="ch1.xhtml"/></navPoint></navMap></ncx>'
        )
        epub_path = self._epub_with_ncx(tmp_path, ncx)
        with zipfile.ZipFile(epub_path) as zf:
            entries = extract_ncx_toc(pkg, "", zf)
        assert entries[0].title == "0"

    def test_ncx_in_subdirectory_href_rerooted_to_opf_dir(self, tmp_path):
        """The NCX lives in OEBPS/ alongside the OPF, but content lives one
        level deeper (OEBPS/text/) -- resolveHref's climb-then-descend must
        express the href relative to the OPF's own directory (also OEBPS),
        matching where the nav document will actually be written."""
        pkg = _pkg(manifest_xml='<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
        ncx = (
            b'<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">'
            b'<navMap><navPoint id="n1"><navLabel><text>C1</text></navLabel>'
            b'<content src="text/ch1.xhtml"/></navPoint></navMap></ncx>'
        )
        path = tmp_path / "book.epub"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("OEBPS/toc.ncx", ncx)
        with zipfile.ZipFile(path) as zf:
            entries = extract_ncx_toc(pkg, "OEBPS", zf)
        assert entries[0].href == "text/ch1.xhtml"


class TestResolveHrefToOpfDir:
    def test_same_directory_is_identity(self):
        assert _resolve_href_to_opf_dir("OEBPS/toc.ncx", "ch1.xhtml", "OEBPS") == "ch1.xhtml"

    def test_ncx_in_subdirectory_climbs_out(self):
        assert _resolve_href_to_opf_dir("OEBPS/ncx/toc.ncx", "../text/ch1.xhtml", "OEBPS") == "text/ch1.xhtml"

    def test_fragment_preserved(self):
        assert _resolve_href_to_opf_dir("OEBPS/toc.ncx", "ch1.xhtml#s1", "OEBPS") == "ch1.xhtml#s1"

    def test_bare_fragment_returned_unchanged(self):
        assert _resolve_href_to_opf_dir("OEBPS/toc.ncx", "#s1", "OEBPS") == "#s1"


# ---------------------------------------------------------------------------
# Nav document assembly
# ---------------------------------------------------------------------------

class TestBuildNavDocument:
    def test_toc_entries_rendered_as_nested_ol(self):
        entries = [
            NavigationItem(title="Chapter 1", href="ch1.xhtml", children=[]),
            NavigationItem(title="Chapter 2", href="ch2.xhtml", children=[
                NavigationItem(title="Section", href="ch2.xhtml#s1", children=[]),
            ]),
        ]
        doc = build_nav_document(entries, [], "ch1.xhtml")
        root = etree.fromstring(doc)
        toc_nav = root.find(".//{http://www.w3.org/1999/xhtml}nav")
        assert toc_nav.get("{http://www.idpf.org/2007/ops}type") == "toc"
        links = toc_nav.findall(".//{http://www.w3.org/1999/xhtml}a")
        assert [a.get("href") for a in links] == ["ch1.xhtml", "ch2.xhtml", "ch2.xhtml#s1"]

    def test_empty_toc_falls_back_to_single_start_entry(self):
        doc = build_nav_document([], [], "ch1.xhtml")
        root = etree.fromstring(doc)
        links = root.findall(".//{http://www.w3.org/1999/xhtml}a")
        assert len(links) == 1
        assert links[0].get("href") == "ch1.xhtml"
        assert links[0].text == "Start"

    def test_landmarks_rendered_when_present(self):
        landmarks = [Landmark(href="cover.xhtml", title="Cover", type="cover")]
        doc = build_nav_document([], landmarks, "ch1.xhtml")
        root = etree.fromstring(doc)
        navs = root.findall(".//{http://www.w3.org/1999/xhtml}nav")
        assert len(navs) == 2
        landmarks_nav = [n for n in navs if n.get("{http://www.idpf.org/2007/ops}type") == "landmarks"][0]
        assert landmarks_nav.get("hidden") == ""

    def test_landmarks_nav_omitted_when_only_typeless_entries(self):
        landmarks = [Landmark(href="notes.xhtml", title="Notes", type="")]
        doc = build_nav_document([], landmarks, "ch1.xhtml")
        root = etree.fromstring(doc)
        navs = root.findall(".//{http://www.w3.org/1999/xhtml}nav")
        assert len(navs) == 1  # toc only

    def test_valid_xhtml_declares_epub_ops_namespace(self):
        doc = build_nav_document([], [], "ch1.xhtml")
        assert b'xmlns:epub="http://www.idpf.org/2007/ops"' in doc
        assert b"<!DOCTYPE html>" in doc


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------

_CONTAINER_XML = (
    '<?xml version="1.0"?><container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
    '<rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles></container>'
)


def _write_epub2_fixture(path: Path, opf_xml: str, files: dict) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", _CONTAINER_XML)
        z.writestr("OEBPS/content.opf", opf_xml)
        for name, data in files.items():
            z.writestr(f"OEBPS/{name}", data)


_MINIMAL_OPF = (
    f'<?xml version="1.0"?><package xmlns="{_OPF_NS}" version="2.0" '
    f'unique-identifier="id"><metadata xmlns:dc="{_DC_NS}">'
    "<dc:title>Minimal</dc:title>"
    '<dc:identifier id="id">urn:uuid:test</dc:identifier></metadata>'
    '<manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>'
    '<spine><itemref idref="ch1"/></spine></package>'
)


class TestUpgradeEpub2ToEpub3:
    def test_minimal_epub_converts_successfully(self, tmp_path):
        src = tmp_path / "book.epub"
        _write_epub2_fixture(src, _MINIMAL_OPF, {"ch1.xhtml": b"<html><body><p>Hi.</p></body></html>"})
        out = tmp_path / "out.epub"
        result = upgrade_epub2_to_epub3(src, out)
        assert result is not None
        assert result.nav_href == "nav.xhtml"
        with zipfile.ZipFile(out) as zf:
            pkg = etree.fromstring(zf.read("OEBPS/content.opf"))
            assert pkg.get("version") == "3.0"
            assert zf.read("OEBPS/nav.xhtml")

    def test_source_file_is_byte_identical_after_conversion(self, tmp_path):
        src = tmp_path / "book.epub"
        _write_epub2_fixture(src, _MINIMAL_OPF, {"ch1.xhtml": b"<html><body><p>Hi.</p></body></html>"})
        original = src.read_bytes()
        out = tmp_path / "out.epub"
        result = upgrade_epub2_to_epub3(src, out)
        assert result is not None
        assert src.read_bytes() == original

    def test_spine_content_bytes_are_untouched(self, tmp_path):
        """Load-bearing: an alignment map fitted against the original spine
        bytes must remain valid against the converted copy."""
        src = tmp_path / "book.epub"
        spine_bytes = b"<html><body><p>Alpha bravo charlie.</p></body></html>"
        _write_epub2_fixture(src, _MINIMAL_OPF, {"ch1.xhtml": spine_bytes})
        out = tmp_path / "out.epub"
        upgrade_epub2_to_epub3(src, out)
        with zipfile.ZipFile(out) as zf:
            assert zf.read("OEBPS/ch1.xhtml") == spine_bytes

    def test_obfuscated_font_still_decodes_with_preserved_identifier(self, tmp_path):
        unique_identifier = "urn:uuid:font-publication"
        font_plain = bytes(range(256)) * 5
        key = hashlib.sha1(unique_identifier.encode("utf-8")).digest()
        font_obfuscated = bytes(
            value ^ key[index % len(key)] if index < 1040 else value
            for index, value in enumerate(font_plain)
        )
        encryption_xml = (
            b'<encryption xmlns="http://www.w3.org/2001/04/xmlenc#">'
            b"<EncryptedData>"
            b'<EncryptionMethod Algorithm="http://www.idpf.org/2008/embedding"/>'
            b"<CipherData><CipherReference URI=\"OEBPS/fonts/font.ttf\"/>"
            b"</CipherData></EncryptedData></encryption>"
        )
        opf = (
            f'<?xml version="1.0"?><package xmlns="{_OPF_NS}" version="2.0" '
            f'unique-identifier="id"><metadata xmlns:dc="{_DC_NS}">'
            "<dc:title>Font</dc:title>"
            f'<dc:identifier id="id">{unique_identifier}</dc:identifier></metadata>'
            '<manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="font" href="fonts/font.ttf" media-type="application/x-font-ttf"/></manifest>'
            '<spine><itemref idref="ch1"/></spine></package>'
        )
        src = tmp_path / "book.epub"
        with zipfile.ZipFile(src, "w") as z:
            z.writestr("mimetype", "application/epub+zip")
            z.writestr("META-INF/container.xml", _CONTAINER_XML)
            z.writestr("META-INF/encryption.xml", encryption_xml)
            z.writestr("OEBPS/content.opf", opf)
            z.writestr("OEBPS/ch1.xhtml", b"<html><body><p>Font.</p></body></html>")
            z.writestr("OEBPS/fonts/font.ttf", font_obfuscated)

        out = tmp_path / "out.epub"
        assert upgrade_epub2_to_epub3(src, out) is not None
        with zipfile.ZipFile(out) as z:
            output_pkg = etree.fromstring(z.read("OEBPS/content.opf"))
            output_identifier = output_pkg.find("{*}metadata/{*}identifier").text
            assert output_identifier == unique_identifier
            assert z.read("META-INF/encryption.xml") == encryption_xml
            output_key = hashlib.sha1(output_identifier.encode("utf-8")).digest()
            decoded = bytes(
                value ^ output_key[index % len(output_key)] if index < 1040 else value
                for index, value in enumerate(z.read("OEBPS/fonts/font.ttf"))
            )
            assert decoded == font_plain

    def test_missing_manifest_refuses(self, tmp_path):
        broken_opf = (
            f'<?xml version="1.0"?><package xmlns="{_OPF_NS}" version="2.0" '
            f'unique-identifier="id"><metadata xmlns:dc="{_DC_NS}">'
            "<dc:title>Broken</dc:title></metadata></package>"
        )
        src = tmp_path / "book.epub"
        _write_epub2_fixture(src, broken_opf, {})
        out = tmp_path / "out.epub"
        assert upgrade_epub2_to_epub3(src, out) is None
        assert not out.exists()

    def test_missing_opf_refuses(self, tmp_path):
        src = tmp_path / "book.epub"
        with zipfile.ZipFile(src, "w") as z:
            z.writestr("mimetype", "application/epub+zip")
            z.writestr("META-INF/container.xml", _CONTAINER_XML)
        out = tmp_path / "out.epub"
        assert upgrade_epub2_to_epub3(src, out) is None

    def test_malformed_opf_xml_refuses(self, tmp_path):
        src = tmp_path / "book.epub"
        _write_epub2_fixture(src, "<package version=2.0 not-well-formed", {})
        out = tmp_path / "out.epub"
        assert upgrade_epub2_to_epub3(src, out) is None

    def test_already_epub3_source_and_output_alias_raises(self, tmp_path):
        """Same-path aliasing must refuse loudly rather than truncate the
        only copy of the file (mirrors readalong_builder._package_epub's
        Finding 5 protection)."""
        src = tmp_path / "book.epub"
        _write_epub2_fixture(src, _MINIMAL_OPF, {"ch1.xhtml": b"<html/>"})
        with pytest.raises(ValueError):
            upgrade_epub2_to_epub3(src, src)

    def test_guide_and_ncx_produce_populated_nav(self, tmp_path):
        opf = (
            f'<?xml version="1.0"?><package xmlns="{_OPF_NS}" version="2.0" '
            f'unique-identifier="id"><metadata xmlns:dc="{_DC_NS}">'
            "<dc:title>With Guide</dc:title>"
            '<dc:identifier id="id">urn:uuid:test</dc:identifier></metadata>'
            '<manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/></manifest>'
            '<spine toc="ncx"><itemref idref="ch1"/></spine>'
            '<guide><reference type="cover" title="Cover" href="ch1.xhtml"/></guide></package>'
        )
        ncx = (
            b'<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">'
            b'<navMap><navPoint id="n1"><navLabel><text>Chapter 1</text></navLabel>'
            b'<content src="ch1.xhtml"/></navPoint></navMap></ncx>'
        )
        src = tmp_path / "book.epub"
        _write_epub2_fixture(src, opf, {"ch1.xhtml": b"<html><body><p>Hi.</p></body></html>", "toc.ncx": ncx})
        out = tmp_path / "out.epub"
        result = upgrade_epub2_to_epub3(src, out)
        assert result is not None
        assert result.toc_entry_count == 1
        assert result.landmark_count == 1
        with zipfile.ZipFile(out) as zf:
            pkg = etree.fromstring(zf.read("OEBPS/content.opf"))
            assert pkg.find("{*}guide") is None
            assert "toc" not in pkg.find("{*}spine").attrib
            nav_bytes = zf.read(f"OEBPS/{result.nav_href}")
            assert b"Chapter 1" in nav_bytes
            assert b'epub:type="cover"' in nav_bytes

    def test_already_epub3_input_is_left_alone_by_looks_like_valid_check(self):
        """Direct unit check on the self-check gate this module runs before
        writing anything -- not exercised end to end here (see
        readalong_builder's own EPUB3-noop path in _resolve_epub3_source)."""
        pkg = _pkg(
            manifest_xml='<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
            version="3.0",
        )
        opf_bytes = etree.tostring(pkg)
        assert _looks_like_valid_epub3(opf_bytes) is True

    def test_looks_like_valid_epub3_false_without_nav_item(self):
        pkg = _pkg(version="3.0")
        assert _looks_like_valid_epub3(etree.tostring(pkg)) is False

    def test_looks_like_valid_epub3_false_for_epub2_version(self):
        pkg = _pkg(
            manifest_xml='<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
            version="2.0",
        )
        assert _looks_like_valid_epub3(etree.tostring(pkg)) is False


def test_find_opf_path_reads_container_xml():
    with tempfile.TemporaryDirectory() as tmp_str:
        path = Path(tmp_str) / "book.epub"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("META-INF/container.xml", _CONTAINER_XML)
        with zipfile.ZipFile(path) as zf:
            assert _find_opf_path(zf) == "OEBPS/content.opf"


def test_find_opf_path_missing_container_returns_none():
    with tempfile.TemporaryDirectory() as tmp_str:
        path = Path(tmp_str) / "book.epub"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("placeholder", b"x")
        with zipfile.ZipFile(path) as zf:
            assert _find_opf_path(zf) is None


def test_find_opf_path_single_quoted_full_path_resolves():
    """Defect 3 (independent review): the previous implementation
    regex-matched only a double-quoted ``full-path="..."`` attribute, so a
    perfectly valid container.xml using single quotes (equally legal XML)
    was rejected as having no OPF at all -- blocking EPUB 2 -> EPUB 3
    conversion outright for such a book. Parsing as real XML (this test's
    fix) has no such quote-style sensitivity."""
    single_quoted_container = (
        "<?xml version='1.0'?><container version='1.0' "
        "xmlns='urn:oasis:names:tc:opendocument:xmlns:container'><rootfiles>"
        "<rootfile full-path='OEBPS/content.opf' "
        "media-type='application/oebps-package+xml'/></rootfiles></container>"
    )
    with tempfile.TemporaryDirectory() as tmp_str:
        path = Path(tmp_str) / "book.epub"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("META-INF/container.xml", single_quoted_container)
        with zipfile.ZipFile(path) as zf:
            assert _find_opf_path(zf) == "OEBPS/content.opf"


def test_find_opf_path_xml_escaped_full_path_resolves():
    """Defect 3's other half: the previous regex captured the attribute's
    raw, still-escaped text verbatim, so a path containing a character that
    must be XML-escaped in an attribute value (e.g. ``&`` -> ``&amp;``) came
    back with the escape sequence still in it -- a path that does not exist
    in the archive. Real XML parsing decodes it."""
    escaped_container = (
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="OEBPS/Books &amp; Beyond/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    with tempfile.TemporaryDirectory() as tmp_str:
        path = Path(tmp_str) / "book.epub"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("META-INF/container.xml", escaped_container)
        with zipfile.ZipFile(path) as zf:
            assert _find_opf_path(zf) == "OEBPS/Books & Beyond/content.opf"
