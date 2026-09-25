"""EPUB 2 -> EPUB 3 upgrade: package metadata, navigation document, and
manifest properties, so read-along generation (``readalong_builder.py``) is
no longer limited to the minority of library books already packaged as
EPUB 3 (302 of 365 books with an alignment map in this install are EPUB 2).

**This is a substantial port of Storyteller's own EPUB 2 upgrade**
(``gitlab.com/storyteller-platform/storyteller``, ``libraries/epub/upgrade.ts``
and the ``Epub.upgrade``/``getNcxTableOfContents``/``parseNavPoints`` orchestration
in ``libraries/epub/index.ts``), which is MIT licensed under the same licence as
BookBridge. The notice and copyright line travel with this port:

    MIT License

    Copyright (c) 2024 Shane Friedman

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

**What is ported vs. written fresh.** The per-field metadata upgrades
(``upgrade_identifiers``, ``upgrade_title``, ``upgrade_languages``,
``upgrade_authors``, ``upgrade_date``, ``upgrade_meta``, ``upgrade_cover``,
``remove_invalid_dc_attrs``, ``set_last_modified``), the guide/landmark
extraction (``extract_guide_landmarks``), the manifest-property detection
(``collect_manifest_properties``, including the svg/scripted/mathml/switch
set), the font MIME-type fix (``fix_font_mime_types``), the nav-document
shape (``build_toc_ol``/``build_nav_document``, including the "EPUB 3
requires a non-empty <ol>" fallback), ``choose_nav_href``'s collision
avoidance, and the overall orchestration order in :func:`upgrade_epub2_to_epub3`
are a direct algorithmic port of Storyteller's TypeScript, adapted from its
own AST-based XML abstraction (``fast-xml-parser``-shaped nodes) onto this
codebase's existing ``lxml``-based OPF handling
(``src/services/readalong_builder.py`` already parses/rewrites the OPF this
way). The NCX-to-navigation-tree walk (``extract_ncx_toc``/``_parse_nav_points``)
is likewise ported from ``Epub.getNcxTableOfContents``/``parseNavPoints``, with
``Epub.resolveHref``'s relative-path algorithm (:func:`_resolve_href_to_opf_dir`)
reproduced because a nav document placed beside the OPF (this module always
places it there, matching ``chooseNavHref``) needs hrefs expressed relative to
the OPF's own directory, not the NCX's.

Written fresh, with no TypeScript counterpart: the zip repackaging
(:func:`_repackage_with_additions`, which independently reproduces
``readalong_builder._package_epub``'s atomic-write and same-path-refusal
safety invariants rather than importing them, so this module carries no
import-time dependency in that direction -- ``readalong_builder`` imports
*this* module, not the reverse), the OPF-namespace-wildcard element lookups
(``{*}metadata`` etc., matching this codebase's own existing
``ebook_utils.py`` precedent for tolerating real-world EPUB2 files that omit
the ``opf:`` namespace prefix on ``opf:role``/``opf:file-as``/``opf:scheme``
attributes -- Storyteller's own parser is namespace-naive in a different way
and does not need this), and the final self-check gate
(:func:`_looks_like_valid_epub3`) that this module's own refuse-over-partial
policy runs before anything is written to disk.

**Deliberately not reused from Storyteller:** its XML AST/adapter
abstraction (``libraries/epub``'s ``Epub``/``ParsedXml`` machinery) and its
"rewrite every XHTML item through the parser to normalize the DOCTYPE" step.
The latter is skipped on purpose -- this module never touches spine XHTML
content at all, only the OPF and a newly added nav document, so a book's
alignment map (fitted against the *original* spine bytes) stays valid
against the upgraded copy without re-verification. This is the load-bearing
invariant :func:`upgrade_epub2_to_epub3` is built around; do not add spine
content rewriting to this module without re-deriving that guarantee.

**Never modifies its source file.** :func:`upgrade_epub2_to_epub3` writes a
brand new EPUB to ``output_path`` (atomically, via a temporary sibling +
``os.replace``, mirroring ``readalong_builder._package_epub``) and refuses
outright if ``output_path`` would alias ``source_path``. Callers (see
``readalong_builder._resolve_epub3_source``) are expected to point
``output_path`` at a private temporary file, never back at the library's own
copy -- this repository has previously shipped a defect where a build wrote
its output over its own source, which is exactly the failure mode this
mirrors the fix for.

**Refuse rather than ship a partial upgrade.** Every step in
:func:`upgrade_epub2_to_epub3` degrades gracefully for a single missing
input (no NCX -> empty TOC with a one-entry fallback nav; no guide -> no
landmarks; unmapped font extension -> media-type left alone), matching
Storyteller's own tolerance for the same real-world sloppiness. But the
function as a whole returns ``None`` -- logging why, never raising for an
expected condition -- if the source OPF cannot be located or parsed, if it
has no ``<metadata>``/``<manifest>`` to upgrade, or if the upgraded OPF fails
this module's own final validity self-check, so a caller can refuse the
whole read-along build rather than emit a package that merely claims to be
EPUB 3.
"""
import logging
import os
import posixpath
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import quote, unquote

from bs4 import BeautifulSoup
from lxml import etree

logger = logging.getLogger(__name__)

_OPF_NS = "http://www.idpf.org/2007/opf"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_XHTML_NS = "http://www.w3.org/1999/xhtml"
_OPS_NS = "http://www.idpf.org/2007/ops"

_NCX_MEDIA_TYPE = "application/x-dtbncx+xml"
_XHTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html"}

# EPUB 2 <guide type="..."> -> EPUB 3 nav "landmarks" epub:type, per the IDPF
# EPUB 3 Changes from EPUB 2.0.1 guidance. Ported verbatim from Storyteller's
# GUIDE_TO_EPUBTYPE (upgrade.ts) -- "notes" intentionally maps to "" (an empty,
# not missing, type): such a landmark is recognized but filtered out when the
# nav document is built (Storyteller has no EPUB 3 landmark type for it),
# exactly as Storyteller does; a guide type absent from this table entirely
# (key not found) is dropped instead of merely filtered.
_GUIDE_TO_EPUBTYPE: Dict[str, str] = {
    "acknowledgements": "acknowledgments",
    "other.afterword": "afterword",
    "other.appendix": "appendix",
    "other.backmatter": "backmatter",
    "bibliography": "bibliography",
    "text": "bodymatter",
    "other.chapter": "chapter",
    "colophon": "colophon",
    "other.conclusion": "conclusion",
    "other.contributors": "contributors",
    "copyright-page": "copyright-page",
    "cover": "cover",
    "dedication": "dedication",
    "other.division": "division",
    "epigraph": "epigraph",
    "other.epilogue": "epilogue",
    "other.errata": "errata",
    "other.footnotes": "footnotes",
    "foreword": "foreword",
    "other.frontmatter": "frontmatter",
    "glossary": "glossary",
    "other.halftitlepage": "halftitlepage",
    "other.imprint": "imprint",
    "other.imprimatur": "imprimatur",
    "index": "index",
    "other.introduction": "introduction",
    "other.landmarks": "landmarks",
    "other.loa": "loa",
    "loi": "loi",
    "lot": "lot",
    "other.lov": "lov",
    "notes": "",
    "other.notice": "notice",
    "other.other-credits": "other-credits",
    "other.part": "part",
    "other.preamble": "preamble",
    "preface": "preface",
    "other.prologue": "prologue",
    "other.rearnotes": "rearnotes",
    "other.subchapter": "subchapter",
    "title-page": "titlepage",
    "toc": "toc",
    "other.volume": "volume",
    "other.warning": "warning",
}

# Ported from Storyteller's upgrade.ts fixFontMimeTypes: correct EPUB 3 font
# media types (IANA-registered "font/*" types, the ones current epubcheck
# releases expect) keyed by file extension.
_FONT_MIME_BY_EXT: Dict[str, str] = {
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}

_RENDITION_META_NAMES = {"orientation", "layout", "spread"}


@dataclass(frozen=True)
class NavigationItem:
    """One entry of the parsed NCX table of contents (possibly nested)."""
    title: str
    href: Optional[str]
    children: List["NavigationItem"]


@dataclass(frozen=True)
class Landmark:
    """One EPUB 2 ``<guide>`` reference, already mapped to its EPUB 3
    ``epub:type`` (see :data:`_GUIDE_TO_EPUBTYPE`). ``href`` is verbatim from
    the guide entry -- both it and the generated nav document live beside the
    OPF, so no re-basing is needed (see this module's docstring)."""
    href: str
    title: str
    type: str


@dataclass(frozen=True)
class Epub3UpgradeResult:
    """The outcome of a successful :func:`upgrade_epub2_to_epub3` call."""
    output_path: str
    nav_href: str
    toc_entry_count: int
    landmark_count: int
    source_version: str


# ---------------------------------------------------------------------------
# Small XML helpers
# ---------------------------------------------------------------------------

def _local_name(tag: object) -> Optional[str]:
    """The local (namespace-stripped) name of an lxml tag, or ``None`` for a
    non-element node (comment/PI) whose ``tag`` is not a plain string."""
    if not isinstance(tag, str):
        return None
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag


def _get_metadata(pkg: etree._Element) -> Optional[etree._Element]:
    return pkg.find("{*}metadata")


def _get_manifest(pkg: etree._Element) -> Optional[etree._Element]:
    return pkg.find("{*}manifest")


def _get_spine(pkg: etree._Element) -> Optional[etree._Element]:
    return pkg.find("{*}spine")


def _get_guide(pkg: etree._Element) -> Optional[etree._Element]:
    return pkg.find("{*}guide")


def _get_attr_local(el: etree._Element, local_name: str) -> Optional[str]:
    """An attribute's value found by local name regardless of namespace
    prefix (or the lack of one) -- real-world EPUB 2 files (notably some
    Calibre output, per this codebase's own ``series_metadata.py`` precedent)
    frequently write ``role``/``file-as``/``scheme`` without the ``opf:``
    prefix the spec calls for."""
    for key, value in el.attrib.items():
        key_local = key.rsplit("}", 1)[-1] if key.startswith("{") else key
        if key_local == local_name:
            return value
    return None


def _text_content(el: etree._Element) -> str:
    """All text directly and indirectly inside ``el``, trimmed -- matches
    Storyteller's ``textContentOf`` (flattened child text, not just
    ``el.text``)."""
    return "".join(el.itertext()).strip()


def _strip_attrs_keep_id(el: etree._Element) -> None:
    """Remove every attribute except ``id`` -- the "clear all but id" step
    every metadata upgrade function in this module performs on the elements
    it has finished normalizing."""
    el_id = el.get("id")
    for key in list(el.attrib):
        del el.attrib[key]
    if el_id:
        el.set("id", el_id)


def _clear_children(el: etree._Element) -> None:
    for child in list(el):
        el.remove(child)


# ---------------------------------------------------------------------------
# Metadata upgrades (ported from upgrade.ts)
# ---------------------------------------------------------------------------

def upgrade_identifiers(pkg: etree._Element) -> None:
    """Normalize ``dc:identifier`` values and strip everything but ``id``.

    The identifier named by ``package@unique-identifier`` is preserved
    verbatim after text extraction.  EPUB font obfuscation derives its key
    from that publication identifier, and the archive's encrypted font bytes
    are copied unchanged during conversion.

    A ``urn:<scheme>:<value>`` identifier has its scheme extracted (unless
    the scheme itself starts with "uri", which already reads naturally); a
    non-``urn`` identifier carrying an ``opf:scheme`` (or bare ``scheme``,
    per :func:`_get_attr_local`) attribute gets that scheme prefixed onto its
    value, since EPUB 3 no longer has an ``opf:scheme`` attribute to carry it
    separately.
    """
    metadata = _get_metadata(pkg)
    if metadata is None:
        return
    unique_identifier_id = pkg.get("unique-identifier")
    for ident in metadata.findall("{*}identifier"):
        val = _text_content(ident)
        if unique_identifier_id and ident.get("id") == unique_identifier_id:
            _strip_attrs_keep_id(ident)
            _clear_children(ident)
            ident.text = val
            continue
        scheme = _get_attr_local(ident, "scheme")

        if val.lower().startswith("urn:"):
            rest = val[4:]
            colon_idx = rest.find(":")
            if colon_idx > 0:
                scheme = rest[:colon_idx]
                val = rest[colon_idx + 1:]

        if scheme and val and not scheme.lower().startswith("uri"):
            val = f"{scheme}:{val}"

        _strip_attrs_keep_id(ident)
        _clear_children(ident)
        ident.text = val


def upgrade_title(pkg: etree._Element) -> None:
    """Drop empty ``dc:title`` elements and mark the first surviving one as
    the EPUB 3 ``title-type: main`` via a ``refines`` meta."""
    metadata = _get_metadata(pkg)
    if metadata is None:
        return
    titles = metadata.findall("{*}title")
    first_title = None
    for title in titles:
        if not _text_content(title):
            metadata.remove(title)
            continue
        if first_title is None:
            first_title = title

    if first_title is None:
        return

    title_id = first_title.get("id")
    if not title_id:
        title_id = f"id-{os.urandom(4).hex()}"
        first_title.set("id", title_id)

    meta = etree.SubElement(metadata, f"{{{_OPF_NS}}}meta")
    meta.set("refines", f"#{title_id}")
    meta.set("property", "title-type")
    meta.text = "main"


def upgrade_languages(pkg: etree._Element) -> None:
    """Strip stray attributes from existing ``dc:language`` elements, or add
    the EPUB 3 "unknown language" code (``und``, per BCP 47/RFC 5646) if none
    is present -- EPUB 3 requires at least one."""
    metadata = _get_metadata(pkg)
    if metadata is None:
        return
    langs = metadata.findall("{*}language")
    if langs:
        for lang in langs:
            _strip_attrs_keep_id(lang)
        return
    lang_el = etree.SubElement(metadata, f"{{{_DC_NS}}}language")
    lang_el.text = "und"


def upgrade_authors(pkg: etree._Element) -> None:
    """Migrate ``opf:role``/``opf:file-as`` (or bare ``role``/``file-as``)
    attributes off ``dc:creator``/``dc:contributor`` elements onto EPUB 3
    ``refines`` metas, since EPUB 3 does not define those attributes."""
    metadata = _get_metadata(pkg)
    if metadata is None:
        return
    for local_name in ("creator", "contributor"):
        for elem in metadata.findall(f"{{*}}{local_name}"):
            role = _get_attr_local(elem, "role")
            file_as = _get_attr_local(elem, "file-as")

            elem_id = elem.get("id")
            if (role or file_as) and not elem_id:
                elem_id = f"id-{os.urandom(4).hex()}"
                elem.set("id", elem_id)

            _strip_attrs_keep_id(elem)

            if role:
                meta = etree.SubElement(metadata, f"{{{_OPF_NS}}}meta")
                meta.set("refines", f"#{elem_id}")
                meta.set("property", "role")
                meta.set("scheme", "marc:relators")
                meta.text = role

            if file_as:
                meta2 = etree.SubElement(metadata, f"{{{_OPF_NS}}}meta")
                meta2.set("refines", f"#{elem_id}")
                meta2.set("property", "file-as")
                meta2.text = file_as


def upgrade_date(pkg: etree._Element) -> None:
    """Keep only the first non-empty ``dc:date``, stripped of attributes --
    EPUB 3 metadata allows at most one."""
    metadata = _get_metadata(pkg)
    if metadata is None:
        return
    kept = False
    for date in metadata.findall("{*}date"):
        if not _text_content(date) or kept:
            metadata.remove(date)
            continue
        kept = True
        _strip_attrs_keep_id(date)


def upgrade_meta(pkg: etree._Element) -> None:
    """Rewrite EPUB 2's ``<meta name="..." content="...">`` rendition hints
    (``orientation``/``layout``/``spread``/``fixed-layout``/``orientation-lock``,
    including the legacy ``rendition:`` prefix form) into EPUB 3's
    ``<meta property="rendition:...">value</meta>`` shape. A ``<meta>`` that
    matches none of these hints is left completely untouched (e.g. Calibre's
    own ``calibre:series`` meta, which ``series_metadata.py`` still reads
    from the *generated* read-along file the same way)."""
    metadata = _get_metadata(pkg)
    if metadata is None:
        return
    for meta in metadata.findall("{*}meta"):
        name = meta.get("name") or ""
        content = meta.get("content") or ""
        prop: Optional[str] = None
        value = content

        clean_name = name[len("rendition:"):] if name.startswith("rendition:") else name

        if clean_name in _RENDITION_META_NAMES:
            prop = f"rendition:{clean_name}"
        elif name == "fixed-layout":
            prop = "rendition:layout"
            value = "pre-paginated" if content.lower() == "true" else "reflowable"
        elif name == "orientation-lock":
            prop = "rendition:orientation"
            value = {"portrait": "portrait", "landscape": "landscape"}.get(content.lower(), "auto")

        if not prop:
            continue

        if "name" in meta.attrib:
            del meta.attrib["name"]
        if "content" in meta.attrib:
            del meta.attrib["content"]
        meta.set("property", prop)
        _clear_children(meta)
        meta.text = value


def upgrade_cover(pkg: etree._Element) -> None:
    """Migrate EPUB 2's ``<meta name="cover" content="<manifest-id>">`` to
    the EPUB 3 ``properties="cover-image"`` on that manifest item -- only
    when the referenced item is actually an image (an EPUB 2 cover
    referencing an XHTML wrapper page is left alone, matching Storyteller's
    own guard)."""
    metadata = _get_metadata(pkg)
    manifest = _get_manifest(pkg)
    if metadata is None or manifest is None:
        return
    for meta in metadata.findall("{*}meta"):
        if meta.get("name") != "cover" or not meta.get("content"):
            continue
        item_id = meta.get("content")
        for item in manifest.findall("{*}item"):
            if item.get("id") != item_id:
                continue
            media_type = (item.get("media-type") or "").lower()
            is_image = bool(media_type) and "xml" not in media_type and "html" not in media_type
            if not is_image:
                continue
            existing = {p for p in (item.get("properties") or "").split() if p}
            existing.add("cover-image")
            item.set("properties", " ".join(sorted(existing)))


def remove_invalid_dc_attrs(pkg: etree._Element) -> None:
    """Final catch-all: strip every attribute but ``id`` from any ``dc:*``
    metadata element the more specific upgrades above did not already
    handle (``dc:publisher``, ``dc:subject``, ``dc:rights``, etc.) -- EPUB 3
    does not define ``opf:*`` attributes on any of them."""
    metadata = _get_metadata(pkg)
    if metadata is None:
        return
    for node in metadata:
        tag = node.tag
        if not isinstance(tag, str) or etree.QName(tag).namespace != _DC_NS:
            continue
        _strip_attrs_keep_id(node)


def set_last_modified(pkg: etree._Element) -> None:
    """Replace any existing ``dcterms:modified`` meta with one stamped at
    conversion time -- EPUB 3 requires this property."""
    metadata = _get_metadata(pkg)
    if metadata is None:
        return
    for meta in metadata.findall("{*}meta"):
        if meta.get("property") == "dcterms:modified":
            metadata.remove(meta)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = etree.SubElement(metadata, f"{{{_OPF_NS}}}meta")
    meta.set("property", "dcterms:modified")
    meta.text = now


def upgrade_package_metadata(pkg: etree._Element) -> None:
    """Run every metadata upgrade, in the same order Storyteller's own
    ``upgradePackageMetadata`` does (later steps assume earlier ones already
    ran -- e.g. ``remove_invalid_dc_attrs`` is a catch-all and must run after
    the field-specific upgrades, and ``set_last_modified`` must run last so
    nothing after it could remove the stamp it adds)."""
    upgrade_identifiers(pkg)
    upgrade_title(pkg)
    upgrade_languages(pkg)
    upgrade_authors(pkg)
    upgrade_date(pkg)
    upgrade_meta(pkg)
    upgrade_cover(pkg)
    remove_invalid_dc_attrs(pkg)
    set_last_modified(pkg)


# ---------------------------------------------------------------------------
# Guide -> landmarks, spine/guide cleanup, manifest properties
# ---------------------------------------------------------------------------

def extract_guide_landmarks(pkg: etree._Element) -> List[Landmark]:
    """Every ``<guide><reference>`` entry whose ``type`` maps to a known EPUB
    3 landmark type (:data:`_GUIDE_TO_EPUBTYPE`) and carries an ``href``.
    A recognized-but-typeless mapping (``notes`` -> ``""``) is still
    returned here -- it is filtered out later, in :func:`build_nav_document`,
    exactly where Storyteller filters it."""
    guide = _get_guide(pkg)
    if guide is None:
        return []
    landmarks: List[Landmark] = []
    for ref in guide.findall("{*}reference"):
        href = ref.get("href") or ""
        title = ref.get("title") or ""
        guide_type = (ref.get("type") or "").lower()
        epub_type = _GUIDE_TO_EPUBTYPE.get(guide_type)
        if epub_type is None or not href:
            continue
        landmarks.append(Landmark(href=href, title=title, type=epub_type))
    return landmarks


def remove_guide(pkg: etree._Element) -> None:
    """Remove the deprecated ``<guide>`` element entirely -- its content is
    carried forward into the nav document's landmarks nav."""
    guide = _get_guide(pkg)
    if guide is not None:
        pkg.remove(guide)


def remove_spine_toc_ref(pkg: etree._Element) -> None:
    """Remove ``<spine toc="...">`` -- EPUB 3 navigation is the nav document
    registered in the manifest, not the spine's ``toc`` attribute."""
    spine = _get_spine(pkg)
    if spine is None:
        return
    if "toc" in spine.attrib:
        del spine.attrib["toc"]


def fix_font_mime_types(pkg: etree._Element) -> None:
    """Correct a manifest font item's declared ``media-type`` when it
    disagrees with its file extension (e.g. a ``.ttf`` file still declared
    as the legacy ``application/x-font-ttf``) -- only touches items whose
    *declared* type already looks like a font, so an item legitimately using
    some other media type is never reclassified."""
    manifest = _get_manifest(pkg)
    if manifest is None:
        return
    for item in manifest.findall("{*}item"):
        declared = (item.get("media-type") or "").lower()
        if "font" not in declared:
            continue
        href = (item.get("href") or "").split("#", 1)[0]
        ext = posixpath.splitext(href)[1].lower()
        corrected = _FONT_MIME_BY_EXT.get(ext)
        if not corrected or corrected == declared:
            continue
        item.set("media-type", corrected)


def _detect_xhtml_properties(content: bytes) -> Set[str]:
    """Manifest ``properties`` an XHTML spine document's actual content
    requires under EPUB 3 (``svg``/``scripted``/``mathml``/``switch``),
    ported from Storyteller's ``collectManifestProperties``. Read-only:
    never mutates or re-serializes ``content``, so spine bytes stay
    untouched (see this module's docstring)."""
    soup = BeautifulSoup(content, "html.parser")
    found: Set[str] = set()
    if soup.find("svg"):
        found.add("svg")
    if soup.find("script"):
        found.add("scripted")
    if soup.find("math"):
        found.add("mathml")
    if soup.find(lambda tag: (tag.name or "").lower() == "epub:switch"):
        found.add("switch")
    return found


def collect_manifest_properties(pkg: etree._Element, opf_dir: str, zf: zipfile.ZipFile) -> None:
    """Add ``svg``/``scripted``/``mathml``/``switch`` to each XHTML manifest
    item's ``properties`` when its content actually uses that feature --
    EPUB 3 requires these to be declared. Reads each item's *original*
    archive bytes (read-only) to detect them."""
    manifest = _get_manifest(pkg)
    if manifest is None:
        return
    zip_names = set(zf.namelist())
    for item in manifest.findall("{*}item"):
        media_type = (item.get("media-type") or "").lower()
        if media_type not in _XHTML_MEDIA_TYPES:
            continue
        href = item.get("href")
        if not href:
            continue
        archive_path = _resolve_archive_path(opf_dir, unquote(href))
        if archive_path not in zip_names:
            continue
        try:
            content = zf.read(archive_path)
        except (KeyError, zipfile.BadZipFile) as e:
            logger.warning(
                "EPUB3 upgrade: could not read manifest item '%s' to detect "
                "properties: %s", archive_path, e, exc_info=True,
            )
            continue
        detected = _detect_xhtml_properties(content)
        if not detected:
            continue
        existing = {p for p in (item.get("properties") or "").split() if p}
        merged = existing | detected
        if merged != existing:
            item.set("properties", " ".join(sorted(merged)))


def set_package_version(pkg: etree._Element, version: str) -> None:
    pkg.set("version", version)


def choose_nav_href(pkg: etree._Element) -> str:
    """A manifest-``href``-collision-free filename for the new nav document
    (``nav.xhtml``, ``nav1.xhtml``, ... ), placed beside the OPF."""
    manifest = _get_manifest(pkg)
    existing_hrefs: Set[str] = set()
    if manifest is not None:
        existing_hrefs = {item.get("href") for item in manifest.findall("{*}item") if item.get("href")}
    candidate = "nav.xhtml"
    i = 1
    while candidate in existing_hrefs:
        candidate = f"nav{i}.xhtml"
        i += 1
    return candidate


# ---------------------------------------------------------------------------
# NCX parsing
# ---------------------------------------------------------------------------

def _resolve_archive_path(base_dir: str, ref: str) -> str:
    """Join a (decoded) relative reference onto ``base_dir`` and normalize
    it into a full archive path -- a fragment, if any, is dropped (this is
    used only for locating a manifest entry's actual archive bytes)."""
    path_part = ref.split("#", 1)[0]
    if base_dir:
        return posixpath.normpath(posixpath.join(base_dir, path_part))
    return posixpath.normpath(path_part)


def _resolve_href_to_opf_dir(ncx_href: str, src: str, opf_dir: str) -> str:
    """Resolve ``src`` (found inside the NCX at archive path ``ncx_href``) to
    a path relative to the OPF's own directory (``opf_dir``).

    Ported from Storyteller's ``Epub.resolveHref(src, ncxHref)`` (called
    with no ``toRoot`` option, i.e. relative to the *package document's*
    directory) -- the new nav document is always written beside the OPF
    (:func:`choose_nav_href`), so an href expressed relative to ``opf_dir``
    is exactly what it needs for its own ``<a href="...">``. A bare fragment
    (``"#..."``) is returned unchanged, matching ``resolveHref``'s own
    same-document shortcut.
    """
    if src.startswith("#"):
        return src
    path_part, _, fragment = src.partition("#")
    ncx_dir = posixpath.dirname(ncx_href)
    target_abs = _resolve_archive_path(ncx_dir, unquote(path_part))

    target_segments = target_abs.split("/") if target_abs else []
    base_segments = opf_dir.split("/") if opf_dir else []
    shared = 0
    while (
        shared < len(base_segments)
        and shared < len(target_segments)
        and base_segments[shared] == target_segments[shared]
    ):
        shared += 1

    climbs = [".."] * (len(base_segments) - shared)
    remainder = [quote(seg) for seg in target_segments[shared:]]
    relative = "/".join(climbs + remainder)
    return f"{relative}#{fragment}" if fragment else relative


def _find_ncx_item(pkg: etree._Element) -> Optional[etree._Element]:
    """The manifest ``<item>`` for the book's NCX: the spine's ``toc``
    attribute if it resolves to a real item, else the first item whose
    media-type is the NCX type -- same fallback order as Storyteller's
    ``getNcxTableOfContents``."""
    manifest = _get_manifest(pkg)
    if manifest is None:
        return None
    items = list(manifest.findall("{*}item"))

    spine = _get_spine(pkg)
    toc_id = spine.get("toc") if spine is not None else None
    if toc_id:
        for item in items:
            if item.get("id") == toc_id:
                return item

    for item in items:
        if (item.get("media-type") or "").lower() == _NCX_MEDIA_TYPE:
            return item
    return None


def _parse_nav_points(container: etree._Element, ncx_href: str, opf_dir: str) -> List[NavigationItem]:
    entries: List[NavigationItem] = []
    for node in container:
        if _local_name(node.tag) not in ("navPoint", "navpoint"):
            continue

        nav_label = node.find("{*}navLabel")
        if nav_label is None:
            nav_label = node.find("{*}navlabel")
        title = None
        if nav_label is not None:
            text_el = nav_label.find("{*}text")
            if text_el is not None:
                title = _text_content(text_el) or None

        content_el = node.find("{*}content")
        src = content_el.get("src") if content_el is not None else None
        href = _resolve_href_to_opf_dir(ncx_href, src, opf_dir) if src else None

        children = _parse_nav_points(node, ncx_href, opf_dir)
        entries.append(NavigationItem(
            title=title if title else str(len(entries)),
            href=href,
            children=children,
        ))
    return entries


def extract_ncx_toc(pkg: etree._Element, opf_dir: str, zf: zipfile.ZipFile) -> List[NavigationItem]:
    """The book's table of contents, walked from its NCX ``navMap``.

    Returns ``[]`` -- never raises -- when there is no NCX manifest item, it
    is missing from the archive, it fails to parse, or it has no
    ``navMap``: :func:`build_nav_document` falls back sanely (a single
    "Start" entry pointing at the first spine item) exactly as Storyteller's
    own ``buildNavDocument`` does for an empty TOC.
    """
    ncx_item = _find_ncx_item(pkg)
    if ncx_item is None:
        return []
    ncx_href = ncx_item.get("href")
    if not ncx_href:
        return []
    ncx_archive_path = _resolve_archive_path(opf_dir, unquote(ncx_href))
    try:
        ncx_bytes = zf.read(ncx_archive_path)
    except KeyError:
        logger.warning(
            "EPUB3 upgrade: NCX manifest item '%s' is not present in the archive",
            ncx_archive_path, exc_info=True,
        )
        return []
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        ncx_root = etree.fromstring(ncx_bytes, parser=parser)
    except etree.XMLSyntaxError as e:
        logger.warning(
            "EPUB3 upgrade: could not parse NCX '%s': %s", ncx_archive_path, e, exc_info=True,
        )
        return []

    nav_map = ncx_root.find("{*}navMap")
    if nav_map is None:
        nav_map = ncx_root.find("{*}navmap")
    if nav_map is None:
        return []
    return _parse_nav_points(nav_map, ncx_archive_path, opf_dir)


# ---------------------------------------------------------------------------
# Nav document assembly
# ---------------------------------------------------------------------------

def build_toc_ol(entries: List[NavigationItem]) -> etree._Element:
    """A nav ``<ol>`` for ``entries``, recursing into nested children --
    ported from Storyteller's ``buildTocOl``."""
    ol = etree.Element(f"{{{_XHTML_NS}}}ol")
    for entry in entries:
        li = etree.SubElement(ol, f"{{{_XHTML_NS}}}li")
        label = re.sub(r"\s+", " ", entry.title).strip()
        if entry.href:
            a = etree.SubElement(li, f"{{{_XHTML_NS}}}a")
            a.set("href", entry.href)
            a.text = label
        else:
            span = etree.SubElement(li, f"{{{_XHTML_NS}}}span")
            span.text = label
        if entry.children:
            li.append(build_toc_ol(entry.children))
    return ol


def build_nav_document(
    toc_entries: List[NavigationItem], landmarks: List[Landmark], fallback_href: str,
) -> bytes:
    """Build a complete EPUB 3 navigation document.

    ``toc_entries`` empty falls back to a single "Start" entry pointing at
    ``fallback_href`` (the first spine item's own manifest href, already
    relative to the OPF's directory) -- EPUB 3 requires the toc nav's
    ``<ol>`` to be non-empty. ``landmarks`` whose ``type`` is falsy (the
    ``notes`` guide type, see :data:`_GUIDE_TO_EPUBTYPE`) are dropped, and
    the landmarks nav is omitted entirely if none remain -- both ported from
    Storyteller's ``buildNavDocument``.
    """
    nsmap = {None: _XHTML_NS, "epub": _OPS_NS}
    html = etree.Element(f"{{{_XHTML_NS}}}html", nsmap=nsmap)
    head = etree.SubElement(html, f"{{{_XHTML_NS}}}head")
    title_el = etree.SubElement(head, f"{{{_XHTML_NS}}}title")
    title_el.text = "Navigation"
    body = etree.SubElement(html, f"{{{_XHTML_NS}}}body")

    toc_nav = etree.SubElement(body, f"{{{_XHTML_NS}}}nav")
    toc_nav.set(f"{{{_OPS_NS}}}type", "toc")
    h1 = etree.SubElement(toc_nav, f"{{{_XHTML_NS}}}h1")
    h1.text = "Table of Contents"

    if toc_entries:
        toc_ol = build_toc_ol(toc_entries)
    else:
        toc_ol = etree.Element(f"{{{_XHTML_NS}}}ol")
        li = etree.SubElement(toc_ol, f"{{{_XHTML_NS}}}li")
        a = etree.SubElement(li, f"{{{_XHTML_NS}}}a")
        a.set("href", fallback_href or "#")
        a.text = "Start"
    toc_nav.append(toc_ol)

    valid_landmarks = [lm for lm in landmarks if lm.type]
    if valid_landmarks:
        landmarks_nav = etree.SubElement(body, f"{{{_XHTML_NS}}}nav")
        landmarks_nav.set(f"{{{_OPS_NS}}}type", "landmarks")
        landmarks_nav.set("hidden", "")
        lm_ol = etree.SubElement(landmarks_nav, f"{{{_XHTML_NS}}}ol")
        for lm in valid_landmarks:
            li = etree.SubElement(lm_ol, f"{{{_XHTML_NS}}}li")
            a = etree.SubElement(li, f"{{{_XHTML_NS}}}a")
            a.set(f"{{{_OPS_NS}}}type", lm.type)
            a.set("href", lm.href)
            a.text = lm.title or lm.type

    return etree.tostring(
        html, xml_declaration=True, encoding="utf-8", standalone=False,
        doctype="<!DOCTYPE html>",
    )


# ---------------------------------------------------------------------------
# Packaging (self-contained -- see module docstring on why this does not
# import readalong_builder._package_epub)
# ---------------------------------------------------------------------------

def _find_opf_path(zf: zipfile.ZipFile) -> Optional[str]:
    """The OPF's full archive path, read from ``META-INF/container.xml``.

    Parses container.xml as XML (external entity resolution and network
    access disabled, matching this module's own OPF-parsing precedent above)
    rather than regex-matching ``full-path="..."`` directly. The regex
    required a double-quoted attribute and left XML-escaped characters
    (``&amp;``, ``&apos;``, ...) undecoded, so a perfectly valid
    single-quoted or escaped container.xml was rejected as having no OPF at
    all -- Defect 3, independent review. The ``rootfile`` element is matched
    by local name regardless of namespace prefix/declaration (mirrors
    ``smil_extractor.py``'s own already-existing, more tolerant lookup),
    since real-world container.xml files are not always as strictly
    namespaced as the OCF spec's own worked examples.

    Shared with ``readalong_builder.py``, which imports this rather than
    keeping its own copy -- both previously carried the identical regex bug.
    """
    try:
        container_bytes = zf.read("META-INF/container.xml")
    except KeyError:
        return None
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        tree = etree.fromstring(container_bytes, parser=parser)
    except etree.XMLSyntaxError as e:
        logger.warning(
            "Could not parse container.xml to find the OPF path: %s", e, exc_info=True,
        )
        return None
    for element in tree.iter():
        tag = element.tag
        local_name = tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""
        if local_name == "rootfile":
            full_path = element.get("full-path")
            if full_path:
                return full_path
    return None


def _repackage_with_additions(
    source_path: Path,
    output_path: Path,
    modified_files: Dict[str, bytes],
    new_bytes_files: Dict[str, bytes],
) -> None:
    """Rewrite ``source_path`` into ``output_path``, overriding
    ``modified_files`` (existing archive entries) and appending
    ``new_bytes_files`` (new ones); every other entry is copied
    byte-for-byte with its original compression. ``mimetype`` is written
    first and stored uncompressed per the EPUB OCF spec.

    Refuses (``ValueError``, without touching either file) when
    ``output_path`` would alias ``source_path`` -- opening the destination
    for writing would truncate the file this function is still reading from.
    The archive is otherwise built into a temporary sibling and only
    ``os.replace()``d onto ``output_path`` once fully written, so a failure
    partway through never leaves a truncated file at the real destination.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == output_path.resolve():
        raise ValueError(
            f"EPUB3 upgrade refuses to write '{output_path}' over its own "
            "source -- opening the destination for writing would truncate "
            "the file still being read from"
        )

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    os.chmod(tmp_path, 0o644)
    try:
        with zipfile.ZipFile(source_path) as src, zipfile.ZipFile(tmp_path, "w") as dst:
            dst.writestr(
                zipfile.ZipInfo("mimetype", date_time=(1980, 1, 1, 0, 0, 0)),
                b"application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            for name in src.namelist():
                if name == "mimetype":
                    continue
                info = src.getinfo(name)
                data = modified_files.get(name, src.read(name))
                new_info = zipfile.ZipInfo(name, date_time=info.date_time)
                new_info.compress_type = info.compress_type
                new_info.external_attr = info.external_attr
                dst.writestr(new_info, data)
            for name, data in new_bytes_files.items():
                dst.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.warning(
                "EPUB3 upgrade: could not remove temporary file '%s' after a "
                "failed conversion: %s", tmp_path, cleanup_error, exc_info=True,
            )
        raise


def _looks_like_valid_epub3(opf_bytes: bytes) -> bool:
    """A minimal, self-contained gate before this module writes anything:
    the produced OPF must parse, declare an EPUB 3 ``version``, and have a
    manifest item with ``properties="nav"``."""
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        pkg = etree.fromstring(opf_bytes, parser=parser)
    except etree.XMLSyntaxError:
        return False
    version = pkg.get("version") or ""
    if not version.startswith("3"):
        return False
    manifest = _get_manifest(pkg)
    if manifest is None:
        return False
    return any(
        "nav" in (item.get("properties") or "").split()
        for item in manifest.findall("{*}item")
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def upgrade_epub2_to_epub3(
    source_path: Union[str, Path], output_path: Union[str, Path],
) -> Optional[Epub3UpgradeResult]:
    """Convert an EPUB 2 file at ``source_path`` into a valid EPUB 3 at
    ``output_path``, never touching ``source_path`` itself.

    Performs, in order (matching Storyteller's own ``Epub.upgrade``):
    parses the NCX into a TOC tree and the guide into landmarks (both read
    before anything is mutated); upgrades OPF metadata to EPUB 3 conventions;
    fixes font MIME types; removes the ``<guide>`` element and the spine's
    ``toc`` attribute; bumps the package version to ``3.0``; scans XHTML
    manifest items for svg/scripted/mathml/switch properties; and adds a new
    navigation document, registered in the manifest with
    ``properties="nav"``.

    Spine XHTML content is never read for modification (only for the
    read-only property scan above) or rewritten -- see this module's
    docstring on why that is load-bearing for the alignment map that will be
    applied against the result.

    Returns ``None`` -- logging why, never raising for an expected failure
    -- if ``source_path``'s OPF cannot be located or parsed, is missing
    ``<metadata>``/``<manifest>``, or if the result fails this module's own
    final validity self-check. Raises ``ValueError`` only for the
    programmer error of aliasing ``output_path`` to ``source_path`` (see
    :func:`_repackage_with_additions`) or a genuine filesystem failure
    (``OSError``/``zipfile.BadZipFile``) reading or writing either file.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)

    try:
        with zipfile.ZipFile(source_path) as zf:
            zip_names = set(zf.namelist())
            opf_path = _find_opf_path(zf)
            if not opf_path or opf_path not in zip_names:
                logger.warning(
                    "EPUB3 upgrade: could not locate the OPF in '%s'", source_path,
                )
                return None
            opf_bytes = zf.read(opf_path)

            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            try:
                pkg = etree.fromstring(opf_bytes, parser=parser)
            except etree.XMLSyntaxError as e:
                logger.warning(
                    "EPUB3 upgrade: could not parse OPF in '%s': %s",
                    source_path, e, exc_info=True,
                )
                return None

            source_version = pkg.get("version") or "<missing>"
            opf_dir = posixpath.dirname(opf_path)

            manifest = _get_manifest(pkg)
            if _get_metadata(pkg) is None or manifest is None:
                logger.warning(
                    "EPUB3 upgrade: OPF in '%s' is missing <metadata> or "
                    "<manifest>; refusing", source_path,
                )
                return None

            toc_entries = extract_ncx_toc(pkg, opf_dir, zf)
            landmarks = extract_guide_landmarks(pkg)

            upgrade_package_metadata(pkg)
            fix_font_mime_types(pkg)
            remove_guide(pkg)
            remove_spine_toc_ref(pkg)
            set_package_version(pkg, "3.0")
            collect_manifest_properties(pkg, opf_dir, zf)

            first_spine_href = _first_spine_item_href(pkg)
            nav_href = choose_nav_href(pkg)
            nav_content = build_nav_document(toc_entries, landmarks, first_spine_href or "#")

            existing_ids = {item.get("id") for item in manifest.findall("{*}item") if item.get("id")}
            nav_id = "nav"
            if nav_id in existing_ids:
                n = 2
                while f"nav-{n}" in existing_ids:
                    n += 1
                nav_id = f"nav-{n}"
            nav_item = etree.SubElement(manifest, f"{{{_OPF_NS}}}item")
            nav_item.set("id", nav_id)
            nav_item.set("href", nav_href)
            nav_item.set("media-type", "application/xhtml+xml")
            nav_item.set("properties", "nav")

            new_opf_bytes = etree.tostring(
                pkg, xml_declaration=True, encoding="utf-8", standalone=False,
            )

            if not _looks_like_valid_epub3(new_opf_bytes):
                logger.error(
                    "EPUB3 upgrade: converted OPF for '%s' failed this "
                    "module's own validity self-check; refusing", source_path,
                )
                return None

            nav_archive_path = _resolve_archive_path(opf_dir, nav_href)
            _repackage_with_additions(
                source_path, output_path,
                modified_files={opf_path: new_opf_bytes},
                new_bytes_files={nav_archive_path: nav_content},
            )
    except (OSError, zipfile.BadZipFile) as e:
        logger.error(
            "EPUB3 upgrade: could not read/write EPUB for '%s': %s",
            source_path, e, exc_info=True,
        )
        return None

    logger.info(
        "📖 Upgraded EPUB 2 (version=%s) to EPUB 3 for '%s': %d TOC entries, "
        "%d landmarks, nav='%s' -> '%s'",
        source_version, source_path, len(toc_entries),
        len([lm for lm in landmarks if lm.type]), nav_href, output_path,
    )
    return Epub3UpgradeResult(
        output_path=str(output_path),
        nav_href=nav_href,
        toc_entry_count=len(toc_entries),
        landmark_count=len([lm for lm in landmarks if lm.type]),
        source_version=source_version,
    )


def _first_spine_item_href(pkg: etree._Element) -> Optional[str]:
    """The first spine itemref's manifest href (relative to the OPF's own
    directory) -- the nav document's fallback "Start" target when the book
    has no usable NCX."""
    manifest = _get_manifest(pkg)
    spine = _get_spine(pkg)
    if manifest is None or spine is None:
        return None
    items_by_id = {item.get("id"): item for item in manifest.findall("{*}item") if item.get("id")}
    for itemref in spine.findall("{*}itemref"):
        item = items_by_id.get(itemref.get("idref"))
        if item is not None and item.get("href"):
            return item.get("href")
    return None
