# Release Notes - 7.8.0

**BookBridge now builds read-along EPUBs for BookOrbit itself, and forced alignment
is a regular, recommended option that runs on any CPU.** Your KOReader devices also
share more than progress: reading status, a part-read book's "reading" state,
finished books and, optionally, your recently-read history now travel between them.

This release ships BridgeSync **0.9.6** for KOReader and fixes positions landing a
paragraph early, Storyteller snapping back to the audiobook position, and a CWA
download that could fetch the wrong book.

## Action Required

- Re-download BridgeSync **0.9.6** on every KOReader device that uses the plugin,
  then restart KOReader.

## What's New

- **See what's new after an upgrade.** The first time you open the Library after an
  update, a banner shows the new version, links to these notes and lists anything you
  need to do. It appears once per update for every user.

- **Read-along EPUBs for BookOrbit, built by BookBridge.** A matched audiobook and
  ebook can become a read-along EPUB: the book with its narration built in,
  highlighting each sentence as it is read, in both the BookOrbit app and the web
  reader (including Safari on iPhone and iPad). It is built from the book's own
  alignment, so no Storyteller server is involved, and delivered straight into the
  book's BookOrbit audiobook folder. Tick **Also generate a read-along EPUB for
  BookOrbit** when matching, or use **Create read-along EPUB** from a book's menu on
  the dashboard. EPUB 2 books are converted automatically, and a failed rebuild keeps
  the previous read-along. A headphones pill marks books that have one.
  **Read-Along Audio Bitrate** (default 32k) sets the size of the embedded audio.

- **Forced alignment is recommended, and runs on any CPU.** Settings now offers a
  plain choice: **Use forced alignment (recommended)**, which aligns each audiobook
  word by word against its ebook's text, or transcription. It no longer needs a
  special image or a GPU: the QuartzNet model runs on the CPU in every image and finds
  each chapter in the audio itself, so a 22-hour audiobook aligns in about 7 minutes.
  Across 15 existing books it matched the previous alignment on 11, fixed one that was
  up to 2 hours off, and sent the two it could not follow (one narrated out of order,
  one that retells rather than reads the text) to transcription on its own. It
  understands English only; other languages use transcription automatically. The
  model downloads once on first use (77 MB); if that fails, point **QuartzNet model
  file** (under **Advanced — forced alignment model**) at a copy you already have.

- **Reading status is shared between your KOReader devices.** A book you finish on
  the Kobo now shows as finished on the Kindle, and one you start on the Kindle shows
  as in progress everywhere, including books delivered to a device but never opened
  there. When devices disagree, the most recent change wins and "finished" settles a
  same-day tie. Only the status is shared, never your position. On by default:
  **Sync reading status between devices** under Settings → KOReader / KoSync.

- **Your readers learn what the bridge already knows.** A part-read book now shows
  as in progress on your devices instead of untouched, unless you marked it finished
  or abandoned. Finishing a book anywhere (BookOrbit, ABS, Grimmory, CWA or
  Storyteller) marks it finished on your KOReader devices. Clearing a book's progress
  also clears its status there, so it goes back to unread before a re-read.

- **Share recently-read books between devices.** KOReader's History only lists
  books opened on that device; this files your reading from other devices into it
  too, so a "Recent" shelf means recently read *by you*. Off by default: **Share
  recently-read books between devices** under Settings → KOReader / KoSync. It only
  adds books already on the device, read in the last 30 days, at most 25 per sync.

- **The device you are reading on can win over the furthest one (#215).** A device
  you have not opened in weeks no longer has to keep pulling you forward. The device
  you are actually reading on can win once it has proved itself with several forward
  page turns in a row; opening a stale reader is never enough. New setting **When two
  KOReader devices disagree** under Settings → KOReader / KoSync → Advanced
  cross-device progress. It ships on **Watch and log only**, which records the
  decision without changing anything your readers receive.

## Fixed

- **Synced ebook positions land in the right place.** Progress sent to an ebook
  reader could land in the previous paragraph, a position on a "* * *" scene break
  jumped back to the start of the chapter, and a repeated line could resolve to its
  first copy. Positions read from Storyteller and the Audiobookshelf ebook reader no
  longer land about a sentence early.

- **Reading on in Storyteller after listening no longer snaps you back (#447).**
  BookBridge mistook your first pages of reading for its own update and put
  Storyteller back where the audiobook stopped, until you read about 1% in one go.
  It now recognises its own updates by the exact timestamp it sent.

- **A missing CWA ebook is no longer re-downloaded as the wrong book (#448).** A
  re-download searched CWA for the book's ID as text and could accept a different
  book whose title contained those digits. Only the exact ID is accepted now.

- **Registering from KOReader or Readest no longer pretends to succeed (#446).**
  "Register" only succeeds for an account set up in BookBridge, and otherwise points
  you to **My Account → My Integrations**.

- **EPUBs that bold part of each word extract correctly.** "Bionic reading" styling
  no longer splits words ("Th e"), which could break alignment and position sync.
  Contributed by [@mehalter](https://github.com/mehalter) in #445.

- **BookBridge and BookOrbit 3.0 no longer both drive a read-along book.** When a
  book's audio and text are the same BookOrbit entry, BookBridge now writes only the
  ebook side and lets BookOrbit keep the audio in step. **BookOrbit → Read-Along
  Sync** can instead switch BookOrbit's own sync off for such books.

- **BookOrbit reads and writes the same ebook when a book has several formats
  (#443).** A secondary KEPUB no longer receives updates while the EPUB stays behind.

- **Going back on a second KOReader device sticks once you carry on reading (#215).**
  The reading that proves a rewind deliberate is now remembered, and one device can
  no longer vouch for another device's rewind.

- **An unstarted BookOrbit or Grimmory audiobook no longer stops its book syncing.**

- **Grimmory shelf-watch no longer retries a book with no ID on every scan.**

- **Readest no longer loses your position when a book was only just added.**

- **The suggested KOReader sync address uses the port you actually browse on.**

## Upgrading

Pull the new image and restart:

```bash
docker compose pull && docker compose up -d
```

Database migrations run automatically during container startup. Re-download
BridgeSync **0.9.6** on every KOReader device that uses the plugin, then restart
KOReader. Reading-status sync and shared history need the new plugin.

## Operational Notes

- **Forced alignment is on by default for new installs only.** Existing installs keep
  their current choice; turn on **Use forced alignment (recommended)** to adopt it.
  Existing alignment maps stay valid; remap a book to rebuild it.
- **The QuartzNet model needs internet once.** It downloads on first use; offline
  installs can point **QuartzNet model file** at a local copy.
- **Read-along EPUBs need BookOrbit as the audiobook source.**
- **BookOrbit's own read-along sync defaults to on** for qualifying books; see
  **BookOrbit → Read-Along Sync** if you prefer BookBridge to drive both sides.
- **Cross-device "device you're reading on wins" starts in watch-only mode**, and
  **shared recently-read history is off** until you enable it.
- **If a CWA book was aligned against the wrong ebook (#448)**, delete the duplicate
  `cwa_*.epub` files in `/data/epub_cache` (they share an identical checksum) and
  re-align those books. No other manual repair is required.
