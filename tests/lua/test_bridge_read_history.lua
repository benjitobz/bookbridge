-- Executable coverage for bridge_read_history.lua.
--
-- KOReader writes History only when YOU open a book on that device, so a book
-- read elsewhere never appears there even once its stats and status have
-- synced -- which is why a "Recent" shelf built from History can look empty on
-- one device while the same book sits at the top of it on another. This merge
-- closes that gap, bounded so a first sync cannot bury the real history.

local plugin_dir = assert(arg[1], "plugin directory argument required")
package.path = plugin_dir .. "/?.lua;" .. package.path

local Merge = require("bridge_read_history")

local added, flushed, reduced
local history = {
    addItem = function(_, file, ts, no_flush)
        assert(type(file) == "string", "addItem needs a path")
        table.insert(added, { file = file, ts = ts, no_flush = no_flush })
        return true
    end,
    _reduce = function() reduced = reduced + 1 end,
    _flush = function() flushed = flushed + 1 end,
}
local function reset(hist) added, flushed, reduced = {}, 0, 0; history.hist = hist or {} end
local function entry(file, ts) return { file = file, time = ts } end

local NOW = 1789000000

-- A book read elsewhere, whose file IS here, gets a history entry.
do
    reset()
    local t = Merge.merge(history, { aaa = "/books/here.epub" }, { aaa = NOW - 3600 }, { now = NOW })
    assert(t.added == 1, "added=" .. t.added)
    assert(added[1].file == "/books/here.epub")
    assert(added[1].ts == NOW - 3600, "must use the foreign read's own timestamp")
    assert(added[1].no_flush == true, "writes are deferred to one flush")
    assert(flushed == 1 and reduced == 1, "exactly one reduce+flush per pass")
end

-- A book this device does not hold is skipped: there is nothing to open.
do
    reset()
    local t = Merge.merge(history, {}, { aaa = NOW - 3600 }, { now = NOW })
    assert(t.added == 0 and t.skipped == 1, "skipped=" .. t.skipped)
    assert(flushed == 0, "nothing added means nothing written")
end

-- Old reads do not resurface. History is not an archive.
do
    reset()
    local t = Merge.merge(history, { aaa = "/books/old.epub" },
        { aaa = NOW - (40 * 86400) }, { now = NOW })
    assert(t.added == 0 and t.too_old == 1, "too_old=" .. t.too_old)
end

-- A first sync cannot bury the device's real history.
do
    reset()
    local index, reads = {}, {}
    for i = 1, 60 do
        local k = string.format("md5%02d", i)
        index[k] = "/books/b" .. i .. ".epub"
        reads[k] = NOW - (i * 60)
    end
    local t = Merge.merge(history, index, reads, { now = NOW })
    assert(t.added == 25, "cap must hold, added=" .. t.added)
    assert(t.capped == 35, "capped=" .. t.capped)
    assert(#added == 25)
end

-- A capped pass keeps the NEWEST reads, not an arbitrary hash order.
do
    reset()
    local index, reads = {}, {}
    for i = 1, 40 do
        local k = string.format("md5%02d", i)
        index[k] = "/books/b" .. i .. ".epub"
        reads[k] = NOW - (i * 3600)   -- i == 1 is the most recent
    end
    Merge.merge(history, index, reads, { now = NOW })
    local oldest_allowed = NOW - (25 * 3600)
    local saw_newest = false
    for _, a in ipairs(added) do
        if a.ts == NOW - 3600 then saw_newest = true end
        assert(a.ts >= oldest_allowed, "a capped pass kept an older read over a newer one")
    end
    assert(saw_newest, "the most recent read must be filed")
end

-- Nothing to merge is a clean no-op.
do
    reset()
    assert(Merge.merge(history, { aaa = "/books/x.epub" }, {}, { now = NOW }).added == 0)
    assert(Merge.merge(history, { aaa = "/books/x.epub" }, nil, { now = NOW }).added == 0)
    assert(flushed == 0)
end

-- addItem declining (already present) is "unchanged", not "added". Counting on
-- pcall alone reported success for entries that were never filed -- the bug that
-- made an earlier version log "added 1" every sync while nothing persisted.
do
    reset()
    local original = history.addItem
    history.addItem = function() return nil end   -- declines, as KOReader does for a dupe
    local t = Merge.merge(history, { aaa = "/books/dupe.epub" }, { aaa = NOW - 60 }, { now = NOW })
    history.addItem = original
    assert(t.added == 0, "a declined add must not count as added, added=" .. t.added)
    assert(t.unchanged == 1, "unchanged=" .. t.unchanged)
    assert(flushed == 0, "nothing filed means nothing written")
end

-- A failing addItem must not take the sync down.
do
    reset()
    local original = history.addItem
    history.addItem = function() error("history is locked") end
    local t = Merge.merge(history, { aaa = "/books/boom.epub" }, { aaa = NOW - 60 }, { now = NOW })
    history.addItem = original
    assert(t.added == 0 and t.skipped == 1, "a raising addItem counts as skipped, not fatal")
end

-- An unusable ReadHistory is tolerated rather than fatal.
do
    reset()
    local t = Merge.merge(nil, { aaa = "/books/x.epub" }, { aaa = NOW - 60 }, { now = NOW })
    assert(t.added == 0)
    local t2 = Merge.merge({}, { aaa = "/books/x.epub" }, { aaa = NOW - 60 }, { now = NOW })
    assert(t2.added == 0)
end

-- ---------------------------------------------------------------------------
-- Case-variant paths. ReadHistory compares path STRINGS, but the managed folder
-- can be spelled differently in BridgeSync's download_dir than on disk, and a
-- device filesystem (FAT) is case-insensitive -- so a naive add appends a twin
-- of a book that is already listed.
-- ---------------------------------------------------------------------------

-- The existing spelling is reused, so addItem can replace rather than append.
do
    reset({ entry("/mnt/onboard/KoreaderBooks/A.epub", NOW - 7200) })
    Merge.merge(history, { aaa = "/mnt/onboard/Koreaderbooks/A.epub" },
        { aaa = NOW - 60 }, { now = NOW })
    assert(#added == 1)
    assert(added[1].file == "/mnt/onboard/KoreaderBooks/A.epub",
        "must reuse the history's own spelling, got " .. added[1].file)
end

-- A path with no case-variant in the history is used as-is.
do
    reset({ entry("/mnt/onboard/KoreaderBooks/Other.epub", NOW - 7200) })
    Merge.merge(history, { aaa = "/mnt/onboard/Koreaderbooks/New.epub" },
        { aaa = NOW - 60 }, { now = NOW })
    assert(added[1].file == "/mnt/onboard/Koreaderbooks/New.epub")
end

-- Duplicates already created by an earlier pass are healed, newest kept.
do
    reset({
        entry("/mnt/onboard/Koreaderbooks/A.epub", NOW - 60),    -- newest, our spelling
        entry("/mnt/onboard/KoreaderBooks/A.epub", NOW - 7200),  -- older twin
        entry("/mnt/onboard/KoreaderBooks/B.epub", NOW - 100),
    })
    local removed = Merge.dedupeCaseVariants(history)
    assert(removed == 1, "removed=" .. removed)
    assert(#history.hist == 2)
    assert(history.hist[1].file == "/mnt/onboard/Koreaderbooks/A.epub", "the newest twin survives")
    assert(history.hist[2].file == "/mnt/onboard/KoreaderBooks/B.epub")
end

-- A clean history is left exactly as it is.
do
    reset({ entry("/books/A.epub", NOW), entry("/books/B.epub", NOW - 1) })
    assert(Merge.dedupeCaseVariants(history) == 0)
    assert(#history.hist == 2)
end

-- merge() heals duplicates and writes even when it adds nothing itself.
do
    reset({
        entry("/mnt/onboard/Koreaderbooks/A.epub", NOW - 60),
        entry("/mnt/onboard/KoreaderBooks/A.epub", NOW - 7200),
    })
    local t = Merge.merge(history, {}, { zzz = NOW - 60 }, { now = NOW })
    assert(t.deduped == 1, "deduped=" .. tostring(t.deduped))
    assert(flushed == 1, "a heal must be written even with nothing added")
end

print("BridgeSync reading-history merge tests passed")
