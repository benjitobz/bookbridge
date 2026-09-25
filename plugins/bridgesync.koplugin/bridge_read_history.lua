-- Files other devices' reads into this device's KOReader reading history.
--
-- The cross-device statistics merge already tells this device that a book was
-- read and when, but KOReader writes History only when YOU open a book here.
-- So a book read on another device never appears in History, nor in anything
-- built from it -- which is why a "Recent" shelf can look empty on one device
-- while the same book sits at the top of it on another, even though progress,
-- status and stats have all synced.
--
-- Bounded on purpose. KOReader's History is a short "what am I reading" list,
-- not an archive, and it has its own `history_size` cap (default 500). Without
-- an age limit and a per-pass cap, the first sync on a large library would file
-- hundreds of old reads at once and evict the device's genuine history.

local ReadHistoryMerge = {}

ReadHistoryMerge.MAX_AGE_DAYS = 30
ReadHistoryMerge.MAX_PER_PASS = 25

-- Existing history paths, keyed by lowercase spelling.
--
-- ReadHistory:getIndexByFile compares path STRINGS, but the managed folder can
-- legitimately be spelled differently in two places -- BridgeSync's configured
-- download_dir versus the on-disk directory -- and on a case-insensitive
-- filesystem (FAT, as on a Kobo or Kindle) both open the same file. Adding our
-- spelling then looks like a different book and appends a second entry instead
-- of replacing the existing one.
local function existingPathsByLowercase(read_history)
    local canon = {}
    local hist = read_history and read_history.hist
    if type(hist) ~= "table" then return canon end
    for _, item in ipairs(hist) do
        local file = item and item.file
        if type(file) == "string" and canon[file:lower()] == nil then
            canon[file:lower()] = file
        end
    end
    return canon
end

-- Drop history entries that differ from an earlier one only by case, keeping
-- the newest. Heals lists already polluted by the mismatch above; a no-op once
-- the merge is using the existing spelling.
-- @treturn number removed
function ReadHistoryMerge.dedupeCaseVariants(read_history)
    local hist = read_history and read_history.hist
    if type(hist) ~= "table" then return 0 end

    local kept, seen, removed = {}, {}, 0
    for _, item in ipairs(hist) do          -- newest first, so the first wins
        local file = item and item.file
        local key = type(file) == "string" and file:lower() or nil
        if key and seen[key] then
            removed = removed + 1
        else
            if key then seen[key] = true end
            table.insert(kept, item)
        end
    end
    if removed > 0 then
        -- ReadHistory holds this table by reference; replace its contents in place.
        for i = #hist, 1, -1 do hist[i] = nil end
        for i, item in ipairs(kept) do hist[i] = item end
    end
    return removed
end

-- @param read_history table  KOReader's ReadHistory (injected for testability)
-- @param hash_index table    md5 -> absolute local file path
-- @param latest_by_md5 table md5 -> newest foreign read timestamp
-- @param opts table          { now = <epoch>, max_age_days = , max_per_pass = }
-- @treturn table { added, unchanged, skipped, too_old, capped }
function ReadHistoryMerge.merge(read_history, hash_index, latest_by_md5, opts)
    opts = opts or {}
    local totals = { added = 0, unchanged = 0, skipped = 0, too_old = 0, capped = 0, deduped = 0 }
    if not latest_by_md5 or next(latest_by_md5) == nil then
        return totals
    end
    if type(read_history) ~= "table" or type(read_history.addItem) ~= "function" then
        return totals
    end

    -- Heal any case-variant duplicates a previous pass created before the
    -- canonical-spelling fix below existed.
    local deduped = ReadHistoryMerge.dedupeCaseVariants(read_history)
    local canon = existingPathsByLowercase(read_history)

    local now = opts.now or os.time()
    local max_age = (opts.max_age_days or ReadHistoryMerge.MAX_AGE_DAYS) * 86400
    local cap = opts.max_per_pass or ReadHistoryMerge.MAX_PER_PASS
    local cutoff = now - max_age
    hash_index = hash_index or {}

    -- Newest first, so a capped pass files the most recent reads rather than
    -- whichever ones the hash table happened to yield first.
    local ordered = {}
    for md5, ts in pairs(latest_by_md5) do
        if type(ts) == "number" then
            table.insert(ordered, { md5 = md5, ts = ts })
        end
    end
    table.sort(ordered, function(a, b)
        if a.ts == b.ts then return tostring(a.md5) < tostring(b.md5) end
        return a.ts > b.ts
    end)

    local added_any = false
    for _, entry in ipairs(ordered) do
        local path = hash_index[entry.md5]
        if not path then
            -- Read on another device, but this one does not hold the file.
            totals.skipped = totals.skipped + 1
        elseif entry.ts < cutoff then
            totals.too_old = totals.too_old + 1
        elseif totals.added >= cap then
            totals.capped = totals.capped + 1
        else
            -- no_flush: one write at the end instead of one per book.
            --
            -- addItem RETURNS the outcome -- true when it filed the entry, nil
            -- when it declined (already present at this timestamp, or the path
            -- is not a readable file). pcall only reports whether it raised, so
            -- counting on pcall alone reports success for entries that were
            -- never added, which is exactly how an earlier version of this
            -- logged "added 1" on every sync while nothing ever persisted.
            -- Reuse the spelling already in the history when it differs only by
            -- case, so addItem REPLACES that entry rather than appending a twin.
            local target = canon[path:lower()] or path
            local ok, filed = pcall(read_history.addItem, read_history, target, math.floor(entry.ts), true)
            if not ok then
                totals.skipped = totals.skipped + 1
            elseif filed then
                totals.added = totals.added + 1
                added_any = true
            else
                -- Idempotent no-op: this read is already in the history.
                totals.unchanged = totals.unchanged + 1
            end
        end
    end

    totals.deduped = deduped
    if added_any or deduped > 0 then
        -- _reduce() enforces KOReader's own history_size cap before writing.
        pcall(function()
            if read_history._reduce then read_history:_reduce() end
            if read_history._flush then read_history:_flush() end
        end)
    end

    return totals
end

return ReadHistoryMerge
