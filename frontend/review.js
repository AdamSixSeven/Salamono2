(function () {
    "use strict";

    var grid = document.getElementById("reviewGrid");
    var emptyEl = document.getElementById("reviewEmpty");
    var resultCount = document.getElementById("resultCount");
    var filterMode = document.getElementById("filterMode");
    var filterSeverity = document.getElementById("filterSeverity");
    var filterKind = document.getElementById("filterKind");
    var filterSince = document.getElementById("filterSince");
    var filterUntil = document.getElementById("filterUntil");
    var applyBtn = document.getElementById("applyFilters");
    var clearBtn = document.getElementById("clearFilters");
    var exportBtn = document.getElementById("exportCsv");
    var quickDateBtns = document.querySelectorAll(".quick-dates .chip");
    var sumTotal = document.getElementById("sumTotal");
    var sumDanger = document.getElementById("sumDanger");
    var sumWarning = document.getElementById("sumWarning");
    var sumPpe = document.getElementById("sumPpe");
    var sumZone = document.getElementById("sumZone");
    var sumSite = document.getElementById("sumSite");
    var lightbox = document.getElementById("lightbox");
    var lightboxClose = document.getElementById("lightboxClose");
    var lightboxImg = document.getElementById("lightboxImg");
    var lightboxTitle = document.getElementById("lightboxTitle");
    var lightboxTime = document.getElementById("lightboxTime");
    var lightboxMode = document.getElementById("lightboxMode");
    var lightboxCamera = document.getElementById("lightboxCamera");
    var lightboxRule = document.getElementById("lightboxRule");
    var lightboxSeverity = document.getElementById("lightboxSeverity");
    var lightboxDetails = document.getElementById("lightboxDetails");

    var MODE_LABEL = { site: "Plac", checkpoint: "Bramka" };
    var records = [];

    function formatTime(ts) {
        var d = new Date(ts * 1000);
        return d.getFullYear() + "-" +
               String(d.getMonth() + 1).padStart(2, "0") + "-" +
               String(d.getDate()).padStart(2, "0") + " " +
               String(d.getHours()).padStart(2, "0") + ":" +
               String(d.getMinutes()).padStart(2, "0") + ":" +
               String(d.getSeconds()).padStart(2, "0");
    }

    function localDatetimeToTs(v) {
        if (!v) return null;
        var d = new Date(v);
        if (isNaN(d.getTime())) return null;
        return d.getTime() / 1000;
    }

    function activeSinceUntil() {
        return {
            since: localDatetimeToTs(filterSince.value),
            until: localDatetimeToTs(filterUntil.value),
        };
    }

    function buildQuery() {
        var qs = new URLSearchParams();
        qs.set("limit", "500");
        if (filterMode.value) qs.set("mode", filterMode.value);
        if (filterSeverity.value) qs.set("severity", filterSeverity.value);
        if (filterKind.value) qs.set("kind", filterKind.value);
        var r = activeSinceUntil();
        if (r.since !== null) qs.set("since", String(r.since));
        if (r.until !== null) qs.set("until", String(r.until));
        return qs.toString();
    }

    function buildSummaryQuery() {
        var qs = new URLSearchParams();
        var r = activeSinceUntil();
        if (r.since !== null) qs.set("since", String(r.since));
        if (r.until !== null) qs.set("until", String(r.until));
        return qs.toString();
    }

    function fetchRecords() {
        return fetch("/api/alerts?" + buildQuery(), {
            credentials: "same-origin",
        }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        });
    }

    function fetchSummary() {
        return fetch("/api/alerts/summary?" + buildSummaryQuery(), {
            credentials: "same-origin",
        }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        });
    }

    function renderSummary(s) {
        sumTotal.textContent = s.total;
        sumDanger.textContent = s.by_severity.DANGER || 0;
        sumWarning.textContent = s.by_severity.WARNING || 0;
        sumPpe.textContent = s.by_kind.ppe_missing || 0;
        sumZone.textContent = s.by_kind.zone_breach || 0;
        sumSite.textContent = s.by_kind.site_hazard || 0;
    }

    function render(list) {
        records = list;
        grid.innerHTML = "";
        resultCount.textContent = list.length + " zdarzen";
        if (list.length === 0) {
            emptyEl.classList.remove("hidden");
            return;
        }
        emptyEl.classList.add("hidden");
        list.forEach(function (rec, idx) {
            grid.appendChild(makeCard(rec, idx));
        });
    }

    function makeCard(rec, idx) {
        var card = document.createElement("div");
        card.className = "review-card";
        card.dataset.idx = idx;

        var thumb = document.createElement("div");
        thumb.className = "thumb";
        if (rec.thumbnail_url) {
            thumb.style.backgroundImage = "url('" + rec.thumbnail_url + "')";
        } else {
            thumb.classList.add("no-image");
            thumb.textContent = "BRAK ZDJECIA";
        }

        var sev = document.createElement("span");
        sev.className = "sev-tag " + rec.severity;
        sev.textContent = rec.severity;
        thumb.appendChild(sev);

        var mode = document.createElement("span");
        mode.className = "mode-tag";
        mode.textContent = (MODE_LABEL[rec.mode] || rec.mode).toUpperCase();
        thumb.appendChild(mode);

        var body = document.createElement("div");
        body.className = "card-body";
        var desc = document.createElement("div");
        desc.className = "desc";
        desc.textContent = rec.description;
        var time = document.createElement("div");
        time.className = "time";
        time.textContent = formatTime(rec.timestamp);
        body.appendChild(desc);
        body.appendChild(time);

        card.appendChild(thumb);
        card.appendChild(body);
        card.onclick = function () { openLightbox(rec); };
        return card;
    }

    function openLightbox(rec) {
        lightboxImg.src = rec.thumbnail_url || "";
        lightboxTitle.textContent = rec.description;
        lightboxTime.textContent = formatTime(rec.timestamp);
        lightboxMode.textContent = MODE_LABEL[rec.mode] || rec.mode;
        lightboxCamera.textContent = rec.camera_id;
        lightboxRule.textContent = rec.rule_name;
        lightboxSeverity.textContent = rec.severity;
        lightboxSeverity.className = "meta-val";
        lightboxDetails.textContent = JSON.stringify(rec.details, null, 2);
        lightbox.classList.remove("hidden");
    }

    function closeLightbox() {
        lightbox.classList.add("hidden");
        lightboxImg.src = "";
    }

    function refresh() {
        Promise.all([fetchRecords(), fetchSummary()])
            .then(function (results) { render(results[0]); renderSummary(results[1]); })
            .catch(function (e) { resultCount.textContent = "Błąd: " + e.message; });
    }

    function setQuickRange(range) {
        quickDateBtns.forEach(function (b) {
            b.classList.toggle("active", b.dataset.range === range);
        });
        var now = new Date();
        var startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        function toLocalInput(d) {
            var pad = function (n) { return String(n).padStart(2, "0"); };
            return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" +
                   pad(d.getDate()) + "T" + pad(d.getHours()) + ":" + pad(d.getMinutes());
        }
        if (range === "today") {
            filterSince.value = toLocalInput(startOfToday);
            filterUntil.value = "";
        } else if (range === "24h") {
            filterSince.value = toLocalInput(new Date(now.getTime() - 24 * 3600 * 1000));
            filterUntil.value = "";
        } else if (range === "7d") {
            filterSince.value = toLocalInput(new Date(now.getTime() - 7 * 24 * 3600 * 1000));
            filterUntil.value = "";
        } else {
            filterSince.value = "";
            filterUntil.value = "";
        }
        refresh();
    }

    quickDateBtns.forEach(function (btn) {
        btn.addEventListener("click", function () { setQuickRange(btn.dataset.range); });
    });

    applyBtn.onclick = refresh;
    clearBtn.onclick = function () {
        filterMode.value = "";
        filterSeverity.value = "";
        filterKind.value = "";
        filterSince.value = "";
        filterUntil.value = "";
        setQuickRange("all");
    };

    function csvEscape(v) {
        if (v === null || v === undefined) return "";
        var s = String(v);
        if (s.indexOf(",") >= 0 || s.indexOf("\"") >= 0 || s.indexOf("\n") >= 0) {
            return "\"" + s.replace(/"/g, "\"\"") + "\"";
        }
        return s;
    }

    exportBtn.onclick = function () {
        if (!records.length) return;
        var header = ["timestamp_iso", "timestamp_epoch", "mode", "kind",
                      "severity", "rule_name", "description", "camera_id",
                      "thumbnail_url"];
        var lines = [header.join(",")];
        records.forEach(function (rec) {
            var iso = new Date(rec.timestamp * 1000).toISOString();
            lines.push([
                iso, rec.timestamp, rec.mode, rec.kind, rec.severity,
                rec.rule_name, rec.description, rec.camera_id,
                rec.thumbnail_url || "",
            ].map(csvEscape).join(","));
        });
        var blob = new Blob([lines.join("\n") + "\n"],
                            { type: "text/csv;charset=utf-8" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "perimetr-audit-" + new Date().toISOString().slice(0, 10) + ".csv";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(url); }, 200);
    };
    lightboxClose.onclick = closeLightbox;
    lightbox.onclick = function (e) {
        if (e.target === lightbox) closeLightbox();
    };
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") closeLightbox();
    });

    refresh();
})();
