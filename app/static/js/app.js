/* MammoRisk+ progressive enhancement. The site works without this file; it only adds:
   - a light/dark theme toggle (remembered in this browser)
   - drag-and-drop highlighting and file names on the upload pages
   - a "working" state on the submit button so a slow upload is not clicked twice */
(function () {
  "use strict";

  var root = document.documentElement;
  var toggle = document.querySelector("[data-theme-toggle]");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var current = root.dataset.theme ||
        (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      var next = current === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("mm-theme", next); } catch (e) { /* private mode: keep for this page only */ }
    });
  }

  document.querySelectorAll(".drop").forEach(function (drop) {
    var input = drop.querySelector("input[type=file]");
    var label = drop.querySelector("[data-file]");
    if (!input || !label) { return; }

    function show() {
      var f = input.files && input.files[0];
      if (!f) { label.textContent = "No file chosen"; drop.classList.remove("has-file"); return; }
      var mb = f.size / (1024 * 1024);
      label.textContent = f.name + " (" + (mb >= 0.1 ? mb.toFixed(1) + " MB" : Math.max(1, Math.round(f.size / 1024)) + " KB") + ")";
      drop.classList.add("has-file");
    }

    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach(function (ev) {
      drop.addEventListener(ev, function () { drop.classList.add("is-over"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      drop.addEventListener(ev, function () { drop.classList.remove("is-over"); });
    });
  });

  var form = document.querySelector("[data-upload-form]");
  if (form) {
    form.addEventListener("submit", function () {
      var btn = form.querySelector("[data-submit]");
      if (btn) { btn.disabled = true; btn.textContent = "Analysing…"; }
    });
  }
})();
