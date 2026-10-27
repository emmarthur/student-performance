(function () {
  const HELP_SELECTOR = 'a[href="/help"]';

  function save(key, data) {
    try {
      sessionStorage.setItem(key, JSON.stringify(data));
    } catch (_) {}
  }

  function load(key) {
    try {
      const raw = sessionStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  }

  function clear(key) {
    try {
      sessionStorage.removeItem(key);
    } catch (_) {}
  }

  function wireHelpSave(key, getState) {
    document.querySelectorAll(HELP_SELECTOR).forEach((link) => {
      link.addEventListener("click", () => save(key, getState()));
    });
  }

  window.AppSession = { save, load, clear, wireHelpSave };
})();
