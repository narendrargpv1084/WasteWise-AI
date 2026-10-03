/* ═══════════════════════════════════════════════════════════════
   WasteWise AI — main.js
   Responsibilities:
     1. Theme toggle (light / dark) with localStorage persistence
     2. Scroll-reveal for .reveal elements
     3. Smooth scroll for .scroll-cta
     4. Upload / camera / drag-and-drop / classify form logic
════════════════════════════════════════════════════════════════════ */

// ─────────────────────────────────────────────────────────────────
// 1. THEME TOGGLE
// ─────────────────────────────────────────────────────────────────
(function () {
  var STORAGE_KEY = 'wwai-theme';
  var root = document.documentElement;
  var toggleBtn = document.getElementById('theme-toggle');

  if (!toggleBtn) { return; }

  function getTheme() {
    return root.getAttribute('data-theme') || 'light';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
    toggleBtn.setAttribute(
      'aria-label',
      theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
    );
  }

  toggleBtn.addEventListener('click', function () {
    var next = getTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(next);
  });

  // Sync aria-label with initial state (set by inline script in <head>)
  applyTheme(getTheme());
})();


// ─────────────────────────────────────────────────────────────────
// 2. SCROLL REVEAL
// ─────────────────────────────────────────────────────────────────
(function () {
  var reveals = document.querySelectorAll('.reveal');
  if (!reveals.length) { return; }

  // Respect reduced motion
  var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReduced) {
    reveals.forEach(function (el) { el.classList.add('is-visible'); });
    return;
  }

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  reveals.forEach(function (el) { observer.observe(el); });
})();


// ─────────────────────────────────────────────────────────────────
// 3. SMOOTH SCROLL — .scroll-cta
// ─────────────────────────────────────────────────────────────────
(function () {
  var ctaLinks = document.querySelectorAll('.scroll-cta');
  ctaLinks.forEach(function (link) {
    link.addEventListener('click', function (e) {
      var href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        var target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });
})();


// ─────────────────────────────────────────────────────────────────
// 4. CLASSIFIER FORM
// ─────────────────────────────────────────────────────────────────
(function () {
  var form = document.getElementById('classify-form');
  if (!form) { return; }

  var input        = document.getElementById('image-input');
  var dropZone     = document.getElementById('drop-zone');
  var preview      = document.getElementById('preview');
  var previewWrap  = document.getElementById('preview-wrap');
  var dropCopy     = document.getElementById('drop-copy');
  var fileMeta     = document.getElementById('file-meta');
  var chooseBtn    = document.getElementById('choose-btn');
  var cameraBtn    = document.getElementById('camera-btn');
  var captureBtn   = document.getElementById('capture-btn');
  var clearBtn     = document.getElementById('clear-btn');
  var predictBtn   = document.getElementById('predict-btn');
  var clientError  = document.getElementById('client-error');
  var clientErrTxt = document.getElementById('client-error-text');
  var status       = document.getElementById('status');
  var cameraPreview = document.getElementById('camera-preview');
  var cameraCanvas  = document.getElementById('camera-canvas');
  var btnText       = predictBtn ? predictBtn.querySelector('.btn-text') : null;
  var btnLoading    = predictBtn ? predictBtn.querySelector('.btn-loading') : null;

  var allowed   = ['image/png', 'image/jpeg', 'image/webp', 'image/jpg'];
  var maxBytes  = 8 * 1024 * 1024;
  var cameraStream = null;

  /* ── helpers ─────────────────────────────────────────────────── */
  function showError(message) {
    if (clientErrTxt) { clientErrTxt.textContent = message; }
    if (clientError)  { clientError.classList.remove('hidden'); }
  }

  function clearError() {
    if (clientErrTxt) { clientErrTxt.textContent = ''; }
    if (clientError)  { clientError.classList.add('hidden'); }
  }

  function setReady(ready) {
    if (predictBtn) { predictBtn.disabled = !ready; }
    if (clearBtn)   { clearBtn.disabled   = !ready; }
  }

  function formatBytes(bytes) {
    if (bytes < 1024)        { return bytes + ' B'; }
    if (bytes < 1024 * 1024) { return (bytes / 1024).toFixed(1) + ' KB'; }
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(function (track) { track.stop(); });
      cameraStream = null;
    }
    if (cameraPreview) { cameraPreview.classList.add('hidden'); }
    if (captureBtn)    { captureBtn.classList.add('hidden'); }
  }

  function showPreview(file) {
    var url = URL.createObjectURL(file);
    preview.src = url;
    previewWrap.classList.remove('hidden');
    dropCopy.classList.add('hidden');
    if (fileMeta) {
      fileMeta.textContent = file.name + ' · ' + formatBytes(file.size);
    }
    setReady(true);
    if (status) { status.textContent = file.name + ' selected.'; }
  }

  function assignFile(file) {
    clearError();
    if (!file) { return; }

    var typeOk = allowed.indexOf(file.type) !== -1 ||
                 /\.(png|jpe?g|webp)$/i.test(file.name);
    if (!typeOk) {
      showError('Please choose a PNG, JPG, or WEBP image.');
      return;
    }
    if (file.size > maxBytes) {
      showError('That file is larger than 8 MB.');
      return;
    }

    var transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    stopCamera();
    showPreview(file);
  }

  function clearSelection() {
    input.value = '';
    preview.removeAttribute('src');
    previewWrap.classList.add('hidden');
    dropCopy.classList.remove('hidden');
    if (fileMeta) { fileMeta.textContent = ''; }
    setReady(false);
    if (status) { status.textContent = ''; }
    clearError();
    stopCamera();
  }

  /* ── events ──────────────────────────────────────────────────── */
  chooseBtn.addEventListener('click', function () { input.click(); });

  input.addEventListener('change', function () {
    if (input.files && input.files[0]) {
      assignFile(input.files[0]);
    }
  });

  // Drag events
  ['dragenter', 'dragover'].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      dropZone.classList.add('is-dragover');
    });
  });

  ['dragleave', 'drop'].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      dropZone.classList.remove('is-dragover');
    });
  });

  dropZone.addEventListener('drop', function (event) {
    var files = event.dataTransfer.files;
    if (files && files[0]) {
      assignFile(files[0]);
    }
  });

  // Keyboard activate
  dropZone.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      input.click();
    }
  });

  clearBtn.addEventListener('click', clearSelection);

  // Camera
  cameraBtn.addEventListener('click', function () {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError('Camera capture is not supported in this browser.');
      return;
    }
    clearError();
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment' } })
      .then(function (stream) {
        cameraStream = stream;
        cameraPreview.srcObject = stream;
        cameraPreview.classList.remove('hidden');
        captureBtn.classList.remove('hidden');
        previewWrap.classList.add('hidden');
        dropCopy.classList.add('hidden');
        if (status) { status.textContent = 'Camera ready. Capture a still image.'; }
      })
      .catch(function () {
        showError('Could not access the camera. Check browser permissions.');
      });
  });

  captureBtn.addEventListener('click', function () {
    if (!cameraStream) { return; }
    var width  = cameraPreview.videoWidth  || 224;
    var height = cameraPreview.videoHeight || 224;
    cameraCanvas.width  = width;
    cameraCanvas.height = height;
    var ctx = cameraCanvas.getContext('2d');
    ctx.drawImage(cameraPreview, 0, 0, width, height);
    cameraCanvas.toBlob(function (blob) {
      if (!blob) {
        showError('Could not capture a frame from the camera.');
        return;
      }
      var file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
      assignFile(file);
    }, 'image/jpeg', 0.92);
  });

  // Submit
  form.addEventListener('submit', function (event) {
    if (!input.files || !input.files[0]) {
      event.preventDefault();
      showError('Choose or capture an image before classifying.');
      return;
    }

    // Show loading state
    if (btnText)    { btnText.classList.add('hidden'); }
    if (btnLoading) { btnLoading.classList.remove('hidden'); }
    if (predictBtn) { predictBtn.disabled = true; }
    if (status)     { status.textContent = 'Running the MobileNetV2 classifier…'; }
  });
})();
