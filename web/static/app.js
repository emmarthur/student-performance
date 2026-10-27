const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const statusMeta = document.getElementById("status-meta");
const samplesCard = document.getElementById("samples-card");
const sampleChips = document.getElementById("sample-chips");
const studentForm = document.getElementById("student-form");
const formGrid = document.getElementById("form-grid");
const errorEl = document.getElementById("error-text");
const resultCard = document.getElementById("result-card");
const resultClass = document.getElementById("result-class");
const barFail = document.getElementById("bar-fail");
const barPass = document.getElementById("bar-pass");
const pctFail = document.getElementById("pct-fail");
const pctPass = document.getElementById("pct-pass");

const SESSION_KEY = "student-performance-v1";

const LABELS = {
  age: "Age",
  Medu: "Mother education (0–4)",
  Fedu: "Father education (0–4)",
  traveltime: "Travel time to school (1–4)",
  studytime: "Weekly study time (1–4)",
  failures: "Past class failures",
  famrel: "Family relationship (1–5)",
  freetime: "Free time after school (1–5)",
  goout: "Going out with friends (1–5)",
  Dalc: "Workday alcohol (1–5)",
  Walc: "Weekend alcohol (1–5)",
  health: "Health (1–5)",
  absences: "School absences",
  sex: "Sex",
  school: "School",
  address: "Home address type",
  famsize: "Family size",
  Pstatus: "Parents cohabitation",
  Mjob: "Mother job",
  Fjob: "Father job",
  reason: "Reason to choose school",
  guardian: "Guardian",
  schoolsup: "Extra educational support",
  famsup: "Family educational support",
  paid: "Extra paid classes",
  activities: "Extra-curricular activities",
  nursery: "Attended nursery",
  higher: "Wants higher education",
  internet: "Internet at home",
  romantic: "Romantic relationship",
};

let schema = null;
let fieldValues = {};
let lastResult = null;
let pollTimer = null;

function persistState() {
  AppSession.save(SESSION_KEY, { fieldValues, lastResult });
}

function restoreState() {
  const saved = AppSession.load(SESSION_KEY);
  if (!saved) return;
  if (saved.fieldValues) fieldValues = saved.fieldValues;
  if (saved.lastResult) applyResult(saved.lastResult);
}

AppSession.wireHelpSave(SESSION_KEY, () => ({ fieldValues, lastResult }));

function setError(message) {
  errorEl.hidden = !message;
  errorEl.textContent = message || "";
}

function applyStatus(data) {
  statusDot.classList.remove("ready", "training", "error");
  if (data.error) {
    statusDot.classList.add("error");
    statusText.textContent = "Model failed to load.";
    statusMeta.hidden = false;
    statusMeta.textContent = data.error;
    return;
  }
  if (data.ready) {
    statusDot.classList.add("ready");
    statusText.textContent = "Model ready";
    statusMeta.hidden = false;
    statusMeta.textContent = `Holdout accuracy ${data.holdout_accuracy}% · CV ${data.cv_mean}% (±${data.cv_std}%) · ${data.pass_rate}% pass in data`;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    loadSchema(data.samples || []);
    return;
  }
  statusDot.classList.add("training");
  statusText.textContent = "Training decision tree…";
  statusMeta.hidden = true;
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    applyStatus(data);
  } catch {
    statusDot.classList.add("error");
    statusText.textContent = "Could not reach the server.";
  }
}

function buildForm() {
  if (!schema) return;
  formGrid.innerHTML = "";
  schema.fields.forEach((name) => {
    const wrap = document.createElement("label");
    wrap.className = "field";
    const label = document.createElement("span");
    label.className = "field-label";
    label.textContent = LABELS[name] || name;
    wrap.appendChild(label);

    const options = schema.options[name] || [];
    let control;
    if (typeof options[0] === "number" || name === "age" || name === "absences") {
      control = document.createElement("input");
      control.type = "number";
      control.name = name;
      control.value = fieldValues[name] ?? schema.defaults[name];
    } else {
      control = document.createElement("select");
      control.name = name;
      options.forEach((opt) => {
        const o = document.createElement("option");
        o.value = String(opt);
        o.textContent = String(opt);
        control.appendChild(o);
      });
      control.value = String(fieldValues[name] ?? schema.defaults[name]);
    }
    control.addEventListener("change", () => {
      fieldValues[name] =
        control.type === "number" ? Number(control.value) : control.value;
      persistState();
    });
    wrap.appendChild(control);
    formGrid.appendChild(wrap);
  });
  studentForm.hidden = false;
}

function loadSamples(samples) {
  sampleChips.innerHTML = "";
  samples.forEach((sample) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = sample.label;
    btn.title = `Actual: ${sample.actual}, predicted: ${sample.predicted}`;
    btn.addEventListener("click", () => {
      fieldValues = { ...sample.fields };
      buildForm();
      predict();
    });
    sampleChips.appendChild(btn);
  });
  samplesCard.hidden = samples.length === 0;
}

async function loadSchema(samples) {
  try {
    const res = await fetch("/api/schema");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Schema unavailable.");
    schema = data;
    if (!Object.keys(fieldValues).length) {
      fieldValues = { ...data.defaults };
    }
    buildForm();
    loadSamples(samples);
    restoreState();
    if (Object.keys(fieldValues).length) buildForm();
  } catch (err) {
    setError(err.message);
  }
}

function applyResult(data) {
  lastResult = data;
  resultCard.hidden = false;
  resultClass.textContent = data.prediction.toUpperCase();
  resultClass.className = "result-class " + data.prediction;
  const fail = data.probabilities.fail;
  const pass = data.probabilities.pass;
  barFail.style.width = `${fail}%`;
  barPass.style.width = `${pass}%`;
  pctFail.textContent = `${fail}%`;
  pctPass.textContent = `${pass}%`;
}

function collectFields() {
  const data = { ...fieldValues };
  studentForm.querySelectorAll("input, select").forEach((el) => {
    data[el.name] = el.type === "number" ? Number(el.value) : el.value;
  });
  fieldValues = data;
  return data;
}

async function predict() {
  setError(null);
  const fields = collectFields();
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fields }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");
    applyResult(data);
    persistState();
  } catch (err) {
    setError(err.message || "Prediction failed.");
  }
}

studentForm.addEventListener("submit", (event) => {
  event.preventDefault();
  predict();
});

pollTimer = setInterval(pollStatus, 1500);
pollStatus();
