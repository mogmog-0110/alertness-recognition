// 画面の描き分けと表示言語。何を出すかの判断は app.js、どう出すかはここ。

import { isMuted } from "./sound.js";

const translations = JSON.parse(document.getElementById("translations").textContent);
const LOCALE_KEY = "alertness.locale";
// 同じ警告は最低これだけ出し続ける。判定は毎秒 20〜30 回届き、軸や段が 0.1 秒単位で
// 入れ替わるので、そのまま描くと文言がちらついて読めない。
const HOLD_MS = 1500;
// 判定の一瞬の途切れを跨ぐ。危険が去れば、この後すぐに消える。
const CLEAR_MS = 700;
const LEVEL_RANK = { medium: 1, high: 2 };
const ICONS = { drowsiness: "drowsiness", distraction: "distraction",
                inattentive: "inattentive", stress: "stress" };

export const els = Object.fromEntries(
  ["message", "text", "reason", "spin", "status", "state", "sent", "dot", "mute", "recal",
   "go", "alert", "alert-icon", "alert-title", "alert-reason", "language-switch", "start-error",
   "sound-check", "sound-check-label"]
    .map((id) => [id.replace(/-(\w)/g, (_, c) => c.toUpperCase()), document.getElementById(id)]),
);

let locale = savedLocale();
let stateKey = "disconnected", connected = false, stalled = false, startErrorName = "";
let shown = null;   // いま出している警告 {key, level, cause, dimension, message, since, last}

function savedLocale() {
  // 保存が使えない環境（プライベートブラウズ等）でも日本語で動く。
  try {
    const saved = localStorage.getItem(LOCALE_KEY);
    if (saved && translations[saved]) return saved;
  } catch (_) { /* 既定に落とす */ }
  return "ja";
}

export function words() { return translations[locale]; }

function copy(template, values = {}) {
  return template.replace(/\{(\w+)\}/g, (_, key) => values[key] ?? "");
}

export function setLocale(next) {
  if (!translations[next]) return;
  locale = next;
  try { localStorage.setItem(LOCALE_KEY, next); } catch (_) { /* 保存できなくても切り替える */ }
  renderLocale();
}

export function renderLocale() {
  const w = words();
  document.documentElement.lang = locale;
  els.languageSwitch.setAttribute("aria-label", w.languageLabel);
  els.languageSwitch.querySelectorAll("button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.locale === locale));
  });
  els.go.textContent = w.start;
  els.soundCheck.setAttribute("aria-label", w.soundCheck);
  els.soundCheckLabel.textContent = w.soundCheck;
  const soundNames = {
    drowsy: w.soundDrowsy, distracted: w.soundDistracted, inattentive: w.soundInattentive,
  };
  els.soundCheck.querySelectorAll("button").forEach((button) => {
    button.textContent = soundNames[button.dataset.sound];
  });
  els.mute.textContent = isMuted() ? w.unmute : w.mute;
  els.recal.textContent = w.recalibrate;
  renderStatus();
  if (startErrorName) {
    els.startError.textContent = `${w.cameraError} (${startErrorName})`;
    els.startError.hidden = false;
  }
}

export function setStartError(name) {
  startErrorName = name;
  els.startError.hidden = !name;
  renderLocale();
}

export function setState(key, on) {
  stateKey = key;
  connected = !!on;
  if (!connected) stalled = false;
  renderStatus();
}

export function setStalled(value) {
  if (stalled === value) return;
  stalled = value;
  renderStatus();
}

function renderStatus() {
  const w = words();
  els.state.textContent = stalled ? w.stalled : w[stateKey];
  els.dot.classList.toggle("on", connected && !stalled);
  // 繋がっていない・判定が来ないことは、緑が消えるだけでは気づけない。
  els.status.classList.toggle("warn", stalled || (!connected && stateKey !== "disconnected"));
}

export function setSent(count) { els.sent.textContent = String(count); }

function setMessage(text, { spin = false, reason = "" } = {}) {
  els.text.textContent = text;
  els.spin.classList.toggle("on", spin);
  els.message.hidden = !text;
  els.reason.textContent = reason;
  els.reason.hidden = !reason;
}

export function clearMessage() { setMessage(""); }

export function showPreparing() { setMessage(words().preparing, { spin: true }); }

export function showCalibrating(waitingFor) {
  const w = words();
  const reasons = { hr_bpm: w.heartRateWait, face: w.calibratingFace, steady: w.calibratingSteady };
  setMessage(w.calibrating, { spin: true, reason: reasons[waitingFor] || "" });
}

export function showGuided(g) {
  const w = words();
  const prompt = w.prompts[g.prompt_key];
  const title = prompt ? prompt.title : (g.title || "");
  const instruction = prompt ? prompt.instruction : (g.instruction || "");
  if (g.step === "done") {
    setMessage(title, { reason: instruction });
    return;
  }
  const head = g.step === "hold" ? w.recording : w.next;
  const left = g.remaining != null ? `  ${Math.round(g.remaining)}s` : "";
  setMessage(`${head}: ${title}${left}`, { reason: instruction });
}

export function showAlert(result, now) {
  const next = {
    key: result.dimension_key || "",
    level: result.level === "high" ? "high" : "medium",
    cause: result.cause || "",
    dimension: result.dimension || "",
    message: result.message || "",
  };
  const same = shown && shown.key === next.key && shown.level === next.level;
  if (same) {
    shown = { ...shown, cause: next.cause || shown.cause, last: now };
  } else if (!shown || LEVEL_RANK[next.level] > LEVEL_RANK[shown.level]
             || now - shown.since >= HOLD_MS) {
    shown = { ...next, since: now, last: now };
  } else {
    shown = { ...shown, last: now };   // 出したばかりの警告を、弱い別の警告で上書きしない
    return;
  }
  renderAlert();
}

export function clearAlertIfStale(now) {
  if (!shown) return;
  if (now - shown.last < CLEAR_MS || now - shown.since < HOLD_MS) return;
  hideAlert();
}

export function hideAlert() {
  shown = null;
  els.alert.hidden = true;
}

function renderAlert() {
  const w = words();
  const faceMissing = shown.cause === "face_absent";
  const dimension = w.dimensions[shown.key] || shown.dimension || shown.key;
  let title = shown.message || w.warning;
  if (faceMissing) title = w.faceMissingTitle;
  else if (shown.key) title = copy(shown.level === "high" ? w.alertHigh : w.alertMedium, { dimension });
  const reason = faceMissing ? w.faceMissingHint : (w.causes[shown.cause] || "");
  const icon = faceMissing ? "face" : (ICONS[shown.key] || "drowsiness");
  els.alert.className = shown.level;
  els.alertTitle.textContent = title;
  els.alertReason.textContent = reason;
  els.alertReason.hidden = !reason;
  els.alertIcon.firstElementChild.setAttribute("href", `#icon-${icon}`);
  els.alert.hidden = false;
}
