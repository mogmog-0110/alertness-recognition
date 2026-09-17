// 端末はカメラとスピーカーだけ。判定は PC 側が行う。
// 送信: [8 byte 撮影時刻 (float64 LE)][JPEG]   受信: 判定 JSON

import { isMuted, play, prepareAudio, resumeAudio, setMuted, vibrate } from "./sound.js";
import * as view from "./view.js";

const { els } = view;
const FPS = 30, QUALITY = 0.7, WIDTH = 1280, HEIGHT = 720;
// 送る映像の長辺の上限。端末が要求より大きい解像度を返しても、帯域を食わせない。
const MAX_SIDE = 1280;
// サーバ側の Wi-Fi が落ちると、ブラウザの WebSocket は TCP タイムアウトまで
// OPEN のまま残ることがある。判定とは別の応答で半開きを短時間で見つける。
const HEARTBEAT_INTERVAL_MS = 1000, HEARTBEAT_TIMEOUT_MS = 4000;
const CONNECT_TIMEOUT_MS = 5000;
// pong は判定ループを通らずに返るので、繋がっていても判定が止まっていることがある。
const STALL_MS = 3000;
// 繋がった直後はまだ端末を構え直している最中で、その姿勢が基準になると以後ずっと
// 「下を向いている」扱いになる。
const CALIBRATE_DELAY_MS = 4000;
// ページを開くたびに変わる識別子。サーバはこれで「張り直し」と「開き直し」を分け、
// 開き直しなら前の人の基準を捨てる。
const SESSION = self.crypto && crypto.randomUUID
  ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const video = document.getElementById("view");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d", { alpha: false });

let socket = null, sent = 0, inFlight = 0, lastSent = 0;
let lastReply = 0, lastResult = 0, reconnectTimer = null;
// 利用者が繋いだ新しいセッションなら測り直す。通信が切れて自力で張り直したときは
// 測り直さない（測っている間は正面を見ている保証がなく、運転中に基準が飛ぶ）。
let calibrateOnOpen = true, retryDelay = 1000;
let phase = "";   // 最後に受けた判定の phase。切り替わりで表示を片付ける
let calibrateTimer = null;
let wakeLock = null;

function sendJson(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

function recalibrate() { sendJson({ command: "recalibrate" }); }

function handle(result) {
  const now = performance.now();
  const previous = phase;
  phase = result.phase || "running";
  lastResult = now;
  view.setStalled(false);
  if (previous === "calibrating" && phase !== "calibrating") play("done");

  if (phase === "preparing") {
    view.hideAlert();
    view.showPreparing();
    return;
  }
  if (phase === "calibrating") {
    // 前方を見ている運転者は画面を読めないので、始まりと終わりを音で知らせる。
    if (previous !== "calibrating") play("ready");
    view.hideAlert();
    view.showCalibrating(result.waiting_for || "");
    return;
  }
  if (phase === "guided" && result.guided) {
    view.hideAlert();
    view.showGuided(result.guided);
    return;
  }
  if (previous !== "running") view.clearMessage();
  if (result.alert) view.showAlert(result, now);
  else view.clearAlertIfStale(now);
  if (result.beep) {
    play(result.beep.sound, result.beep.level);
    vibrate(result.beep.level);   // 鳴らすときだけ。毎回呼ぶと振動が途切れず続く
  }
}

function connect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  view.setState("connecting", false);
  const ws = new WebSocket(`wss://${location.host}/ws`);
  socket = ws;
  ws.binaryType = "arraybuffer";
  // 接続試行中に Wi-Fi が切れていると、OPEN にも close にも進まず長く止まる
  // ブラウザがある。試行自体にも期限を設けて次の接続へ進む。
  const openTimer = setTimeout(() => {
    if (socket === ws && ws.readyState !== WebSocket.OPEN) abandon(ws);
  }, CONNECT_TIMEOUT_MS);
  ws.onopen = () => {
    if (socket !== ws) {
      // 期限切れ後に古い試行だけが成功すると、サーバーの返送先を奪う。
      try { ws.close(); } catch (_) { /* 閉じられなくても置き換え済み */ }
      return;
    }
    clearTimeout(openTimer);
    lastReply = lastResult = performance.now();
    view.setState("connected", true);
    retryDelay = 1000;   // 繋がったので次に落ちたら即座に試す
    ws.send(JSON.stringify({ type: "hello", session: SESSION }));
    if (!calibrateOnOpen) return;
    calibrateOnOpen = false;
    calibrateTimer = setTimeout(recalibrate, CALIBRATE_DELAY_MS);
  };
  ws.onclose = () => {
    if (socket !== ws) return;  // 置き換え済みの古い接続からの通知
    clearTimeout(openTimer);
    lose();
    // 相手が落ちている間ずっと同じ間隔で叩いても意味が無い。倍々で最大 10 秒。
    reconnectTimer = setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 10000);
  };
  ws.onerror = () => {
    if (socket === ws) view.setState("connectionError", false);
  };
  ws.onmessage = (event) => {
    if (socket !== ws) return;
    lastReply = performance.now();
    let result;
    try { result = JSON.parse(event.data); } catch (_) { return; }
    if (result.type !== "pong") handle(result);
  };
}

// 切れた接続の表示を片付ける。古い警告を出したままにすると、動いているように見える。
function lose() {
  socket = null;
  phase = "";
  view.hideAlert();
  view.setState("reconnecting", false);
}

// close イベント自体が半開きでは遅れるため、先に現接続から切り離して新しい
// ソケットを作る。古い onclose は socket !== ws となり無視される。
function abandon(ws) {
  lose();
  try { ws.close(); } catch (_) { /* 切り離し済み */ }
  connect();
}

function checkConnection() {
  const ws = socket;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const now = performance.now();
  if (now - lastReply > HEARTBEAT_TIMEOUT_MS) {
    abandon(ws);
    return;
  }
  const silent = now - lastResult > STALL_MS;
  view.setStalled(silent);
  if (silent) view.hideAlert();
  try {
    ws.send(JSON.stringify({ type: "ping" }));
  } catch (_) {
    abandon(ws);
  }
}

function sendFrame(now) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  if (now - lastSent < 1000 / FPS) return;
  // 変換は 1 枚ずつ。並べると完了順が撮影順と入れ替わり、時刻の戻ったフレームが届く。
  if (inFlight > 0) return;
  // 送信待ちが残っている間は撮らない。混んだ無線で溜めると、届く映像も警告も
  // 実時間からどんどん遅れていく。
  if (socket.bufferedAmount > 0) return;
  if (!fitCanvas()) return;
  lastSent = now;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  inFlight++;
  const target = socket;   // 変換中に張り直したら、古い接続向けに撮った 1 枚は捨てる
  canvas.toBlob(async (blob) => {
    try {
      if (!blob || socket !== target || target.readyState !== WebSocket.OPEN) return;
      const jpeg = await blob.arrayBuffer();
      if (socket !== target) return;
      const out = new ArrayBuffer(8 + jpeg.byteLength);
      // 撮影時刻。到着時刻で代用すると、詰まった分だけ生理指標が歪む。
      new DataView(out).setFloat64(0, now / 1000, true);
      new Uint8Array(out, 8).set(new Uint8Array(jpeg));
      target.send(out);
      view.setSent(++sent);
    } catch (_) {
      /* この 1 枚は落とす。次のフレームで送る */
    } finally {
      inFlight--;   // 戻し損ねると、以後 1 枚も送られなくなる
    }
  }, "image/jpeg", QUALITY);
}

// 端末を回すと映像の縦横が入れ替わる。キャンバスを開始時の大きさのままにすると
// 顔が引き伸ばされて送られ、目の開きの比も頭の向きも狂う。
function fitCanvas() {
  const width = video.videoWidth, height = video.videoHeight;
  if (!width || !height) return false;
  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  const w = Math.round(width * scale), h = Math.round(height * scale);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return true;
}

function loop(now) {
  sendFrame(now);
  requestAnimationFrame(loop);
}

async function startCamera() {
  // 鏡像にしない。PC 側の webcam と同じ幾何にする。
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: WIDTH }, height: { ideal: HEIGHT },
             frameRate: { ideal: FPS } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

// 画面が消えると撮影ごと止まる。解放されたら持ち直せるよう、手放した時点で空にする。
async function keepAwake() {
  if (!("wakeLock" in navigator) || wakeLock) return;
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    wakeLock.addEventListener("release", () => { wakeLock = null; });
  } catch (_) { /* 取れなくても続行する */ }
}

async function recoverFromBackground() {
  keepAwake();
  resumeAudio();
  const track = video.srcObject && video.srcObject.getVideoTracks()[0];
  try {
    if (!track || track.readyState === "ended") await startCamera();
    else if (video.paused) await video.play();
  } catch (_) { /* 次に前面へ来たときにもう一度試す */ }
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && video.srcObject) recoverFromBackground();
});

els.languageSwitch.querySelectorAll("button").forEach((button) => {
  button.onclick = () => view.setLocale(button.dataset.locale);
});
// 開始前に 2 つの警告音を聞き比べる。押す操作そのものが、音を出す許可の代わりになる。
els.soundCheck.querySelectorAll("button").forEach((button) => {
  button.onclick = async () => {
    await prepareAudio();
    play(button.dataset.sound, "high");
  };
});
els.recal.onclick = () => {
  clearTimeout(calibrateTimer);   // 予約した測定が後から走ると、測っている途中で取り直しになる
  recalibrate();
};
els.mute.onclick = () => {
  setMuted(!isMuted());
  view.renderLocale();
};

els.go.onclick = async () => {
  els.go.disabled = true;
  view.setStartError("");
  try {
    // 「はじめる」を押させているのは、一度ユーザー操作を挟まないと音を鳴らせないため。
    await prepareAudio();
    await startCamera();
    document.getElementById("start").remove();
    keepAwake();
    connect();
    setInterval(checkConnection, HEARTBEAT_INTERVAL_MS);
    requestAnimationFrame(loop);
  } catch (err) {
    view.setStartError(err && err.name ? err.name : "Error");
    els.go.disabled = false;
  }
};

view.renderLocale();
