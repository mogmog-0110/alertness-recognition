// 警告音・合図音と振動。鳴らす時点はサーバが決めて beep で送ってくるので、ここは鳴らし方だけ。

// 画面を見なくても何の警告か分かるよう、同じ「ピー」の音色で、鳴らす数と長さで分ける。
// 数だけに頼らないよう、長さと速さと高さも少しずつずらす。
// - 眠気: 長めのピーを 2 回（ピーピー）
// - 脇見: 短いピを 3 回（ピピピ）
// - 注意散漫: さらに短いピを 5 回、速く（ピピピピピ）
// どれも耳に刺さる 1〜4kHz を避け、スマホのスピーカーが出せる 780〜990Hz に置く。
// notes は [周波数Hz, 鳴らす秒, 次の音までの無音秒]。overtones は [倍音の次数, 強さ]。
const PATTERNS = {
  drowsy: {
    notes: [[784, 0.38, 0.16], [784, 0.38, 0]],
    attack: 0.01,
    overtones: [[1, 1], [2, 0.2]],
  },
  distracted: {
    notes: [[988, 0.1, 0.09], [988, 0.1, 0.09], [988, 0.1, 0]],
    attack: 0.005,
    overtones: [[1, 1], [2, 0.2]],
  },
  inattentive: {
    notes: [[880, 0.06, 0.05], [880, 0.06, 0.05], [880, 0.06, 0.05], [880, 0.06, 0.05],
            [880, 0.06, 0]],
    attack: 0.005,
    overtones: [[1, 1], [2, 0.2]],
  },
  ready: { notes: [[523, 0.16, 0]], attack: 0.02, overtones: [[1, 1], [2, 0.3]] },
  done: { notes: [[659, 0.12, 0.04], [988, 0.18, 0]], attack: 0.02, overtones: [[1, 1], [2, 0.3]] },
};
const GAIN = 0.22;
// MEDIUM は同じ形を小さく鳴らす。形を変えると 3 つの聞き分けが崩れる。
const MEDIUM_GAIN = 0.55;

let context = null;
let muted = false;

// 利用者の操作の中で呼ぶこと。iOS は操作を経ないと音を出させない。2 回目以降は同じものを使う。
export function prepareAudio() {
  if (context) return context.resume();
  try {
    // iOS 17 以降の Safari は、既定だと消音スイッチでページの音ごと消える。
    if (navigator.audioSession) navigator.audioSession.type = "playback";
  } catch (_) { /* 未対応なら従来どおり */ }
  context = new (window.AudioContext || window.webkitAudioContext)();
  return context.resume();
}

// 着信や裏へ回ったあと、AudioContext は止まったまま戻らないことがある。
export function resumeAudio() {
  if (context && context.state !== "running") context.resume().catch(() => {});
}

export function setMuted(value) { muted = value; }
export function isMuted() { return muted; }

export function play(kind, level = "high") {
  if (!context || muted) return;
  resumeAudio();
  const pattern = PATTERNS[kind] || PATTERNS.drowsy;
  const gain = level === "medium" ? GAIN * MEDIUM_GAIN : GAIN;
  let at = context.currentTime + 0.02;
  for (const [frequency, length, rest] of pattern.notes) {
    tone(frequency, length, gain, at, pattern);
    at += length + rest;
  }
}

function tone(frequency, length, gain, at, { attack, overtones }) {
  const envelope = context.createGain();
  envelope.gain.setValueAtTime(0, at);
  envelope.gain.linearRampToValueAtTime(gain, at + attack);
  envelope.gain.setValueAtTime(gain, at + Math.max(attack, length * 0.6));
  envelope.gain.linearRampToValueAtTime(0, at + length);
  envelope.connect(context.destination);
  for (const [ratio, weight] of overtones) {
    const oscillator = context.createOscillator();
    const mix = context.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = frequency * ratio;
    mix.gain.value = weight;
    oscillator.connect(mix).connect(envelope);
    oscillator.start(at);
    oscillator.stop(at + length + 0.02);
  }
}

// iOS は Vibration API に対応していない。非対応を理由に警告ごと落とさない。
export function vibrate(level) {
  if (!navigator.vibrate) return;
  navigator.vibrate(level === "high" ? [200, 80, 200] : [120]);
}
