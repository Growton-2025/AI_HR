import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import axios from 'axios';

import { API_BASE } from '../store/useAppStore';

const VoIPContext = createContext({
  activeCall: null,
  answerCall: async () => ({ success: false }),
  rejectCall: async () => ({ success: false }),
  placeCall: async () => ({ success: false }),
  ensureMicrophonePermission: async () => ({ success: true }),
  waitForPlivoDial: async () => ({ success: false }),
  startDialTone: () => {},
  stopDialTone: () => {},
  voipStatus: 'disconnected',
  voipError: '',
  voipErrorCode: '',
  voipActionLabel: '',
  voipActionUrl: '',
  voipMeta: null,
  voipConnectionEvent: null,
  voipCallEvent: null,
  agentEmail: '',
  endpointUsername: '',
  voipDegraded: null,
  retryVoip: () => {},
});

const RELOGIN_THROTTLE_MS = 4000;
const RELOGIN_WAIT_AT_DIAL_MS = 8000;
const PLIVO_SDK_URL = '/plivo.min.js';
const PLIVO_SDK_LOAD_TIMEOUT_MS = 15000;
const PLIVO_LOGIN_TIMEOUT_MS = 20000;
const PLIVO_DIAL_HANDSHAKE_TIMEOUT_MS = 12000;
const PLIVO_DIAL_HANDSHAKE_POLL_MS = 400;
// Sends the per-attempt X-PH-DialToken on each outbound dial so the backend can
// attribute the call to an exact `calls` row. See docs/call-attribution-plan.md.
// Flip to false to fall back to username matching everywhere.
const SEND_DIAL_TOKEN = true;

let plivoSdkPromise = null;

// Diagnostic beacon: mirror timing logs into the backend log so call-setup
// latency can be debugged without access to the recruiter's browser console.
export const reportTiming = (leg, ms, detail = '') => {
  console.log(`[VoIP][timing] ${leg}: ${Math.round(ms)}ms${detail ? ` (${detail})` : ''}`);
  try {
    fetch(`${API_BASE}/plivo/client-timing`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ leg, ms: Math.round(ms), detail }),
    }).catch(() => {});
  } catch (_) {
    // Beacon must never affect the call path.
  }
};

const resolvePlivoConstructor = () => {
  const ctor = window.Plivo
    || globalThis.Plivo
    || Function('return typeof Plivo !== "undefined" ? Plivo : undefined')();
  if (ctor) {
    window.Plivo = ctor;
  }
  return ctor;
};

const loadPlivoSdk = () => {
  const existing = resolvePlivoConstructor();
  if (existing) {
    return Promise.resolve(existing);
  }

  if (plivoSdkPromise) {
    return plivoSdkPromise;
  }

  plivoSdkPromise = new Promise((resolve, reject) => {
    const existingScripts = Array.from(document.scripts)
      .filter(script => script.src === PLIVO_SDK_URL || script.src.endsWith('/plivo.min.js'));
    existingScripts.forEach(script => script.remove());

    const script = document.createElement('script');
    script.src = PLIVO_SDK_URL;
    script.async = true;

    const timeoutId = window.setTimeout(() => {
      script.remove();
      reject(new Error('Plivo browser SDK timed out while loading'));
    }, PLIVO_SDK_LOAD_TIMEOUT_MS);

    script.onload = () => {
      window.clearTimeout(timeoutId);
      const Plivo = resolvePlivoConstructor();
      if (Plivo) {
        resolve(Plivo);
        return;
      }
      reject(new Error('Plivo browser SDK loaded but did not expose window.Plivo'));
    };

    script.onerror = () => {
      window.clearTimeout(timeoutId);
      script.remove();
      reject(new Error('Unable to load Plivo browser SDK'));
    };

    document.body.appendChild(script);
  }).catch(error => {
    plivoSdkPromise = null;
    throw error;
  });

  return plivoSdkPromise;
};

const normalizePlivoDialNumber = (value) => {
  const digits = String(value || '').replace(/\D/g, '');
  if (!digits) return '';
  if (digits.length === 12 && digits.startsWith('91')) return digits.slice(-10);
  return digits;
};

const extractVoipDetailText = (value, depth = 0) => {
  if (!value || depth > 3) return '';
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  if (Array.isArray(value)) {
    return value.map(item => extractVoipDetailText(item, depth + 1)).filter(Boolean).join(' ');
  }
  if (typeof value === 'object') {
    return [
      value.error,
      value.message,
      value.reason,
      value.cause,
      value.hangupCause,
      value.hangup_cause,
      value.code,
      value.status,
    ]
      .map(item => extractVoipDetailText(item, depth + 1))
      .filter(Boolean)
      .join(' ');
  }
  return '';
};

const formatVoipDetailText = (value) => (
  String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
);

const buildVoipCallEvent = (type, detail = {}, fallbackNumber = '') => {
  const reasonText = formatVoipDetailText(extractVoipDetailText(detail));
  return {
    at: Date.now(),
    type,
    origin: detail?.origin || '',
    number: detail?.to || detail?.from || fallbackNumber || '',
    reasonText,
    raw: detail,
  };
};

const stopAudioElement = (audio) => {
  if (!audio) return;
  try {
    audio.pause?.();
    audio.currentTime = 0;
    const stream = audio.srcObject;
    if (stream && typeof stream.getTracks === 'function') {
      stream.getTracks().forEach(track => track.stop());
    }
    audio.srcObject = null;
    audio.removeAttribute?.('src');
    audio.load?.();
  } catch (_) {
    // Best-effort browser audio cleanup.
  }
};

const stopPlivoManagedAudio = () => {
  if (typeof document === 'undefined') return;
  document.querySelectorAll('audio').forEach(audio => {
    const idClass = `${audio.id || ''} ${audio.className || ''}`.toLowerCase();
    const src = String(audio.src || '').toLowerCase();
    const isPlivoOrRingtone =
      idClass.includes('plivo') ||
      idClass.includes('ringtone') ||
      /\bring\b/.test(idClass) ||
      src.includes('plivo') ||
      src.includes('ringtone');
    if (isPlivoOrRingtone) {
      stopAudioElement(audio);
    }
  });
};

export function useVoIP() {
  return useContext(VoIPContext);
}

export function VoIPProvider({ children }) {
  const [activeCall, setActiveCall] = useState(null);
  const [voipStatus, setVoipStatus] = useState('disconnected');
  const [voipError, setVoipError] = useState('');
  const [voipErrorCode, setVoipErrorCode] = useState('');
  const [voipActionLabel, setVoipActionLabel] = useState('');
  const [voipActionUrl, setVoipActionUrl] = useState('');
  const [voipMeta, setVoipMeta] = useState(null);
  const [voipConnectionEvent, setVoipConnectionEvent] = useState(null);
  const [voipCallEvent, setVoipCallEvent] = useState(null);
  const [agentEmail, setAgentEmail] = useState('');
  const [endpointUsername, setEndpointUsername] = useState('');
  // Set when the backend could not provision this recruiter their own SIP
  // endpoint and handed back the shared one. Deliberately NOT folded into
  // voipError: the softphone works, so the error path would wrongly show it as
  // broken and trigger reconnect retries. This is a standing warning about
  // silent misattribution, not a failure. See docs/call-attribution-plan.md.
  const [voipDegraded, setVoipDegraded] = useState(null);
  // Inbound call ringing this browser. Held here (not in Calls.jsx) so the
  // banner follows the recruiter across every page.
  const [incomingCall, setIncomingCall] = useState(null);
  // Read inside SDK callbacks, which close over the state value from the render
  // that registered them.
  const incomingCallRef = useRef(null);
  const softphoneRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const localAudioRef = useRef(null);
  const initDoneRef = useRef(false);
  const initInFlightRef = useRef(false);
  const activeCallRef = useRef(null);
  const dialStartedAtRef = useRef(null);
  const voipStatusRef = useRef('disconnected');
  const voipConnectionEventRef = useRef(null);
  const endpointUsernameRef = useRef('');
  const softphoneGenerationRef = useRef(0);
  const queuedForceInitRef = useRef(false);
  // Kept so a dropped WebSocket can re-register without refetching /credentials.
  const credentialsRef = useRef(null);
  const lastReloginAtRef = useRef(0);

  // Locally generated ringback: the real Plivo ringback only starts once the
  // remote leg is ringing (~5s after the click: backend initiate + SIP setup),
  // which feels like a dead line. Play a soft synthetic ring immediately and
  // stop it as soon as the SDK reports remote ringing or any terminal event.
  const dialToneRef = useRef(null);

  const stopDialTone = () => {
    const tone = dialToneRef.current;
    if (!tone) return;
    dialToneRef.current = null;
    try {
      window.clearInterval(tone.intervalId);
      tone.osc.stop();
      tone.ctx.close();
    } catch (_) {
      // Best-effort audio teardown.
    }
  };

  const startDialTone = () => {
    if (dialToneRef.current) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      if (ctx.state === 'suspended') {
        ctx.resume().catch(() => {});
      }
      const gain = ctx.createGain();
      gain.gain.value = 0;
      gain.connect(ctx.destination);
      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = 440;
      osc.connect(gain);
      osc.start();
      // Not a ringback — a faint "connecting" blip. A synthetic ring here was
      // misleading (the candidate's phone is not ringing yet) and grating next
      // to the real carrier tone that follows. One soft 150ms pulse every 2s
      // just signals the line is alive, with ramps to avoid clicks.
      const cadence = () => {
        const t = ctx.currentTime;
        gain.gain.cancelScheduledValues(t);
        gain.gain.setValueAtTime(0, t);
        gain.gain.linearRampToValueAtTime(0.035, t + 0.02);
        gain.gain.setValueAtTime(0.035, t + 0.13);
        gain.gain.linearRampToValueAtTime(0, t + 0.15);
      };
      cadence();
      const intervalId = window.setInterval(cadence, 2000);
      dialToneRef.current = { ctx, osc, gain, intervalId };
    } catch (_) {
      // The tone is a nicety — never let it break the call path.
    }
  };

  const hangupSoftphoneCall = () => {
    const softphone = softphoneRef.current;
    if (!softphone) return;
    [
      () => softphone.hangup?.(),
      () => softphone.reject?.(),
      () => softphone.client?.hangup?.(),
      () => softphone.client?.reject?.(),
    ].forEach(attempt => {
      try {
        attempt();
      } catch (_) {
        // Different Plivo SDK builds expose different call-ending methods.
      }
    });
  };

  const stopCallAudio = () => {
    stopDialTone();
    stopAudioElement(remoteAudioRef.current);
    stopAudioElement(localAudioRef.current);
    stopPlivoManagedAudio();
  };

  useEffect(() => {
    const remoteAudio = document.createElement('audio');
    remoteAudio.autoplay = true;
    remoteAudio.style.display = 'none';
    remoteAudio.id = 'plivo-remote-audio';
    document.body.appendChild(remoteAudio);
    remoteAudioRef.current = remoteAudio;

    const localAudio = document.createElement('audio');
    localAudio.autoplay = true;
    localAudio.muted = true;
    localAudio.style.display = 'none';
    localAudio.id = 'plivo-local-audio';
    document.body.appendChild(localAudio);
    localAudioRef.current = localAudio;

    return () => {
      softphoneGenerationRef.current += 1;
      // React 18 StrictMode double-invokes mount effects in dev (mount →
      // cleanup → mount) to catch exactly this class of bug: bumping the
      // generation here orphans the first initSoftphone() call mid-flight
      // (it's still awaiting the credentials fetch), but initInFlightRef
      // stays true, so the second mount's initSoftphone() call used to be
      // silently swallowed as a no-op — nothing ever retried until the
      // calling modal's own 12-18s rescue timer forced it. Reset the flags
      // here so the next mount starts a genuinely fresh, unblocked init
      // instead of stalling for that whole window.
      initInFlightRef.current = false;
      queuedForceInitRef.current = false;
      hangupSoftphoneCall();
      stopCallAudio();
      try {
        // logout() lives on .client in the Plivo v2 SDK — calling it on the
        // instance is a silent no-op that leaves a zombie SIP registration.
        softphoneRef.current?.client?.logout?.();
        softphoneRef.current?.logout?.();
      } catch (_) {
        // Ignore shutdown errors when the provider unmounts.
      }
      remoteAudio.remove();
      localAudio.remove();
    };
  }, []);

  useEffect(() => {
    if (initDoneRef.current) return undefined;
    initDoneRef.current = true;
    initSoftphone();
    // StrictMode-safe: the paired cleanup of the audio-setup effect bumps the
    // softphone generation, silently aborting this in-flight init. Reset the
    // guard so the second effect invocation re-initializes instead of leaving
    // the softphone stuck in 'connecting' until the modal's 12s rescue retry.
    return () => { initDoneRef.current = false; };
  }, []);

  useEffect(() => {
    activeCallRef.current = activeCall;
  }, [activeCall]);

  useEffect(() => {
    incomingCallRef.current = incomingCall;
  }, [incomingCall]);

  // Tell the backend whether this recruiter is on a call, so an inbound
  // "ring everyone" fork skips them rather than ringing over a live
  // conversation. Driven off activeCall so every path is covered — outbound
  // dial, connect, hangup, and accepting an inbound call — instead of having to
  // remember the beacon in each handler. Best effort: the server ages the flag
  // out, so a dropped 'idle' post cannot strand the endpoint as busy forever.
  const busyBeaconRef = useRef(null);
  useEffect(() => {
    const busy = Boolean(activeCall);
    if (busyBeaconRef.current === busy) return;
    busyBeaconRef.current = busy;
    axios.post(`${API_BASE}/plivo/busy`, { busy }).catch(() => { /* best effort */ });
  }, [activeCall]);

  useEffect(() => {
    voipStatusRef.current = voipStatus;
  }, [voipStatus]);

  useEffect(() => {
    voipConnectionEventRef.current = voipConnectionEvent;
  }, [voipConnectionEvent]);

  useEffect(() => {
    endpointUsernameRef.current = endpointUsername;
  }, [endpointUsername]);

  const clearVoipErrorState = () => {
    setVoipError('');
    setVoipErrorCode('');
    setVoipActionLabel('');
    setVoipActionUrl('');
    setVoipMeta(null);
  };

  const applyVoipErrorState = (detail, fallbackMessage) => {
    const parsed = (detail && typeof detail === 'object' && !Array.isArray(detail))
      ? {
          message: detail.message || detail.error || fallbackMessage,
          code: detail.code || '',
          actionLabel: detail.action_label || detail.actionLabel || '',
          actionUrl: detail.action_url || detail.actionUrl || '',
          meta: detail.metadata || detail.meta || null,
        }
      : {
          message: (typeof detail === 'string' && detail.trim()) ? detail : fallbackMessage,
          code: '',
          actionLabel: '',
          actionUrl: '',
          meta: null,
        };

    setVoipError(parsed.message);
    setVoipErrorCode(parsed.code);
    setVoipActionLabel(parsed.actionLabel);
    setVoipActionUrl(parsed.actionUrl);
    setVoipMeta(parsed.meta);
    setVoipStatus('error');
    return parsed;
  };

  const updateActiveCallState = (nextState, extra = {}) => {
    setActiveCall(prev => {
      if (!prev) return prev;
      return { ...prev, ...extra, state: nextState };
    });
  };

  const hasLiveVoipActivity = () => (
    Boolean(activeCallRef.current) || ['answer_required', 'invite_received', 'connected'].includes(voipStatusRef.current)
  );

  // Re-register the existing Plivo client after a silent WebSocket drop or SIP
  // logout. Without this the UI keeps reporting 'registered' while dials go
  // nowhere — "Connecting..." with no dialer tone until a full page refresh.
  const requestRelogin = (reason, { bypassThrottle = false } = {}) => {
    const client = softphoneRef.current?.client;
    const creds = credentialsRef.current;
    if (!client || !creds || hasLiveVoipActivity()) return false;
    const now = Date.now();
    if (!bypassThrottle && now - lastReloginAtRef.current < RELOGIN_THROTTLE_MS) return false;
    lastReloginAtRef.current = now;
    console.warn(`[VoIP] ${reason} — re-registering Plivo softphone`);
    reportTiming('softphone_relogin', 0, reason);
    setVoipStatus(current => (
      ['answer_required', 'invite_received', 'connected'].includes(current) ? current : 'connecting'
    ));
    try {
      client.login(creds.username, creds.password);
      return true;
    } catch (_) {
      return false;
    }
  };

  const initSoftphone = async ({ force = false } = {}) => {
    if (initInFlightRef.current) {
      if (force) queuedForceInitRef.current = true;
      return { success: false, pending: true };
    }
    initInFlightRef.current = true;
    const instanceId = softphoneGenerationRef.current + 1;
    softphoneGenerationRef.current = instanceId;

    try {
      clearVoipErrorState();
      setVoipStatus('connecting');
      setVoipCallEvent(null);
      if (force && softphoneRef.current) {
        try {
          // Deregister the old client for real (.client.logout, not .logout —
          // the latter doesn't exist and left a zombie registration that
          // fought the new client and stole call audio/events).
          softphoneRef.current.client?.logout?.();
          softphoneRef.current.logout?.();
        } catch (_) {
          // Ignore stale SDK logout failures before rebuilding the client.
        }
        softphoneRef.current = null;
      }

      const initStart = performance.now();
      const credentialsStart = performance.now();
      // Must go through axios, not bare fetch: /plivo/credentials is now
      // authenticated and per-user, and the bearer token lives on
      // axios.defaults.headers.common. A raw fetch() sends no Authorization
      // header, which 401s — leaving the softphone unregistered, so outbound
      // dialling fails AND inbound has no endpoint to ring.
      let res = {};
      let credentialsOk = false;
      let lastStatus = 0;
      // The bearer token is restored asynchronously on a cold page load, so this
      // can fire before axios has an Authorization header and 401. Retry a few
      // times instead of giving up: the softphone used to be rescued only by a
      // retry living in Calls.jsx, which is why incoming calls appeared on that
      // page and nowhere else.
      for (let attempt = 0; attempt < 4 && !credentialsOk; attempt += 1) {
        try {
          const credentialsResponse = await axios.get(`${API_BASE}/plivo/credentials`);
          res = credentialsResponse.data || {};
          credentialsOk = true;
        } catch (error) {
          lastStatus = error?.response?.status || 0;
          res = error?.response?.data || {};
          if (lastStatus !== 401 && lastStatus !== 403) break;
          await new Promise(resolve => window.setTimeout(resolve, 400 * (attempt + 1)));
        }
      }
      reportTiming('credentials_fetch', performance.now() - credentialsStart);
      if (!credentialsOk) {
        const detail = res?.detail;
        const message = (detail && typeof detail === 'object' ? detail.message : '')
          || (lastStatus === 401 || lastStatus === 403
            ? 'Not signed in yet — softphone will connect once your session is ready'
            : 'Unable to prepare Plivo softphone');
        setVoipStatus('error');
        setVoipError(message);
        return;
      }
      if (!res.username || !res.password) {
        setVoipStatus('error');
        setVoipError('Plivo softphone credentials are missing');
        return;
      }
      setAgentEmail(res.username);
      setEndpointUsername(res.username);
      setVoipDegraded(res.degraded
        ? {
            reason: res.degraded_reason
              || 'Your softphone is running on a shared line. Call recordings may be attached to the wrong candidate.',
            at: Date.now(),
          }
        : null);
      if (res.degraded) {
        console.error('[VoIP] DEGRADED — shared Plivo endpoint in use:', res.degraded_reason);
      }
      credentialsRef.current = { username: res.username, password: res.password };

      const sdkLoadStart = performance.now();
      const Plivo = await loadPlivoSdk();
      reportTiming('sdk_load', performance.now() - sdkLoadStart);
      if (softphoneGenerationRef.current !== instanceId) return;

      const options = {
        debug: 'ALL',
        permOnClick: true,
        audioConstraints: { optional: [{ googAutoGainControl: false }] },
        enableDscp: true,
      };
      const sdk = new Plivo(options);
      softphoneRef.current = sdk;
      try {
        // Disable the SDK's own connect/ringback tones — they sound different
        // from our local ring and produced a jarring two-tone dial experience.
        // One synthetic Indian ringback (startDialTone) plays from click until
        // answer instead.
        sdk.client.setConnectTone?.(false);
        sdk.client.setRingToneBack?.(false);
      } catch (_) {
        // Tone helpers are best-effort; older SDK builds may not expose them.
      }

      const loginPromise = new Promise((resolve, reject) => {
        const timeoutId = window.setTimeout(() => {
          reject(new Error('Plivo softphone registration timed out'));
        }, PLIVO_LOGIN_TIMEOUT_MS);

        sdk.client.on('onLogin', () => {
          window.clearTimeout(timeoutId);
          clearVoipErrorState();
          setVoipConnectionEvent({ at: Date.now(), state: 'registered', maxRetriesReached: false, error: '' });
          setVoipStatus('registered');
          console.log('[VoIP] Connected to Plivo softphone');
          // Tell the backend this endpoint is live so inbound calls ring it.
          axios.post(`${API_BASE}/plivo/registered`).catch(() => { /* best effort */ });
          resolve();
        });

        sdk.client.on('onLoginFailed', (reason) => {
          window.clearTimeout(timeoutId);
          const message = formatVoipDetailText(extractVoipDetailText(reason)) || 'Plivo registration failed';
          setVoipConnectionEvent({ at: Date.now(), state: 'login_failed', maxRetriesReached: false, error: message });
          setVoipStatus('error');
          setVoipError(message);
          reject(new Error(message));
        });
      });

      // Detect silent registration loss (laptop sleep, network change, Plivo
      // dropping the socket). Without these, voipStatus stays 'registered'
      // forever and the next dial hangs on "Connecting..." with no tone.
      sdk.client.on('onLogout', () => {
        if (softphoneGenerationRef.current !== instanceId) return;
        requestRelogin('SIP session logged out');
      });

      sdk.client.on('onConnectionChange', (info) => {
        if (softphoneGenerationRef.current !== instanceId) return;
        const state = typeof info === 'string' ? info : info?.state;
        if (state === 'disconnected') {
          // Socket died mid-dial: the server-side leg may still ring the
          // candidate, but this client can no longer hear or control it.
          // Fail the attempt immediately instead of showing "Connecting..."
          // forever, then re-register so the retry works.
          const pendingDial = activeCallRef.current && activeCallRef.current.state !== 'connected';
          if (pendingDial) {
            hangupSoftphoneCall();
            stopCallAudio();
            setVoipCallEvent(buildVoipCallEvent(
              'failed',
              { origin: 'local', reason: 'connection lost while dialing' },
              activeCallRef.current?.number,
            ));
            setActiveCall(null);
            // Sync the ref immediately — it normally updates via effect after
            // render, and requestRelogin's live-activity guard reads it now.
            activeCallRef.current = null;
          }
          requestRelogin('Plivo WebSocket disconnected');
        } else if (state === 'connected' && sdk.client.isLoggedIn) {
          clearVoipErrorState();
          setVoipStatus(current => (
            ['answer_required', 'invite_received', 'connected'].includes(current) ? current : 'registered'
          ));
        }
      });

      sdk.client.on('onCallRemoteRinging', () => {
        if (softphoneGenerationRef.current !== instanceId) return;
        // Remote ringing means the carrier's real in-band audio (ringback,
        // caller tune, busy/switched-off announcements) is now streaming in
        // as early media. Silence the synthetic ring so the two never overlap
        // — the SDK's own tones stay disabled, so from here the caller hears
        // only the genuine network audio.
        stopDialTone();
        if (dialStartedAtRef.current) {
          reportTiming('dial_to_remote_ringing', performance.now() - dialStartedAtRef.current);
        }
      });

      sdk.client.on('onCallAnswered', (callInfo) => {
        if (softphoneGenerationRef.current !== instanceId) return;
        stopDialTone();
        if (dialStartedAtRef.current) {
          reportTiming('dial_to_answered', performance.now() - dialStartedAtRef.current);
          dialStartedAtRef.current = null;
        }
        clearVoipErrorState();
        setVoipStatus('connected');
        setVoipCallEvent(buildVoipCallEvent('connected', callInfo, activeCallRef.current?.number));
        setActiveCall({ state: 'connected', number: callInfo?.to || activeCallRef.current?.number || '' });
      });

      // A candidate calling back rings every registered recruiter at once. It
      // must never make a sound or steal focus — the recruiter is mid-task, so
      // this only surfaces a dismissible banner they can choose to answer.
      sdk.client.on('onIncomingCall', (callUUID, extraHeaders, callInfo) => {
        if (softphoneGenerationRef.current !== instanceId) return;
        try {
          sdk.client.setRingTone?.(false);
        } catch (_) { /* older SDK builds may not expose it */ }
        stopPlivoManagedAudio();
        // The SDK's argument shape varies by version and call type, so pull the
        // caller id out of whichever slot carries it and treat it as optional —
        // the banner resolves the name from the backend, which has already
        // matched the number to a candidate.
        const uuid = typeof callUUID === 'string' ? callUUID : callUUID?.callUUID;
        const from =
          callInfo?.from
          || callInfo?.callerId
          || callInfo?.callerName
          || extraHeaders?.from
          || extraHeaders?.['X-Ph-From']
          || (typeof callUUID === 'object' ? callUUID?.from : '')
          || '';
        console.log('[VoIP] Incoming call', { uuid, from, callInfo, extraHeaders });

        // Already on a call: do not surface a banner over a live conversation.
        // The backend now excludes busy endpoints from the fork, but this can
        // still fire — the busy flag is set a moment after the call starts, and
        // a shared fallback endpoint marks every recruiter on it at once. Drop
        // it here; the call still rings other recruiters and still lands in
        // Inbound Callbacks, so nothing is lost.
        if (activeCallRef.current) {
          console.log('[VoIP] Suppressing incoming banner — already on a call');
          try {
            sdk.client.ignore?.(uuid);
          } catch (_) { /* best effort — the <Dial timeout> covers it */ }
          return;
        }

        // Single slot: a second caller arriving inside the ring window would
        // silently replace the first banner. Keep the one already showing —
        // the newcomer still rings every other recruiter, and both rows appear
        // in Inbound Callbacks regardless.
        if (incomingCallRef.current?.callUUID && incomingCallRef.current.callUUID !== uuid) {
          console.log('[VoIP] Suppressing incoming banner — one already pending');
          return;
        }

        setIncomingCall({ callUUID: uuid, from, at: Date.now() });
      });

      sdk.client.on('onIncomingCallCanceled', (canceledUUID) => {
        if (softphoneGenerationRef.current !== instanceId) return;
        // Another recruiter answered first, or the caller hung up.
        stopPlivoManagedAudio();
        // Only clear the banner if this cancel is for the call it is showing —
        // a concurrent caller we suppressed above also cancels, and must not
        // take down a banner the recruiter can still answer. When the SDK gives
        // us no UUID, fall back to the old unconditional clear.
        const uuid = typeof canceledUUID === 'string' ? canceledUUID : canceledUUID?.callUUID;
        const pending = incomingCallRef.current?.callUUID;
        if (uuid && pending && uuid !== pending) return;
        setIncomingCall(null);
      });

      sdk.client.on('onCallTerminated', (reason) => {
        if (softphoneGenerationRef.current !== instanceId) return;
        stopDialTone();
        setIncomingCall(null);
        setVoipCallEvent(buildVoipCallEvent('terminated', reason, activeCallRef.current?.number));
        setVoipStatus('registered');
        setActiveCall(null);
      });

      sdk.client.on('onCallFailed', (reason) => {
        if (softphoneGenerationRef.current !== instanceId) return;
        stopDialTone();
        console.warn('[VoIP] Call failed', reason);
        setVoipCallEvent(buildVoipCallEvent('failed', reason, activeCallRef.current?.number));
        setVoipStatus('registered');
        setActiveCall(null);
      });

      const loginStart = performance.now();
      sdk.client.login(res.username, res.password);
      await loginPromise;
      reportTiming('sip_login', performance.now() - loginStart);
      reportTiming('softphone_init_total', performance.now() - initStart, `force=${force}`);
      return { success: true };
    } catch (error) {
      if (softphoneGenerationRef.current !== instanceId) return;
      setVoipStatus('error');
      const message = error?.message || 'Failed to initialize Plivo VoIP Softphone';
      setVoipError(message);
      return { success: false, error: message };
    } finally {
      initInFlightRef.current = false;
      if (queuedForceInitRef.current && softphoneGenerationRef.current === instanceId) {
        queuedForceInitRef.current = false;
        window.setTimeout(() => {
          initSoftphone({ force: true });
        }, 0);
      }
    }
  };

  const placeCall = async (toNumber, dialToken = '') => {
    const client = softphoneRef.current?.client;
    if (!client || typeof client.call !== 'function') {
      return { success: false, error: 'Softphone client not available' };
    }
    const dialNumber = normalizePlivoDialNumber(toNumber);
    if (!dialNumber) {
      return { success: false, error: 'Candidate phone number is missing' };
    }
    // Stale-registration guard: the UI can believe it is registered long after
    // the SIP session died. Dialing on a dead session never rings, so confirm
    // the SDK is actually logged in and re-register first if it is not.
    if (client.isLoggedIn === false) {
      reportTiming('dial_preflight_relogin', 0, 'client not logged in at dial time');
      requestRelogin('softphone was not registered at dial time', { bypassThrottle: true });
      const deadline = Date.now() + RELOGIN_WAIT_AT_DIAL_MS;
      while (Date.now() < deadline && !client.isLoggedIn) {
        await new Promise(resolve => window.setTimeout(resolve, 300));
      }
      if (!client.isLoggedIn) {
        return { success: false, error: 'Plivo softphone lost its connection. Please try the call again.' };
      }
      clearVoipErrorState();
      setVoipStatus('registered');
    }
    try {
      setVoipCallEvent({ at: Date.now(), type: 'dialing', origin: 'local', number: dialNumber, reasonText: '', raw: null });
      setActiveCall({ state: 'dialing', number: dialNumber });
      dialStartedAtRef.current = performance.now();
      // The token identifies this specific dial attempt, so the backend can
      // attribute the call to the exact `calls` row instead of guessing the
      // most recently updated row for this SIP username. Hex only: Plivo
      // requires X-PH-* header values to be alphanumeric.
      //
      // Whether Plivo forwards X-PH-* to the answer URL is documented but
      // unverified against this account's SDK build. Nothing breaks if it does
      // not arrive — the backend logs [DialAttribution] and falls back to the
      // old username matching. Set SEND_DIAL_TOKEN to false to revert entirely.
      if (SEND_DIAL_TOKEN && dialToken) {
        console.log('[VoIP] Dialing with token', dialToken);
        softphoneRef.current.client.call(dialNumber, { 'X-PH-DialToken': dialToken });
      } else {
        softphoneRef.current.client.call(dialNumber);
      }
      setVoipStatus('connecting');
      return { success: true, dialNumber, username: endpointUsernameRef.current };
    } catch (error) {
      setActiveCall(null);
      return { success: false, error: error?.message || 'Mic access denied or Plivo failure' };
    }
  };

  const ensureMicrophonePermission = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      return { success: true };
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      return { success: true };
    } catch (error) {
      return { success: false, error: error?.message || 'Microphone access denied' };
    }
  };

  const waitForPlivoDial = async (
    username = endpointUsernameRef.current,
    timeoutMs = PLIVO_DIAL_HANDSHAKE_TIMEOUT_MS,
    dialToken = '',
    dialedNumber = '',
  ) => {
    const cleanUsername = String(username || '').trim();
    const cleanToken = String(dialToken || '').trim();
    if (!cleanUsername && !cleanToken) {
      return { success: false, error: 'Plivo endpoint username is missing' };
    }

    const tokenUrl = cleanToken
      ? `${API_BASE}/plivo/call-state-by-token/${encodeURIComponent(cleanToken)}`
      : '';
    const usernameUrl = cleanUsername
      ? `${API_BASE}/plivo/call-state/${encodeURIComponent(cleanUsername)}`
      : '';
    const dialedTail = String(dialedNumber || '').replace(/\D/g, '').slice(-10);

    const readState = async (url) => {
      const response = await fetch(url);
      const state = await response.json().catch(() => ({}));
      return response.ok ? state : null;
    };

    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        // Token first — an exact match on this attempt.
        if (tokenUrl) {
          const state = await readState(tokenUrl);
          if (state?.call_uuid) return { success: true, state };
        }

        // Fallback for when Plivo does not forward the X-PH header (documented
        // but unverified on this account) — without it a working call would
        // fail the handshake and look broken to the recruiter.
        //
        // The username state is never cleared, so it happily returns the
        // *previous* call's UUID. Require the number to match the one we are
        // dialing, which rejects a stale attempt to a different candidate. A
        // stale entry for the same candidate can still pass, but that is a
        // redial of the same person and harmless for a liveness gate.
        if (usernameUrl) {
          const state = await readState(usernameUrl);
          const stateTail = String(state?.to_number || '').replace(/\D/g, '').slice(-10);
          if (state?.call_uuid && (!dialedTail || stateTail === dialedTail)) {
            return { success: true, state };
          }
        }
      } catch (_) {
        // Keep polling until the deadline so transient dev-server hiccups do not fail the call immediately.
      }
      await new Promise(resolve => window.setTimeout(resolve, PLIVO_DIAL_HANDSHAKE_POLL_MS));
    }

    return {
      success: false,
      error: 'Plivo browser call did not reach the backend',
      code: 'plivo_dial_webhook_timeout',
    };
  };

  const answerCall = async () => {
     return { success: true };
  };

  const acceptIncomingCall = async () => {
    const pending = incomingCall;
    if (!pending?.callUUID) return { success: false };
    try {
      await ensureMicrophonePermission();
      // 'ignore' so a second inbound never rings over a live conversation.
      softphoneRef.current?.client?.answer(pending.callUUID, 'ignore');
      setIncomingCall(null);
      setVoipStatus('connected');
      setActiveCall({ state: 'connected', number: pending.from || '', direction: 'inbound' });
      return { success: true };
    } catch (error) {
      console.error('[VoIP] Failed to answer inbound call', error);
      setIncomingCall(null);
      return { success: false, error: error?.message };
    }
  };

  const dismissIncomingCall = () => {
    const pending = incomingCall;
    setIncomingCall(null);
    if (!pending?.callUUID) return;
    try {
      // Stop it ringing here only. Other recruiters keep ringing, and the
      // callback still lands in Inbound Callbacks either way.
      softphoneRef.current?.client?.ignore?.(pending.callUUID);
    } catch (error) {
      console.warn('[VoIP] Could not ignore inbound call', error);
    }
  };

  const rejectCall = async () => {
    hangupSoftphoneCall();
    stopCallAudio();
    setVoipCallEvent({
      at: Date.now(),
      type: 'terminated',
      origin: 'local',
      number: activeCallRef.current?.number || '',
      reasonText: 'ended locally',
      raw: null,
    });
    setVoipStatus('registered');
    setActiveCall(null);
    return { success: true };
  };

  return (
    <VoIPContext.Provider
      value={{
        activeCall,
        answerCall,
        rejectCall,
        placeCall,
        ensureMicrophonePermission,
        waitForPlivoDial,
        startDialTone,
        stopDialTone,
        voipStatus,
        voipError,
        voipErrorCode,
        voipActionLabel,
        voipActionUrl,
        voipMeta,
        voipConnectionEvent,
        voipCallEvent,
        agentEmail,
        endpointUsername,
        incomingCall,
        acceptIncomingCall,
        dismissIncomingCall,
        voipDegraded,
        retryVoip: () => initSoftphone({ force: true }),
      }}
    >
      {children}
    </VoIPContext.Provider>
  );
}
