import React, { createContext, useContext, useEffect, useRef, useState } from 'react';

import { API_BASE } from '../store/useAppStore';

const VoIPContext = createContext({
  activeCall: null,
  answerCall: async () => ({ success: false }),
  rejectCall: async () => ({ success: false }),
  placeCall: async () => ({ success: false }),
  ensureMicrophonePermission: async () => ({ success: true }),
  waitForPlivoDial: async () => ({ success: false }),
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
  retryVoip: () => {},
});

const SOFTPHONE_RECOVERY_WINDOW_MS = 5000;
const PLIVO_SDK_URL = '/plivo.min.js';
const PLIVO_SDK_LOAD_TIMEOUT_MS = 15000;
const PLIVO_LOGIN_TIMEOUT_MS = 20000;
const PLIVO_DIAL_HANDSHAKE_TIMEOUT_MS = 12000;
const PLIVO_DIAL_HANDSHAKE_POLL_MS = 750;

let plivoSdkPromise = null;

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
  const softphoneRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const localAudioRef = useRef(null);
  const initDoneRef = useRef(false);
  const initInFlightRef = useRef(false);
  const activeCallRef = useRef(null);
  const voipStatusRef = useRef('disconnected');
  const voipConnectionEventRef = useRef(null);
  const endpointUsernameRef = useRef('');
  const softphoneGenerationRef = useRef(0);
  const recoveryInFlightRef = useRef(false);
  const recoveryTimerRef = useRef(null);
  const queuedForceInitRef = useRef(false);

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
      if (recoveryTimerRef.current) {
        window.clearTimeout(recoveryTimerRef.current);
        recoveryTimerRef.current = null;
      }
      hangupSoftphoneCall();
      stopCallAudio();
      try {
        softphoneRef.current?.logout?.();
      } catch (_) {
        // Ignore shutdown errors when the provider unmounts.
      }
      remoteAudio.remove();
      localAudio.remove();
    };
  }, []);

  useEffect(() => {
    if (initDoneRef.current) return;
    initDoneRef.current = true;
    initSoftphone();
  }, []);

  useEffect(() => {
    activeCallRef.current = activeCall;
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

  const clearRecoveryTimer = () => {
    if (recoveryTimerRef.current) {
      window.clearTimeout(recoveryTimerRef.current);
      recoveryTimerRef.current = null;
    }
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

  const recoverSoftphone = async (instanceId, { reconnect = false } = {}) => {
    if (recoveryInFlightRef.current || softphoneGenerationRef.current !== instanceId || !softphoneRef.current) {
      return;
    }

    recoveryInFlightRef.current = true;
    clearRecoveryTimer();
    clearVoipErrorState();
    setVoipStatus(current => (
      ['answer_required', 'invite_received', 'connected'].includes(current) ? current : 'connecting'
    ));
    recoveryTimerRef.current = window.setTimeout(() => {
      if (softphoneGenerationRef.current !== instanceId) return;
      recoveryInFlightRef.current = false;
      recoveryTimerRef.current = null;

      if (hasLiveVoipActivity() || voipStatusRef.current === 'registered') {
        return;
      }

      const latestEvent = voipConnectionEventRef.current;
      const message = latestEvent?.error || 'Plivo softphone registration failed';
      applyVoipErrorState(
        { message, code: 'softphone_registration_failed' },
        'Plivo softphone registration failed',
      );
    }, SOFTPHONE_RECOVERY_WINDOW_MS);

    try {
      await softphoneRef.current.reset(reconnect);
    } catch (error) {
      clearRecoveryTimer();
      recoveryInFlightRef.current = false;
      if (softphoneGenerationRef.current !== instanceId || hasLiveVoipActivity()) {
        return;
      }

      const message = error?.message || 'Plivo softphone registration failed';
      applyVoipErrorState(
        { message, code: 'softphone_registration_failed' },
        'Plivo softphone registration failed',
      );
    } finally {
      if (softphoneGenerationRef.current === instanceId) {
        recoveryInFlightRef.current = Boolean(recoveryTimerRef.current);
      }
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
    clearRecoveryTimer();
    recoveryInFlightRef.current = false;

    try {
      clearVoipErrorState();
      setVoipStatus('connecting');
      setVoipCallEvent(null);
      if (force && softphoneRef.current) {
        try {
          softphoneRef.current.logout?.();
        } catch (_) {
          // Ignore stale SDK logout failures before rebuilding the client.
        }
        softphoneRef.current = null;
      }

      const credentialsResponse = await fetch(`${API_BASE}/plivo/credentials`).catch(() => null);
      const res = credentialsResponse ? await credentialsResponse.json().catch(() => ({})) : {};
      if (!credentialsResponse?.ok) {
        const detail = res?.detail;
        const message = (detail && typeof detail === 'object' ? detail.message : '') || 'Unable to prepare Plivo softphone';
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

      const Plivo = await loadPlivoSdk();
      if (softphoneGenerationRef.current !== instanceId) return;

      const options = {
        debug: 'ALL',
        permOnClick: true,
        audioConstraints: { optional: [{ googAutoGainControl: false }] },
        enableDscp: true,
      };
      const sdk = new Plivo(options);
      softphoneRef.current = sdk;

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

      sdk.client.on('onCallAnswered', (callInfo) => {
        clearVoipErrorState();
        setVoipStatus('connected');
        setVoipCallEvent(buildVoipCallEvent('connected', callInfo, activeCallRef.current?.number));
        setActiveCall({ state: 'connected', number: callInfo?.to || activeCallRef.current?.number || '' });
      });

      sdk.client.on('onCallTerminated', (reason) => {
        setVoipCallEvent(buildVoipCallEvent('terminated', reason, activeCallRef.current?.number));
        setVoipStatus('registered');
        setActiveCall(null);
      });

      sdk.client.on('onCallFailed', (reason) => {
        console.warn('[VoIP] Call failed', reason);
        setVoipCallEvent(buildVoipCallEvent('failed', reason, activeCallRef.current?.number));
        setVoipStatus('registered');
        setActiveCall(null);
      });

      sdk.client.login(res.username, res.password);
      await loginPromise;
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

  const placeCall = async (toNumber) => {
    if (!softphoneRef.current?.client || typeof softphoneRef.current.client.call !== 'function') {
      return { success: false, error: 'Softphone client not available' };
    }
    const dialNumber = normalizePlivoDialNumber(toNumber);
    if (!dialNumber) {
      return { success: false, error: 'Candidate phone number is missing' };
    }
    try {
      setVoipCallEvent({ at: Date.now(), type: 'dialing', origin: 'local', number: dialNumber, reasonText: '', raw: null });
      setActiveCall({ state: 'dialing', number: dialNumber });
      softphoneRef.current.client.call(dialNumber);
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

  const waitForPlivoDial = async (username = endpointUsernameRef.current, timeoutMs = PLIVO_DIAL_HANDSHAKE_TIMEOUT_MS) => {
    const cleanUsername = String(username || '').trim();
    if (!cleanUsername) {
      return { success: false, error: 'Plivo endpoint username is missing' };
    }

    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        const response = await fetch(`${API_BASE}/plivo/call-state/${encodeURIComponent(cleanUsername)}`);
        const state = await response.json().catch(() => ({}));
        if (response.ok && state?.call_uuid) {
          return { success: true, state };
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
        retryVoip: () => initSoftphone({ force: true }),
      }}
    >
      {children}
    </VoIPContext.Provider>
  );
}
