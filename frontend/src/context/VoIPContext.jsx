import React, { createContext, useContext, useEffect, useRef, useState } from 'react';

import { API_BASE } from '../store/useAppStore';

const VoIPContext = createContext({
  activeCall: null,
  answerCall: async () => ({ success: false }),
  rejectCall: async () => ({ success: false }),
  voipStatus: 'disconnected',
  voipError: '',
  voipErrorCode: '',
  voipActionLabel: '',
  voipActionUrl: '',
  voipMeta: null,
  voipConnectionEvent: null,
  agentEmail: '',
  retryVoip: () => {},
});

const SOFTPHONE_RECOVERY_WINDOW_MS = 5000;

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
  const [agentEmail, setAgentEmail] = useState('');
  const softphoneRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const localAudioRef = useRef(null);
  const initDoneRef = useRef(false);
  const initInFlightRef = useRef(false);
  const activeCallRef = useRef(null);
  const voipStatusRef = useRef('disconnected');
  const voipConnectionEventRef = useRef(null);
  const softphoneGenerationRef = useRef(0);
  const recoveryInFlightRef = useRef(false);
  const recoveryTimerRef = useRef(null);

  useEffect(() => {
    const remoteAudio = document.createElement('audio');
    remoteAudio.autoplay = true;
    remoteAudio.style.display = 'none';
    remoteAudio.id = 'frejun-remote-audio';
    document.body.appendChild(remoteAudio);
    remoteAudioRef.current = remoteAudio;

    const localAudio = document.createElement('audio');
    localAudio.autoplay = true;
    localAudio.muted = true;
    localAudio.style.display = 'none';
    localAudio.id = 'frejun-local-audio';
    document.body.appendChild(localAudio);
    localAudioRef.current = localAudio;

    return () => {
      softphoneGenerationRef.current += 1;
      if (recoveryTimerRef.current) {
        window.clearTimeout(recoveryTimerRef.current);
        recoveryTimerRef.current = null;
      }
      try {
        softphoneRef.current?.logout?.();
      } catch (_) {
        // Ignore shutdown errors when the provider unmounts.
      }
      document.body.removeChild(remoteAudio);
      document.body.removeChild(localAudio);
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

  const getFreshToken = async () => {
    try {
      const appToken = localStorage.getItem('token') || sessionStorage.getItem('token') || '';
      const res = await fetch(`${API_BASE}/voip/token`, {
        headers: appToken ? { Authorization: `Bearer ${appToken}` } : {},
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data?.detail;
        return {
          success: false,
          error: (typeof detail === 'object' ? detail.message : detail) || data.error || 'Failed to refresh FreJun VoIP token',
          code: typeof detail === 'object' ? detail.code || '' : '',
          actionLabel: typeof detail === 'object' ? detail.action_label || '' : '',
          actionUrl: typeof detail === 'object' ? detail.action_url || '' : '',
          meta: typeof detail === 'object' ? detail.metadata || null : null,
          detail,
        };
      }

      return {
        success: true,
        accessToken: data.access_token || '',
        agentEmail: data.agent_email || '',
        meta: data.metadata || null,
      };
    } catch (error) {
      return {
        success: false,
        error: error?.message || 'Failed to reach the backend VoIP token endpoint',
      };
    }
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
      const message = latestEvent?.error || 'FreJun softphone registration failed';
      applyVoipErrorState(
        { message, code: 'softphone_registration_failed' },
        'FreJun softphone registration failed',
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

      const message = error?.message || 'FreJun softphone registration failed';
      applyVoipErrorState(
        { message, code: 'softphone_registration_failed' },
        'FreJun softphone registration failed',
      );
    } finally {
      if (softphoneGenerationRef.current === instanceId) {
        recoveryInFlightRef.current = Boolean(recoveryTimerRef.current);
      }
    }
  };

  const initSoftphone = async ({ force = false } = {}) => {
    if (initInFlightRef.current) return;
    initInFlightRef.current = true;
    const instanceId = softphoneGenerationRef.current + 1;
    softphoneGenerationRef.current = instanceId;
    clearRecoveryTimer();
    recoveryInFlightRef.current = false;

    try {
      clearVoipErrorState();
      setVoipStatus('connecting');

      const res = await fetch(`${API_BASE}/plivo/credentials`).then(r => r.json()).catch(() => ({}));
      if (!res.username || !res.password) {
        setVoipStatus('error');
        setVoipError('Missing Plivo credentials');
        return;
      }
      setAgentEmail(res.username);

      if (window.Plivo) {
        const options = {
          "debug": "ALL",
          "permOnClick": true,
          "audioConstraints": { "optional": [{ "googAutoGainControl": false }] },
          "enableDscp": true
        };
        const sdk = new window.Plivo(options);
        softphoneRef.current = sdk;

        sdk.client.on('onLogin', () => {
          setVoipStatus('registered');
          console.log('[VoIP] Connected to Plivo softphone');
        });

        sdk.client.on('onLoginFailed', () => {
          setVoipStatus('error');
          setVoipError('Plivo registration failed');
        });

        sdk.client.on('onCallAnswered', (callInfo) => {
          setVoipStatus('connected');
          setActiveCall({ state: 'connected', number: callInfo?.to });
        });

        sdk.client.on('onCallTerminated', (reason) => {
          setVoipStatus('registered');
          setActiveCall(null);
        });

        sdk.client.on('onCallFailed', (reason) => {
          setVoipStatus('registered');
          setActiveCall(null);
        });

        sdk.client.login(res.username, res.password);
      } else {
        setVoipStatus('registered');
      }
    } catch (error) {
      if (softphoneGenerationRef.current !== instanceId) return;
      setVoipStatus('error');
      setVoipError('Failed to initialize Plivo VoIP Softphone');
    } finally {
      initInFlightRef.current = false;
    }
  };

  const placeCall = async (toNumber) => {
    if (!softphoneRef.current) {
      return { success: false, error: 'Softphone client not available' };
    }
    try {
      // Trigger browser microphone access
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
         await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      softphoneRef.current.client.call(toNumber);
      setVoipStatus('connecting');
      return { success: true };
    } catch (error) {
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

  const answerCall = async () => {
     return { success: true };
  };

  const rejectCall = async () => {
    if (softphoneRef.current) {
      try {
        if (typeof softphoneRef.current.hangup === 'function') {
          softphoneRef.current.hangup();
        } else if (softphoneRef.current.client && typeof softphoneRef.current.client.hangup === 'function') {
          softphoneRef.current.client.hangup();
        }
      } catch (e) {
        console.error('Plivo hangup error:', e);
      }
    }
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
        voipStatus,
        voipError,
        voipErrorCode,
        voipActionLabel,
        voipActionUrl,
        voipMeta,
        voipConnectionEvent,
        agentEmail,
        retryVoip: () => initSoftphone({ force: true }),
      }}
    >
      {children}
    </VoIPContext.Provider>
  );
}
