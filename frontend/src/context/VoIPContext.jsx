import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Softphone } from '@frejun/softphone-web-sdk';

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
  agentEmail: '',
  retryVoip: () => {},
});

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
  const [agentEmail, setAgentEmail] = useState('');
  const softphoneRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const localAudioRef = useRef(null);
  const initDoneRef = useRef(false);
  const initInFlightRef = useRef(false);

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

  const initSoftphone = async ({ force = false } = {}) => {
    if (initInFlightRef.current) return;
    initInFlightRef.current = true;

    try {
      clearVoipErrorState();
      setVoipStatus('connecting');

      if (force && softphoneRef.current?.logout) {
        try {
          await softphoneRef.current.logout();
        } catch (_) {
          // Ignore logout failures while rebuilding the browser softphone.
        }
        softphoneRef.current = null;
      }

      const tokenResult = await getFreshToken();
      if (!tokenResult.success || !tokenResult.accessToken || !tokenResult.agentEmail) {
        applyVoipErrorState(
          tokenResult.detail || {
            message: tokenResult.error || 'FreJun VoIP token is unavailable',
            code: tokenResult.code,
            action_label: tokenResult.actionLabel,
            action_url: tokenResult.actionUrl,
            metadata: tokenResult.meta,
          },
          'FreJun VoIP token is unavailable',
        );
        return;
      }

      setAgentEmail(tokenResult.agentEmail);
      setVoipMeta(tokenResult.meta || null);

      const sp = new Softphone();
      softphoneRef.current = sp;

      await sp.login({
        type: 'OAuth2.0',
        token: tokenResult.accessToken,
        email: tokenResult.agentEmail,
      });

      const listeners = {
        onConnectionStateChange: (type, state, maxRetriesReached, error) => {
          console.log(`[VoIP] ${type}: ${state}${error ? ` ERR:${error}` : ''}`);
          if (type === 'RegisterState' && state === 'Registered') {
            clearVoipErrorState();
            setVoipStatus(current => (current === 'connected' ? current : 'registered'));
            return;
          }

          if (state === 'Error' || maxRetriesReached) {
            applyVoipErrorState(
              { message: String(error || 'FreJun softphone registration failed'), code: 'softphone_registration_failed' },
              'FreJun softphone registration failed',
            );
            return;
          }

          if (state === 'Unregistered' || state === 'Terminated') {
            setVoipStatus(current => (current === 'connected' ? current : 'disconnected'));
          }
        },
        onCallCreated: (sessionType, metadata) => {
          console.log('[VoIP] CALL CREATED', sessionType, metadata);
          const session = softphoneRef.current?.getSession;
          if (!session) {
            applyVoipErrorState(
              { message: 'FreJun created a session, but the browser SDK could not attach to it.', code: 'softphone_session_attach_failed' },
              'FreJun created a session, but the browser SDK could not attach to it.',
            );
            return;
          }

          clearVoipErrorState();
          setActiveCall({
            session,
            metadata,
            sessionType,
            state: 'answer_required',
          });
          setVoipStatus('answer_required');
        },
        onCallRinging: (sessionType, metadata) => {
          console.log('[VoIP] CALL ESTABLISHED', sessionType, metadata);
          updateActiveCallState('connected', { metadata, sessionType });
          clearVoipErrorState();
          setVoipStatus('connected');
        },
        onCallHangup: (sessionType, metadata) => {
          console.log('[VoIP] CALL HANGUP', sessionType, metadata);
          setActiveCall(null);
          setVoipStatus('registered');
        },
      };

      await sp.start(listeners, {
        local: localAudioRef.current,
        remote: remoteAudioRef.current,
      });
    } catch (error) {
      const message = error?.message || 'Failed to initialize the FreJun softphone';
      console.error('[VoIP] Init failed:', message, error);
      applyVoipErrorState(
        { message, code: 'softphone_init_failed' },
        'Failed to initialize the FreJun softphone',
      );
      setActiveCall(null);
    } finally {
      initInFlightRef.current = false;
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
      const message = error?.name === 'NotAllowedError'
        ? 'Microphone permission is required to answer browser VoIP calls.'
        : (error?.message || 'Could not access the microphone for browser VoIP.');
      applyVoipErrorState({ message, code: 'microphone_permission_required' }, message);
      return { success: false, error: message };
    }
  };

  const answerCall = async () => {
    if (!activeCall?.session) {
      return { success: false, error: 'No incoming browser call is ready to answer.' };
    }

    const permission = await ensureMicrophonePermission();
    if (!permission.success) {
      return permission;
    }

    try {
      updateActiveCallState('invite_received');
      clearVoipErrorState();
      setVoipStatus('invite_received');

      const accepted = await activeCall.session.accept();
      if (!accepted) {
        const message = 'FreJun did not accept the browser call session.';
        applyVoipErrorState({ message, code: 'softphone_accept_failed' }, message);
        return { success: false, error: message };
      }

      return { success: true };
    } catch (error) {
      const message = error?.message || 'Failed to answer the browser VoIP call.';
      applyVoipErrorState({ message, code: 'softphone_answer_failed' }, message);
      return { success: false, error: message };
    }
  };

  const rejectCall = async () => {
    if (!activeCall?.session) {
      return { success: false, error: 'No active browser call to end.' };
    }

    try {
      await activeCall.session.end();
      setActiveCall(null);
      clearVoipErrorState();
      setVoipStatus('registered');
      return { success: true };
    } catch (error) {
      const message = error?.message || 'Failed to end the browser VoIP call.';
      applyVoipErrorState({ message, code: 'softphone_end_failed' }, message);
      return { success: false, error: message };
    }
  };

  return (
    <VoIPContext.Provider
      value={{
        activeCall,
        answerCall,
        rejectCall,
        voipStatus,
        voipError,
        voipErrorCode,
        voipActionLabel,
        voipActionUrl,
        voipMeta,
        agentEmail,
        retryVoip: () => initSoftphone({ force: true }),
      }}
    >
      {children}
    </VoIPContext.Provider>
  );
}
