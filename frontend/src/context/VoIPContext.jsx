import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Softphone } from '@frejun/softphone-web-sdk';

import { API_BASE } from '../store/useAppStore';

const VoIPContext = createContext({
  activeCall: null,
  answerCall: async () => ({ success: false }),
  rejectCall: async () => ({ success: false }),
  voipStatus: 'disconnected',
  voipError: '',
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
  const [agentEmail, setAgentEmail] = useState('ashwin@growton.co');
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

  const getFreshToken = async () => {
    try {
      const appToken = localStorage.getItem('token') || sessionStorage.getItem('token') || '';
      const res = await fetch(`${API_BASE}/voip/token`, {
        headers: appToken ? { Authorization: `Bearer ${appToken}` } : {},
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        return {
          success: false,
          error: data.detail || data.error || 'Failed to refresh FreJun VoIP token',
        };
      }

      return {
        success: true,
        accessToken: data.access_token || '',
        agentEmail: data.agent_email || '',
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
      setVoipError('');
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
        const message = tokenResult.error || 'FreJun VoIP token is unavailable';
        setVoipError(message);
        setVoipStatus('error');
        return;
      }

      setAgentEmail(tokenResult.agentEmail);

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
            setVoipError('');
            setVoipStatus(current => (current === 'connected' ? current : 'registered'));
            return;
          }

          if (state === 'Error' || maxRetriesReached) {
            const message = error || 'FreJun softphone registration failed';
            setVoipError(String(message));
            setVoipStatus('error');
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
            setVoipError('FreJun created a session, but the browser SDK could not attach to it.');
            setVoipStatus('error');
            return;
          }

          setVoipError('');
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
          setVoipError('');
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
      setVoipError(message);
      setVoipStatus('error');
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
      setVoipError(message);
      setVoipStatus('error');
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
      setVoipError('');
      setVoipStatus('invite_received');

      const accepted = await activeCall.session.accept();
      if (!accepted) {
        const message = 'FreJun did not accept the browser call session.';
        setVoipError(message);
        setVoipStatus('error');
        return { success: false, error: message };
      }

      return { success: true };
    } catch (error) {
      const message = error?.message || 'Failed to answer the browser VoIP call.';
      setVoipError(message);
      setVoipStatus('error');
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
      setVoipStatus('registered');
      return { success: true };
    } catch (error) {
      const message = error?.message || 'Failed to end the browser VoIP call.';
      setVoipError(message);
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
        agentEmail,
        retryVoip: () => initSoftphone({ force: true }),
      }}
    >
      {children}
    </VoIPContext.Provider>
  );
}
