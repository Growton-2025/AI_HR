import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Softphone } from '@frejun/softphone-web-sdk';

const BACKEND_URL = 'http://127.0.0.1:3002';
const FREJUN_AGENT_EMAIL = 'ashwin@growton.co';

const VoIPContext = createContext({
  activeCall: null,
  answerCall: () => {},
  rejectCall: () => {},
  voipStatus: 'disconnected',
});

export function useVoIP() {
  return useContext(VoIPContext);
}

export function VoIPProvider({ children }) {
  const [activeCall, setActiveCall] = useState(null);
  const [voipStatus, setVoipStatus] = useState('disconnected');
  const softphoneRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const initDoneRef = useRef(false);

  // Hidden audio element for call audio output
  useEffect(() => {
    const audio = document.createElement('audio');
    audio.autoplay = true;
    audio.style.display = 'none';
    document.body.appendChild(audio);
    remoteAudioRef.current = audio;
    return () => document.body.removeChild(audio);
  }, []);

  // Auto-initialize once on mount
  useEffect(() => {
    if (initDoneRef.current) return;
    initDoneRef.current = true;
    initSoftphone();
  }, []);

  const getFreshToken = async () => {
    try {
      const appToken = localStorage.getItem('token') || sessionStorage.getItem('token') || '';
      const res = await fetch(`${BACKEND_URL}/api/voip/token`, {
        headers: appToken ? { Authorization: `Bearer ${appToken}` } : {}
      });
      if (res.ok) {
        const data = await res.json();
        return data.access_token || null;
      }
    } catch (e) {
      console.warn('[VoIP] Could not fetch fresh token:', e.message);
    }
    return null;
  };

  const initSoftphone = async () => {
    try {
      setVoipStatus('connecting');
      console.log('[VoIP] Fetching fresh access token...');
      const accessToken = await getFreshToken();
      if (!accessToken) {
        console.error('[VoIP] No access token available. VoIP disabled.');
        setVoipStatus('error');
        return;
      }
      console.log('[VoIP] Got token. Starting initialization...');

      const sp = new Softphone();
      softphoneRef.current = sp;

      console.log(`[VoIP] Logging in as ${FREJUN_AGENT_EMAIL}...`);
      await sp.login({
        type: 'OAuth2.0',
        token: accessToken,
        email: FREJUN_AGENT_EMAIL,
      });

      console.log('[VoIP] Login OK. Starting listener...');

      const listeners = {
        onConnectionStateChange: (type, state, maxRetriesReached, error) => {
          console.log(`[VoIP] ${type}: ${state}${error ? ' ERR:' + error : ''}`);
          if (type === 'RegisterState' && state === 'Registered') {
            setVoipStatus('registered');
            console.log('[VoIP] ✅ Registered. Waiting for calls...');
          } else if (state === 'Unregistered' || state === 'Terminated' || state === 'Error') {
            setVoipStatus('disconnected');
          }
        },
        onCallCreated: (sessionType, metadata) => {
          console.log('[VoIP] 📞 CALL CREATED!', sessionType, metadata);
          const session = softphoneRef.current?.getSession;
          if (session) {
            setActiveCall({ session, metadata, sessionType });
            session.on('stateChanged', (s) => {
              console.log(`[VoIP] Session state: ${s}`);
              if (s === 'terminated' || s === 'Terminated') {
                setActiveCall(null);
                console.log('[VoIP] Call ended.');
              }
            });
          }
        },
      };

      await sp.start(listeners, { remote: remoteAudioRef.current });
      console.log('[VoIP] ✅ Softphone online and ready!');
    } catch (err) {
      console.error('[VoIP] Init failed:', err.message);
      setVoipStatus('error');
    }
  };

  // Answer the incoming call — same as test console
  const answerCall = async () => {
    if (!activeCall?.session) return;
    try {
      console.log('[VoIP] Answering call...');
      await activeCall.session.accept();
      console.log('[VoIP] Call answered!');
    } catch (e) {
      console.error('[VoIP] Answer failed:', e.message);
    }
  };

  // Reject / hang up the call
  const rejectCall = () => {
    if (!activeCall?.session) return;
    try {
      activeCall.session.end();
    } catch (_) {}
    setActiveCall(null);
  };

  return (
    <VoIPContext.Provider value={{ activeCall, answerCall, rejectCall, voipStatus }}>
      {children}
    </VoIPContext.Provider>
  );
}
