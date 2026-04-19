import React, { useState, useEffect, useRef } from 'react';
import { Softphone } from '@frejun/softphone-web-sdk';
import { Phone, PhoneOff, Activity, ShieldCheck } from 'lucide-react';
import { toast } from 'sonner';

// Valid JWT access token obtained via OAuth2 authorization code flow
// Expires in 6 hours. Re-run the OAuth flow to refresh if needed.
const ACCESS_TOKEN = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvcmdfaWQiOjQ1OTUyLCJzY29wZSI6Im9hdXRoIiwicmVmcmVzaCI6ZmFsc2UsInRva2VuX3R5cGUiOiJhY2Nlc3MiLCJqdGkiOiIzMjI4NWQ4NC1iMWQ2LTRiYzQtYjY3YS1jNTViZWY2MDk2YWUiLCJpYXQiOjE3NzY1NzU5MDYsImV4cCI6MTc3NjU5NzUwNn0.W7sInfb7z0NtccD2Q4uT0nI1I-Wnh9RTXgmo_2V327npEHjcu6OQhX0MvN4NIEB76ELOXWoZoZXjsNxmO0RYDYxqLAG18-BLM4jgczAbKy2OeSaRfTfpe0eDcYoQ4FZRP1jgvlcWhTwm498BJjkL4h8vCAlb4rW-KcdQn8sZGo05ZBy6ebjFnQTXtUoS0155uePWdVw0J5dQpw0Y2kzgo4i_Qxg7vub_63xQVB756j81-2hIhoRui4A1dI-ebY1Q_2ZCOrk3zuVrV5FoB06sxTD2TQLeGnjsRjrbQQZyJZREnjOHBAkOHCu5BfzgLioy6ZDVOfZNPeA32I1isVPtjA';
const AGENT_EMAIL = 'ashwin@growton.co';

export default function TestVoIP() {
  const [status, setStatus] = useState('Disconnected');
  const [softphone, setSoftphone] = useState(null);
  const [incomingSession, setIncomingSession] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isInitializing, setIsInitializing] = useState(false);
  const remoteAudioRef = useRef(null);
  const softphoneRef = useRef(null);

  const addLog = (msg, isError = false) => {
    const entry = { time: new Date().toLocaleTimeString(), msg, isError };
    setLogs(prev => [entry, ...prev].slice(0, 80));
    if (isError) console.error(msg);
    else console.log(msg);
  };

  const initializeSoftphone = async () => {
    if (isInitializing || softphoneRef.current) return;
    setIsInitializing(true);
    setStatus('Initializing...');
    addLog('Starting VoIP initialization...');

    try {
      // The SDK requires: new Softphone() → login() → start()
      const sp = new Softphone();
      softphoneRef.current = sp;

      addLog(`Attempting login with: ${AGENT_EMAIL}`);

      // Use the real JWT obtained via OAuth2 authorization code flow
      const bearerToken = ACCESS_TOKEN;
      addLog('Calling sp.login(type=OAuth2.0) with real JWT...');
      
      await sp.login({
        type: 'OAuth2.0',
        token: bearerToken,
        email: AGENT_EMAIL
      });

      addLog('Login successful! Starting call listener...');

      // Exact listener keys required by FreJun SDK (from UserAgent.js source)
      const listeners = {
        onConnectionStateChange: (type, state, maxRetriesReached, error) => {
          const msg = `[${type}] ${state}${error ? ' ERR:' + error : ''}`;
          addLog(msg);
          setStatus(state);
        },
        onCallCreated: (sessionType, metadata) => {
          addLog(`CALL CREATED! Type: ${sessionType}`);
          // The session object is available via softphoneRef.current.getSession
          const session = softphoneRef.current?.getSession;
          if (session) {
            setIncomingSession(session);
            toast.info('📞 Incoming VoIP Call!');
            session.on('stateChanged', (s) => {
              addLog(`Session: ${s}`);
              if (s === 'terminated' || s === 'Terminated') {
                setIncomingSession(null);
                addLog('Call ended.');
              }
            });
          }
        }
      };

      await sp.start(listeners, { remote: remoteAudioRef.current });

      setSoftphone(sp);
      setStatus('Ready');
      addLog('Softphone READY. Waiting for calls...');
      toast.success('VoIP Online!');
    } catch (error) {
      const msg = error?.message || String(error);
      addLog(`ERROR: ${msg}`, true);
      setStatus('Error');
      toast.error(msg.slice(0, 80));
      softphoneRef.current = null;
    } finally {
      setIsInitializing(false);
    }
  };

  const handleAnswer = async () => {
    if (!incomingSession) return;
    addLog('Answering...');
    try {
      await incomingSession.accept();  // Correct method from Session.js
      addLog('Call answered!');
    } catch (e) {
      addLog(`Answer failed: ${e.message}`, true);
    }
  };

  const handleHangup = () => {
    if (!incomingSession) return;
    addLog('Hanging up...');
    try { incomingSession.end(); } catch (e) { /* ignore */ }  // Correct method from Session.js
    setIncomingSession(null);
  };

  return (
    <div style={{ padding: '32px', maxWidth: '760px', margin: '0 auto', fontFamily: 'Inter, sans-serif' }}>
      <audio ref={remoteAudioRef} autoPlay style={{ display: 'none' }} />

      <div style={{ background: '#fff', borderRadius: '20px', padding: '28px', border: '1px solid #e2e8f0', boxShadow: '0 4px 24px rgba(0,0,0,0.08)' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h1 style={{ fontSize: '22px', fontWeight: 800, color: '#0f172a', margin: 0 }}>VoIP Test Console</h1>
            <p style={{ color: '#64748b', fontSize: '14px', margin: '4px 0 0' }}>{AGENT_EMAIL}</p>
          </div>
          <div style={{
            padding: '6px 14px', borderRadius: '999px', fontSize: '13px', fontWeight: 700,
            background: status === 'Ready' ? '#ecfdf5' : status === 'Error' ? '#fef2f2' : '#f1f5f9',
            color: status === 'Ready' ? '#10b981' : status === 'Error' ? '#ef4444' : '#64748b',
            display: 'flex', alignItems: 'center', gap: '6px'
          }}>
            <Activity size={13} /> {status}
          </div>
        </div>

        {/* Main action area */}
        {!softphone ? (
          <div style={{ textAlign: 'center', padding: '32px 0' }}>
            <button
              onClick={initializeSoftphone}
              disabled={isInitializing}
              style={{
                padding: '16px 40px', background: '#2563eb', color: '#fff',
                border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: 700,
                cursor: isInitializing ? 'wait' : 'pointer', width: '100%',
                opacity: isInitializing ? 0.7 : 1, transition: 'all 0.2s'
              }}
            >
              {isInitializing ? 'Connecting...' : 'Initialize Softphone'}
            </button>
          </div>
        ) : incomingSession ? (
          <div style={{
            padding: '32px', background: '#eff6ff', borderRadius: '16px',
            border: '2px solid #3b82f6', textAlign: 'center',
            animation: 'ring 1s infinite'
          }}>
            <h2 style={{ color: '#1e40af', fontWeight: 800, fontSize: '22px', marginBottom: '20px' }}>📞 INCOMING CALL</h2>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '16px' }}>
              <button onClick={handleAnswer} style={{ padding: '16px 32px', background: '#10b981', color: '#fff', border: 'none', borderRadius: '12px', fontWeight: 800, cursor: 'pointer', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Phone size={20} /> Answer
              </button>
              <button onClick={handleHangup} style={{ padding: '16px 32px', background: '#ef4444', color: '#fff', border: 'none', borderRadius: '12px', fontWeight: 800, cursor: 'pointer', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <PhoneOff size={20} /> Reject
              </button>
            </div>
          </div>
        ) : (
          <div style={{ padding: '40px', background: '#f8fafc', borderRadius: '16px', textAlign: 'center', border: '1px dashed #cbd5e1' }}>
            <Activity size={36} color="#94a3b8" style={{ marginBottom: '12px' }} />
            <p style={{ color: '#64748b', fontWeight: 600, margin: 0 }}>Listening for incoming call bridge...</p>
          </div>
        )}

        {/* Logs */}
        <div style={{ marginTop: '24px', background: '#0f172a', borderRadius: '14px', padding: '18px' }}>
          <div style={{ color: '#fff', fontWeight: 700, fontSize: '12px', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ShieldCheck size={14} /> LOGS
          </div>
          <div style={{ height: '200px', overflowY: 'auto', display: 'flex', flexDirection: 'column-reverse' }}>
            {logs.length === 0 ? (
              <div style={{ color: '#475569', fontSize: '12px', fontFamily: 'monospace' }}>Waiting...</div>
            ) : logs.map((l, i) => (
              <div key={i} style={{
                fontFamily: 'monospace', fontSize: '12px', marginBottom: '6px',
                color: l.isError ? '#f87171' : '#94a3b8',
                borderLeft: `3px solid ${l.isError ? '#ef4444' : '#1e293b'}`,
                paddingLeft: '10px'
              }}>
                [{l.time}] {l.msg}
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes ring {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.02); }
        }
      `}</style>
    </div>
  );
}
