import { useEffect, useState } from 'react';
import axios from 'axios';
import { toast } from 'sonner';
import { API_BASE } from '../store/useAppStore';
import { useVoIP } from '../context/VoIPContext';
import { CallingModal } from '../pages/Calls';

/**
 * Shows the same in-call modal used for outbound once an INBOUND call is
 * answered — timer, live notes, outcome and wrap-up — instead of leaving the
 * recruiter on a bare audio session with nowhere to record what was said.
 *
 * Mounted in the app shell rather than the Calls page: a candidate can call
 * back while the recruiter is anywhere in the app.
 *
 * CallingModal saves notes and outcome against a `calls` row, so we first ask
 * the backend for a real task for this caller (the same endpoint the 1-Click
 * callback uses). Unknown numbers have no candidate and therefore no task — the
 * audio still works, we just cannot offer the wrap-up UI.
 */
export default function InboundCallModal() {
  const { connectedInbound, clearConnectedInbound } = useVoIP();
  const [call, setCall] = useState(null);

  useEffect(() => {
    if (!connectedInbound?.inboundId) {
      setCall(null);
      return undefined;
    }
    let cancelled = false;
    axios
      .post(`${API_BASE}/calls/inbound/${connectedInbound.inboundId}/callback-task`)
      .then(res => { if (!cancelled) setCall(res.data?.call || null); })
      .catch(() => {
        if (cancelled) return;
        // The call itself is unaffected — only the wrap-up UI is lost.
        toast.error('Connected, but this call could not be linked to a task');
        clearConnectedInbound();
      });
    return () => { cancelled = true; };
  }, [connectedInbound?.inboundId]);

  if (!connectedInbound || !call) return null;

  return (
    <CallingModal
      call={call}
      alreadyConnected
      onRefresh={() => {}}
      onClose={async () => {
        const inboundId = connectedInbound.inboundId;
        clearConnectedInbound();
        setCall(null);
        if (!inboundId) return;
        // Same rule as the outbound callback: the row leaves Pending once the
        // call has been handled, regardless of how it went.
        try {
          await axios.post(`${API_BASE}/calls/inbound/${inboundId}/resolve`, {
            call_id: call.id ?? null,
          });
        } catch (_) {
          toast.error('Call handled, but the callback could not be marked complete');
        }
      }}
    />
  );
}
