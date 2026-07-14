import { useCallback, useRef, useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../store/useAppStore';
import { longOperationAxios } from '../api/longTimeoutAxios';

const TERMINAL_STATUSES = new Set(['complete', 'low_text', 'failed']);
const POLL_INTERVAL_MS = 2000;
const POLL_MAX_ATTEMPTS = 60; // ~2 minutes

/**
 * Shared resume state for the Talent Pool and Manage Role grids.
 * resumesById overlays the row's own `resume` key with fresher data from
 * uploads/polls; uploadingIds drives the cell spinner.
 */
export default function useResumes({ onParsed } = {}) {
  const [resumesById, setResumesById] = useState({});
  const [uploadingIds, setUploadingIds] = useState(() => new Set());
  const [viewer, setViewer] = useState({ open: false, candidate: null, resume: null });
  const pollTimers = useRef({});

  const patchResume = useCallback((candidateId, resume) => {
    setResumesById((prev) => ({ ...prev, [candidateId]: resume }));
  }, []);

  const stopPolling = useCallback((candidateId) => {
    const timer = pollTimers.current[candidateId];
    if (timer) {
      clearTimeout(timer);
      delete pollTimers.current[candidateId];
    }
  }, []);

  const pollResume = useCallback((candidateId, attempt = 0) => {
    stopPolling(candidateId);
    if (attempt >= POLL_MAX_ATTEMPTS) return;
    pollTimers.current[candidateId] = setTimeout(async () => {
      try {
        const res = await axios.get(`${API_BASE}/candidates/${candidateId}/resume`);
        const resume = res.data?.resume;
        if (resume) {
          patchResume(candidateId, resume);
          setViewer((prev) => (
            prev.open && prev.candidate?.id === candidateId ? { ...prev, resume } : prev
          ));
          if (TERMINAL_STATUSES.has(resume.parse_status)) {
            if (resume.parse_status === 'complete') onParsed?.(candidateId, resume);
            return;
          }
        }
      } catch {
        // Transient poll errors: keep trying until the attempt cap.
      }
      pollResume(candidateId, attempt + 1);
    }, POLL_INTERVAL_MS);
  }, [patchResume, stopPolling, onParsed]);

  const uploadResume = useCallback(async (candidateId, file) => {
    setUploadingIds((prev) => new Set(prev).add(candidateId));
    try {
      const fd = new FormData();
      fd.append('file', file);
      // No explicit Content-Type: the browser must set the multipart boundary.
      const res = await longOperationAxios.post(`${API_BASE}/candidates/${candidateId}/resume`, fd);
      const resume = res.data?.resume;
      if (resume) {
        patchResume(candidateId, resume);
        pollResume(candidateId);
      }
      return { ok: true, resume };
    } catch (err) {
      const detail = err?.response?.data?.detail || 'Resume upload failed';
      return { ok: false, error: detail };
    } finally {
      setUploadingIds((prev) => {
        const next = new Set(prev);
        next.delete(candidateId);
        return next;
      });
    }
  }, [patchResume, pollResume]);

  const openResume = useCallback(async (candidate) => {
    const candidateId = candidate?.id;
    if (!candidateId) return;
    setViewer({ open: true, candidate, resume: resumesById[candidateId] || candidate.resume || null });
    try {
      const res = await axios.get(`${API_BASE}/candidates/${candidateId}/resume`);
      const resume = res.data?.resume;
      if (resume) {
        patchResume(candidateId, resume);
        setViewer((prev) => (prev.open && prev.candidate?.id === candidateId ? { ...prev, resume } : prev));
      }
    } catch {
      // Metadata fetch failure: the modal falls back to the row's own data.
    }
  }, [resumesById, patchResume]);

  const closeResume = useCallback(() => setViewer({ open: false, candidate: null, resume: null }), []);

  const reparseResume = useCallback(async (candidateId) => {
    try {
      const res = await axios.post(`${API_BASE}/candidates/${candidateId}/resume/reparse`);
      const resume = res.data?.resume;
      if (resume) {
        patchResume(candidateId, resume);
        setViewer((prev) => (prev.open && prev.candidate?.id === candidateId ? { ...prev, resume } : prev));
      }
      pollResume(candidateId);
    } catch {
      // The failed banner stays; the user can retry.
    }
  }, [patchResume, pollResume]);

  const resolveResume = useCallback(
    (candidate) => resumesById[candidate?.id] ?? candidate?.resume ?? null,
    [resumesById],
  );

  return {
    resolveResume,
    uploadingIds,
    uploadResume,
    viewer,
    openResume,
    closeResume,
    reparseResume,
  };
}
