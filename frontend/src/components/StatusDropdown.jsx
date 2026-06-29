import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import axios from 'axios';
import { ChevronDown } from 'lucide-react';
import { API_BASE } from '../store/useAppStore';

export const RECRUITMENT_STAGES = [
  'To be started', 'Shortlisted', 'Rejected', 'For Future',
  'Reached out - Linkedin', 'Reached out - Phone', 'Not Interested',
  'Followup / In conversation', 'Shortlist - Rejected', 'High CTC',
  'Duplicate', 'Not responding', 'Internal Review', 'Shared with customer'
];

export const STATUS_STYLES = {
  'to be started': { bg: '#f1f5f9', color: '#64748b', dot: '#94a3b8' },
  'shortlisted': { bg: '#dcfce7', color: '#16a34a', dot: '#16a34a' },
  'rejected': { bg: '#fee2e2', color: '#b91c1c', dot: '#ef4444' },
  'for future': { bg: '#fff7ed', color: '#9a3412', dot: '#f97316' },
  'reached out - linkedin': { bg: '#e0f2fe', color: '#0369a1', dot: '#0284c7' },
  'reached out - phone': { bg: '#dbeafe', color: '#1d4ed8', dot: '#1d4ed8' },
  'not interested': { bg: '#f1f5f9', color: '#475569', dot: '#64748b' },
  'followup / in conversation': { bg: '#fef9c3', color: '#854d0e', dot: '#ca8a04' },
  'shortlist - rejected': { bg: '#fef2f2', color: '#991b1b', dot: '#dc2626' },
  'high ctc': { bg: '#fce7f3', color: '#be185d', dot: '#db2777' },
  'duplicate': { bg: '#f3f4f6', color: '#374151', dot: '#4b5563' },
  'not responding': { bg: '#fff1f2', color: '#9f1239', dot: '#e11d48' },
  'internal review': { bg: '#f3e8ff', color: '#7e22ce', dot: '#9333ea' },
  'shared with customer': { bg: '#ecfdf5', color: '#065f46', dot: '#059669' },
};

export function StatusDropdown({ status, candidateId, onUpdate, onShortlisted, updateStatus, disabled = false }) {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [menuPosition, setMenuPosition] = useState(null);
  const dropdownRef = useRef(null);
  const menuRef = useRef(null);

  const toggleMenu = () => {
    if (loading) return;
    if (isOpen) {
      setIsOpen(false);
      return;
    }

    const rect = dropdownRef.current?.getBoundingClientRect();
    if (!rect) return;

    const menuWidth = 190;
    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceAbove = rect.top;
    const openUp = spaceBelow < 260 && spaceAbove > spaceBelow;

    setMenuPosition({
      left: Math.max(8, Math.min(rect.left, window.innerWidth - menuWidth - 8)),
      top: openUp ? undefined : rect.bottom + 4,
      bottom: openUp ? window.innerHeight - rect.top + 4 : undefined,
      maxHeight: Math.max(140, Math.min(300, (openUp ? spaceAbove : spaceBelow) - 12)),
      width: menuWidth,
    });
    setIsOpen(true);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target) &&
        !menuRef.current?.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };
    const handleViewportChange = (event) => {
      if (event?.type === 'scroll' && menuRef.current?.contains(event.target)) return;
      setIsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    window.addEventListener('resize', handleViewportChange);
    window.addEventListener('scroll', handleViewportChange, true);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      window.removeEventListener('resize', handleViewportChange);
      window.removeEventListener('scroll', handleViewportChange, true);
    };
  }, []);

  const handleUpdate = async (newStatus) => {
    setLoading(true);
    try {
      if (updateStatus) await updateStatus(candidateId, newStatus);
      else await axios.post(`${API_BASE}/candidates/${candidateId}/status`, { status: newStatus });
      if (newStatus === 'Shortlisted' && onShortlisted) {
        await onShortlisted(candidateId);
      }
      onUpdate(candidateId, newStatus);
    } catch (error) {
      console.error('Failed to update status:', error);
    } finally {
      setLoading(false);
      setIsOpen(false);
    }
  };

  const statusNorm = String(status ?? '').trim().toLowerCase();
  const currentStyle = STATUS_STYLES[statusNorm] || { bg: '#f1f5f9', color: '#475569', dot: '#94a3b8' };

  if (disabled) {
    return (
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: '6px',
        padding: '4px 10px', borderRadius: '20px',
        fontSize: '11.5px', fontWeight: 700,
        background: currentStyle.bg, color: currentStyle.color,
        opacity: 0.85,
      }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: currentStyle.dot }} />
        {status != null && status !== '' ? String(status) : '—'}
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }} ref={dropdownRef}>
      <button
        onClick={toggleMenu}
        disabled={loading}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: '6px',
          padding: '4px 10px', borderRadius: '20px',
          fontSize: '11.5px', fontWeight: 700,
          background: currentStyle.bg, color: currentStyle.color,
          border: 'none', cursor: loading ? 'wait' : 'pointer',
          whiteSpace: 'nowrap', transition: 'filter 0.1s',
          opacity: loading ? 0.7 : 1,
        }}
        onMouseEnter={e => e.currentTarget.style.filter = 'brightness(0.95)'}
        onMouseLeave={e => e.currentTarget.style.filter = 'none'}
      >
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: currentStyle.dot }} />
        {loading ? 'Updating...' : (status != null && status !== '' ? String(status) : 'Select Status')}
        <ChevronDown size={12} style={{ opacity: 0.5 }} />
      </button>

      {isOpen && menuPosition && createPortal(
        <div style={{
          position: 'fixed',
          top: menuPosition.top,
          bottom: menuPosition.bottom,
          left: menuPosition.left,
          width: menuPosition.width,
          background: '#fff', border: '1px solid #e2e8f0', borderRadius: '12px',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          zIndex: 10050, padding: '6px', maxHeight: menuPosition.maxHeight, overflowY: 'auto'
        }} ref={menuRef}>
          {RECRUITMENT_STAGES.map(stage => {
            const style = STATUS_STYLES[stage.toLowerCase()] || { dot: '#94a3b8' };
            const isActive = stage === String(status ?? '');
            return (
              <button
                key={stage}
                onClick={() => handleUpdate(stage)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', gap: '8px',
                  padding: '8px 10px', borderRadius: '8px',
                  background: isActive ? '#f8fafc' : 'transparent',
                  border: 'none', cursor: 'pointer', textAlign: 'left',
                  fontSize: '12px', color: isActive ? '#0f172a' : '#475569',
                  fontWeight: isActive ? 700 : 500, transition: 'background 0.1s'
                }}
                onMouseEnter={e => e.currentTarget.style.background = '#f8fafc'}
                onMouseLeave={e => !isActive && (e.currentTarget.style.background = 'transparent')}
              >
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: style.dot }} />
                {stage}
              </button>
            );
          })}
        </div>,
        document.body
      )}
    </div>
  );
}

export default StatusDropdown;
