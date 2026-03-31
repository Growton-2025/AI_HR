import React, { useState, useRef, useEffect } from 'react';
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

export function StatusDropdown({ status, candidateId, onUpdate, onShortlisted }) {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleUpdate = async (newStatus) => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/candidates/${candidateId}/status`, { status: newStatus });
      if (newStatus === 'Shortlisted' && onShortlisted) {
        onShortlisted(candidateId);
      }
      onUpdate(candidateId, newStatus);
    } catch (error) {
      console.error('Failed to update status:', error);
    } finally {
      setLoading(false);
      setIsOpen(false);
    }
  };

  const currentStyle = STATUS_STYLES[(status || '').toLowerCase()] || { bg: '#f1f5f9', color: '#475569', dot: '#94a3b8' };

  return (
    <div style={{ position: 'relative' }} ref={dropdownRef}>
      <button
        onClick={() => !loading && setIsOpen(!isOpen)}
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
        {loading ? 'Updating...' : (status || 'Select Status')}
        <ChevronDown size={12} style={{ opacity: 0.5 }} />
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, marginTop: '4px',
          background: '#fff', border: '1px solid #e2e8f0', borderRadius: '12px',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          zIndex: 50, padding: '6px', minWidth: '180px', maxHeight: '300px', overflowY: 'auto'
        }}>
          {RECRUITMENT_STAGES.map(stage => {
            const style = STATUS_STYLES[stage.toLowerCase()] || { dot: '#94a3b8' };
            const isActive = stage === status;
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
        </div>
      )}
    </div>
  );
}

export default StatusDropdown;
