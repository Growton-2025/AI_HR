import { useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { toast } from 'sonner';
import { X } from 'lucide-react';

const FIELD_STYLE = {
  width: '100%', padding: '12px 16px', borderRadius: 12, border: '1.5px solid #e2e8f0',
  fontSize: 14, outline: 'none', boxSizing: 'border-box', background: '#fff',
};

const LABEL_STYLE = {
  display: 'block', fontSize: 12, fontWeight: 700, color: '#94a3b8',
  marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.03em',
};

function Field({ label, required, value, onChange, placeholder, type = 'text', multiline = false }) {
  return (
    <div>
      <label style={LABEL_STYLE}>{label}{required && <span style={{ color: '#f97316' }}> *</span>}</label>
      {multiline ? (
        <textarea
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          style={{ ...FIELD_STYLE, minHeight: 80, resize: 'none' }}
        />
      ) : (
        <input
          type={type}
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          style={FIELD_STYLE}
        />
      )}
    </div>
  );
}

export default function AddCandidateModal({ roleId, onClose, onSuccess }) {
  const createCandidate = useAppStore(state => state.createCandidate);
  const [loading, setLoading] = useState(false);
  const [fields, setFields] = useState({
    first_name: '', last_name: '', linkedin: '', city: '', title: '',
    company_name: '', email: '', phone: '', location: '', notes: '', about: '',
  });

  const setField = (key) => (value) => setFields(prev => ({ ...prev, [key]: value }));

  const missingRequired = ['first_name', 'last_name', 'linkedin', 'city', 'title']
    .filter(key => !fields[key].trim());

  const handleSubmit = async () => {
    if (loading) return;
    if (missingRequired.length) {
      toast.error('First name, last name, LinkedIn URL, city, and title are required');
      return;
    }
    setLoading(true);
    try {
      const payload = { ...fields };
      Object.keys(payload).forEach(key => {
        if (typeof payload[key] === 'string') payload[key] = payload[key].trim() || undefined;
      });
      if (roleId) payload.role_id = roleId;

      const res = await createCandidate(payload);
      if (res.success) {
        toast.success(`Added ${fields.first_name} ${fields.last_name}`);
        onSuccess?.(res.data);
      } else {
        toast.error(res.error || 'Failed to add candidate');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000, padding: '24px' }}>
      <div style={{ background: '#fff', borderRadius: '24px', width: '100%', maxWidth: '560px', maxHeight: '90vh', overflowY: 'auto', padding: '32px', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', border: '1px solid #e2e8f0', position: 'relative' }}>
        <button
          onClick={onClose}
          disabled={loading}
          style={{ position: 'absolute', top: '24px', right: '24px', background: 'none', border: 'none', cursor: loading ? 'wait' : 'pointer', color: '#94a3b8', padding: '4px' }}
        >
          <X size={20} />
        </button>
        <h3 style={{ fontSize: '20px', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>Add Candidate</h3>
        <p style={{ color: '#64748b', fontSize: '14px', marginBottom: '24px' }}>
          Add a single candidate directly — the fields below match what CSV upload expects.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
          <Field label="First Name" required value={fields.first_name} onChange={setField('first_name')} placeholder="Jane" />
          <Field label="Last Name" required value={fields.last_name} onChange={setField('last_name')} placeholder="Doe" />
        </div>

        <div style={{ marginBottom: '16px' }}>
          <Field label="LinkedIn URL" required value={fields.linkedin} onChange={setField('linkedin')} placeholder="https://linkedin.com/in/janedoe" />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
          <Field label="City" required value={fields.city} onChange={setField('city')} placeholder="Bengaluru" />
          <Field label="Title" required value={fields.title} onChange={setField('title')} placeholder="Account Executive" />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
          <Field label="Company" value={fields.company_name} onChange={setField('company_name')} placeholder="Acme Inc." />
          <Field label="Location" value={fields.location} onChange={setField('location')} placeholder="Bengaluru, India" />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
          <Field label="Email" type="email" value={fields.email} onChange={setField('email')} placeholder="jane@acme.com" />
          <Field label="Phone" type="tel" value={fields.phone} onChange={setField('phone')} placeholder="+91 98765 43210" />
        </div>

        <div style={{ marginBottom: '16px' }}>
          <Field label="About" multiline value={fields.about} onChange={setField('about')} placeholder="Short profile summary..." />
        </div>

        <div style={{ marginBottom: '24px' }}>
          <Field label="Notes" multiline value={fields.notes} onChange={setField('notes')} placeholder="Any notes for this candidate..." />
        </div>

        <div style={{ display: 'flex', gap: 12 }}>
          <button
            onClick={onClose}
            disabled={loading}
            style={{ flex: 1, padding: '14px', background: '#f1f5f9', color: '#475569', border: 'none', borderRadius: 12, fontWeight: 700, cursor: loading ? 'wait' : 'pointer', opacity: loading ? 0.7 : 1 }}
          >Cancel</button>
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{
              flex: 1, padding: '14px', background: 'var(--accent-primary, #f97316)', color: '#fff', border: 'none', borderRadius: 12,
              fontWeight: 700, cursor: loading ? 'wait' : 'pointer', opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? 'Adding...' : 'Add Candidate'}
          </button>
        </div>
      </div>
    </div>
  );
}
