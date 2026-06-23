import React from 'react';
import { X, ChevronDown } from 'lucide-react';

export function TagFilterInput({ label, values, inputValue, onInputChange, onTagsChange, placeholder, icon: Icon }) {
  const tagList = Array.isArray(values) ? values : [];
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault();
      const val = inputValue.trim();
      if (!tagList.includes(val)) {
        onTagsChange([...tagList, val]);
      }
      onInputChange('');
    } else if (e.key === 'Backspace' && !inputValue && tagList.length > 0) {
      onTagsChange(tagList.slice(0, -1));
    }
  };

  const removeTag = (tag) => {
    onTagsChange(tagList.filter(t => t !== tag));
  };

  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 7 }}>
        {label}
      </label>
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ position: 'relative' }}>
          {Icon && (
            <span style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: '#9ca3af', display: 'flex' }}>
              <Icon size={13} />
            </span>
          )}
          <input
            type="text"
            value={inputValue}
            onChange={e => onInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={tagList.length === 0 ? placeholder : "Add more..."}
            style={{
              width: '100%', padding: Icon ? '10px 12px 10px 34px' : '10px 12px',
              background: 'rgba(255,255,255,0.92)', border: '1px solid rgba(203, 213, 225, 0.9)',
              borderRadius: 12, color: '#111827', fontSize: 13,
              outline: 'none', transition: 'border-color 0.15s, box-shadow 0.15s, background 0.15s',
              fontFamily: 'inherit', boxSizing: 'border-box',
              boxShadow: 'inset 0 1px 2px rgba(15,23,42,0.03)',
            }}
            onFocus={e => {
              e.target.style.borderColor = 'rgba(194, 124, 63, 0.5)';
              e.target.style.boxShadow = '0 0 0 3px rgba(194, 124, 63, 0.12)';
              e.target.style.background = '#fff';
            }}
            onBlur={e => {
              e.target.style.borderColor = 'rgba(203, 213, 225, 0.9)';
              e.target.style.boxShadow = 'inset 0 1px 2px rgba(15,23,42,0.03)';
              e.target.style.background = 'rgba(255,255,255,0.92)';
            }}
          />
        </div>
        {tagList.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {tagList.map(tag => (
              <span key={tag} style={{
                display: 'inline-flex', alignItems: 'center', gap: '6px',
                background: '#eef2f6', color: '#334155', padding: '4px 9px',
                borderRadius: '999px', fontSize: '11px', fontWeight: 600,
                border: '1px solid rgba(203, 213, 225, 0.9)'
              }}>
                {tag}
                <X size={12} style={{ cursor: 'pointer', color: '#94a3b8' }} onClick={() => removeTag(tag)} />
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function SelectFilter({ label, value, onChange, options, placeholder }) {
  const opts = [...new Set((Array.isArray(options) ? options : []).map(o => String(o || '').trim()).filter(Boolean))];
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 7 }}>
        {label}
      </label>
      <div style={{ position: 'relative' }}>
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          style={{
            width: '100%', padding: '10px 32px 10px 12px',
            background: 'rgba(255,255,255,0.92)', border: '1px solid rgba(203, 213, 225, 0.9)',
            borderRadius: 12, color: value ? '#111827' : '#94a3b8', fontSize: 13,
            outline: 'none', appearance: 'none', cursor: 'pointer',
            fontFamily: 'inherit', boxSizing: 'border-box',
            boxShadow: 'inset 0 1px 2px rgba(15,23,42,0.03)',
            transition: 'border-color 0.15s, box-shadow 0.15s, background 0.15s',
          }}
          onFocus={e => {
            e.target.style.borderColor = 'rgba(194, 124, 63, 0.5)';
            e.target.style.boxShadow = '0 0 0 3px rgba(194, 124, 63, 0.12)';
            e.target.style.background = '#fff';
          }}
          onBlur={e => {
            e.target.style.borderColor = 'rgba(203, 213, 225, 0.9)';
            e.target.style.boxShadow = 'inset 0 1px 2px rgba(15,23,42,0.03)';
            e.target.style.background = 'rgba(255,255,255,0.92)';
          }}
        >
          <option value="">{placeholder}</option>
          {opts.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <ChevronDown size={13} color="#94a3b8" style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
      </div>
    </div>
  );
}

export function RangeSlider({ label, min, max, minValue, maxValue, onChange }) {
  const minPos = ((minValue - min) / (max - min)) * 100;
  const maxPos = ((maxValue - min) / (max - min)) * 100;

  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <label style={{ fontSize: 11, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          {label}
        </label>
        <span style={{ fontSize: 11, fontWeight: 700, color: '#64748b' }}>
          {minValue} - {maxValue} yrs
        </span>
      </div>

      <div style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center', padding: '0 4px' }}>
        {/* Track Background */}
        <div style={{ position: 'absolute', left: 4, right: 4, height: 4, background: '#d8dee8', borderRadius: 999 }} />

        {/* Active Range Highlight */}
        <div style={{
          position: 'absolute',
          left: `calc(4px + ${minPos}%)`,
          width: `${maxPos - minPos}%`,
          height: 4,
          background: '#0f172a',
          borderRadius: 999,
          zIndex: 1
        }} />

        {/* Dual Range Inputs Layered */}
        <input
          type="range" min={min} max={max} value={minValue}
          onChange={e => {
            const val = Math.min(Number(e.target.value), maxValue - 1);
            onChange(val, maxValue);
          }}
          style={{
            position: 'absolute', width: '100%', left: 0, appearance: 'none', background: 'none',
            pointerEvents: 'none', zIndex: 2, cursor: 'pointer', outline: 'none'
          }}
          className="dual-range-input"
        />
        <input
          type="range" min={min} max={max} value={maxValue}
          onChange={e => {
            const val = Math.max(Number(e.target.value), minValue + 1);
            onChange(minValue, val);
          }}
          style={{
            position: 'absolute', width: '100%', left: 0, appearance: 'none', background: 'none',
            pointerEvents: 'none', zIndex: 3, cursor: 'pointer', outline: 'none'
          }}
          className="dual-range-input"
        />

        <style dangerouslySetInnerHTML={{
          __html: `
          .dual-range-input::-webkit-slider-thumb {
            appearance: none;
            pointer-events: auto;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #0f172a;
            border: 2px solid #fff;
            box-shadow: 0 6px 12px rgba(15,23,42,0.18);
            cursor: pointer;
            z-index: 10;
          }
          .dual-range-input::-moz-range-thumb {
            appearance: none;
            pointer-events: auto;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #0f172a;
            border: 2px solid #fff;
            box-shadow: 0 6px 12px rgba(15,23,42,0.18);
            cursor: pointer;
            z-index: 10;
          }
        `}} />
      </div>
    </div>
  );
}
