import { useAppStore } from '../store/useAppStore'
import { BriefcaseBusiness, ExternalLink, MapPin, ChevronDown, ChevronUp, ShieldCheck, ChevronLeft, ChevronRight, Loader2, UserPlus, CheckCircle2, AlertTriangle, FileSpreadsheet, Linkedin, Globe, StickyNote, Sparkles } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { useState, useEffect, useMemo } from 'react'
import { renderTextWithLinks, SourcesList } from './AiColumnCellDrawer'
import { toast } from 'sonner'

// Internal evidence ids ("ev3") are a join key between the auditor and the
// evidence log. They must never be read by a recruiter; the backend strips
// them, and this is the belt-and-braces on the way to the screen.
export function stripEvidenceIds(text) {
    let out = String(text || '')
    if (!out) return ''
    out = out.replace(/\s*[([{]\s*(?:(?:per|see|via|from|cf\.?)\s+)?(?:ev\d+\s*(?:[,;/&]|and)?\s*)+[)\]}]/gi, '')
    out = out.replace(/\s*\b(?:(?:per|see|via|from|cf\.?)\s+)?ev\d+\b(?:\s*[,;/&]\s*ev\d+\b)*/gi, '')
    out = out.replace(/\s+([,.;:!?])/g, '$1').replace(/([,.;:])\1+/g, '$1').replace(/\(\s*\)|\[\s*\]/g, '').replace(/[ \t]{2,}/g, ' ')
    return out.trim()
}

// Older payloads only carry the internal source label; say it in plain words.
export function humanizeSource(source) {
    const s = String(source || '').trim()
    const lower = s.toLowerCase()
    if (!s) return 'Profile data'
    if (lower.startsWith('uploaded field')) {
        const column = s.replace(/^uploaded fields?\.?/i, '').trim()
        return column ? `Uploaded sheet · ${column}` : 'Uploaded sheet'
    }
    if (lower === 'headline') return 'LinkedIn headline'
    if (lower === 'about') return 'LinkedIn about'
    if (['location', 'city', 'candidate location'].includes(lower)) return 'Profile location'
    if (['enriched profile geography', 'profile text', 'profile'].includes(lower)) return 'LinkedIn profile'
    if (/^role|candidate selling record|employer history|current employer/.test(lower)) return 'Work history'
    if (/web/.test(lower)) return 'Web research'
    if (/notes|response/.test(lower)) return 'Recruiter notes'
    return 'Profile data'
}

function firstSentence(text) {
    const s = stripEvidenceIds(text)
    if (!s) return ''
    const match = s.match(/^.+?[.!?](?=\s|$)/)
    return (match ? match[0] : s).trim()
}

function sourceDomain(sources) {
    const withUrl = (sources || []).find(s => s && s.url)
    if (!withUrl) return ''
    const m = String(withUrl.url).match(/^https?:\/\/(?:www\.)?([^/]+)/i)
    return m ? m[1] : ''
}

function formatExperience(value) {
    const num = Number(value)
    if (!Number.isFinite(num) || num <= 0) return 'Exp: N/A'
    const rounded = Number.isInteger(num) ? num : num.toFixed(1)
    return `Exp: ${rounded} yrs`
}

function ConfidenceDot({ confidence }) {
    const color = confidence === 'high' ? '#16a34a' : confidence === 'medium' ? '#d97706' : '#94a3b8'
    return (
        <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 4,
            fontSize: 11, fontWeight: 700, color, textTransform: 'capitalize',
        }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, display: 'inline-block' }} />
            {confidence || 'medium'} evidence
        </span>
    )
}

function SourceBadge({ sources }) {
    const withUrl = (sources || []).filter(s => s && s.url)
    if (!withUrl.length) return null
    return (
        <a
            href={withUrl[0].url}
            target="_blank"
            rel="noreferrer"
            onClick={e => e.stopPropagation()}
            title={withUrl.length > 1 ? `Plus ${withUrl.length - 1} more source(s)` : withUrl[0].title || ''}
            style={{
                fontSize: 10, color: '#2563eb', textDecoration: 'none',
                display: 'inline-flex', alignItems: 'center', gap: 3,
                background: '#dbeafe', padding: '2px 7px', borderRadius: 6,
                fontWeight: 700, whiteSpace: 'nowrap',
            }}
        >
            <ExternalLink size={10} /> Web source
        </a>
    )
}

function ShortlistStatusBadge({ candidate }) {
    const status = candidate.shortlist_status || (candidate.is_verified_match ? 'shortlisted' : 'shortlisted')
    const config = {
        shortlisted: { label: 'Evidence match', bg: '#dcfce7', color: '#166534', border: '#bbf7d0' },
        verified_match: { label: 'Verified from evidence', bg: '#dcfce7', color: '#166534', border: '#bbf7d0' },
    }
    const item = config[status] || config.shortlisted
    return (
        <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 4,
            fontSize: 10, fontWeight: 800, color: item.color,
            background: item.bg, border: `1px solid ${item.border}`,
            borderRadius: 999, padding: '2px 7px', whiteSpace: 'nowrap',
        }}>
            <ShieldCheck size={10} />
            {item.label}
        </span>
    )
}

// Shimmer shown while LLM reasoning is being generated for this candidate
function PendingReasoningShimmer() {
    return (
        <div style={{
            margin: '8px 0 0',
            background: 'linear-gradient(135deg, #faf5ff, #eff6ff)',
            border: '1px solid #e0e7ff',
            borderRadius: 10,
            padding: '10px 14px',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: '#6366f1',
            fontSize: 12,
            fontWeight: 600,
        }}>
            <Loader2 size={13} className="animate-spin" style={{ flexShrink: 0 }} />
            Generating match reasoning…
            <span style={{
                marginLeft: 'auto', fontSize: 10, fontWeight: 700,
                background: '#e0e7ff', color: '#4338ca',
                borderRadius: 99, padding: '2px 8px',
            }}>Passed evidence filter</span>
        </div>
    )
}

function formatYears(value) {
    const num = Number(value)
    if (!Number.isFinite(num)) return '0 yrs'
    return `${Number.isInteger(num) ? num : num.toFixed(1)} yrs`
}

function truncateText(text, maxLen = 60) {
    const s = String(text || '')
    if (s.length > maxLen) return s.slice(0, maxLen - 3) + '...'
    return s
}

function ScopedTenureChips({ items }) {
    if (!Array.isArray(items) || !items.length) return null
    return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 8 }}>
            {items.slice(0, 5).map((item, index) => (
                <span key={`${item.key || item.label}-${index}`} style={{
                    fontSize: 10.5, fontWeight: 800, padding: '3px 8px', borderRadius: 99,
                    background: '#ecfdf5', color: '#047857', border: '1px solid #a7f3d0',
                }} title={item.label || item.dimension}>
                    {formatYears(item.duration)} {truncateText(item.label || item.dimension, 45)}
                </span>
            ))}
        </div>
    )
}

function EvidenceText({ value }) {
    const text = String(value || '').trim()
    if (!text) return null

    // Bullets arrive as "•", " * " or, from the backend's readable text, "\n- ".
    const markerPattern = /\s*(?:[•●▪◦]|\n-\s|\s\*\s)\s*/g
    const firstMarker = text.search(markerPattern)
    if (firstMarker < 0) {
        return <div style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{text}</div>
    }

    const prefix = text.slice(0, firstMarker).trim().replace(/\s*details:\s*$/i, '').trim()
    const points = text
        .slice(firstMarker)
        .split(markerPattern)
        .map(point => point.trim())
        .filter(Boolean)

    return (
        <div style={{ overflowWrap: 'anywhere' }}>
            {prefix && (
                <div style={{ marginBottom: 6, fontWeight: 700, color: '#334155' }}>{prefix}</div>
            )}
            <ul style={{ margin: 0, paddingLeft: 20, display: 'grid', gap: 4 }}>
                {points.map((point, index) => <li key={`${point.slice(0, 32)}-${index}`}>{point}</li>)}
            </ul>
        </div>
    )
}

function EvidenceDrawer({ candidate }) {
    const evidence = Array.isArray(candidate.evidence_log) ? candidate.evidence_log : []
    const tenure = Array.isArray(candidate.scoped_tenure) ? candidate.scoped_tenure : []
    if (!evidence.length && !tenure.length) return null
    return (
        <div style={{
            marginTop: 10, paddingTop: 10,
            borderTop: '1px solid rgba(99,102,241,0.15)',
            display: 'grid', gap: 8,
        }}>
            {tenure.length > 0 && (
                <div style={{ display: 'grid', gap: 6 }}>
                    {tenure.slice(0, 4).map((item, index) => (
                        <div key={`${item.key || item.label}-${index}`} style={{ fontSize: 12, color: '#334155', lineHeight: 1.45 }}>
                            <strong>{truncateText(item.label || item.dimension, 60)}</strong>: {formatYears(item.duration)} verified against {formatYears(item.required)} required
                        </div>
                    ))}
                </div>
            )}
            {evidence.map((item) => (
                <div key={item.id || `${item.source}-${item.snippet}`} style={{
                    fontSize: 12, color: '#475569', lineHeight: 1.45,
                    padding: '7px 9px', borderRadius: 8,
                    background: '#f8fafc', border: '1px solid #e2e8f0',
                }}>
                    <strong style={{ color: '#0f172a' }}>{item.criterion ? truncateText(stripEvidenceIds(item.criterion), 60) : 'Evidence'}</strong>
                    {` · ${item.where || humanizeSource(item.source)}`}
                    <div style={{ marginTop: 5 }}>
                        <EvidenceText value={stripEvidenceIds(item.source_text || item.snippet || item.value)} />
                    </div>
                </div>
            ))}
        </div>
    )
}

// ── Match explanation: verdict → requirement checklist → details ──────────
// Everything here is rendered from `requirement_breakdown` (one row per
// requirement in the recruiter's query) and `audit_claims`; the quote is the
// stored text the scorer matched and the chip says where it came from.

const PROVENANCE_STYLE = {
    uploaded_sheet: { icon: <FileSpreadsheet size={11} />, bg: '#ecfdf5', color: '#047857', border: '#a7f3d0' },
    linkedin_profile: { icon: <Linkedin size={11} />, bg: '#eff6ff', color: '#1d4ed8', border: '#bfdbfe' },
    role_history: { icon: <BriefcaseBusiness size={11} />, bg: '#f5f3ff', color: '#6d28d9', border: '#ddd6fe' },
    web_research: { icon: <Globe size={11} />, bg: '#fff7ed', color: '#c2410c', border: '#fed7aa' },
    recruiter_notes: { icon: <StickyNote size={11} />, bg: '#fefce8', color: '#a16207', border: '#fde68a' },
    computed: { icon: <Sparkles size={11} />, bg: '#f1f5f9', color: '#475569', border: '#e2e8f0' },
}

function ProvenanceChip({ provenance, where, sources }) {
    const style = PROVENANCE_STYLE[provenance] || PROVENANCE_STYLE.computed
    const url = (sources || []).find(s => s && s.url)?.url
    const label = stripEvidenceIds(where || 'Profile data')
    const inner = (
        <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 10.5, fontWeight: 700,
            padding: '2px 8px', borderRadius: 999, background: style.bg, color: style.color,
            border: `1px solid ${style.border}`, maxWidth: '100%', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }} title={label}>
            {style.icon}
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
            {url && <ExternalLink size={10} />}
        </span>
    )
    if (!url) return inner
    return <a href={url} target="_blank" rel="noreferrer" onClick={e => e.stopPropagation()} style={{ textDecoration: 'none', maxWidth: '100%' }}>{inner}</a>
}

function RequirementRow({ item, claimText }) {
    const qualified = item.status === 'qualified'
    const lead = item.profile_evidence?.[0]
    const quote = stripEvidenceIds(lead?.quote || item.exact_evidence_text?.[0] || '')
    const summary = stripEvidenceIds(claimText || item.evidence_found?.[0]?.summary || item.why_it_supports || '')
    return (
        <div style={{
            display: 'grid', gridTemplateColumns: '18px 1fr', gap: 10, alignItems: 'start',
            padding: '10px 12px', borderRadius: 12,
            background: qualified ? '#ffffff' : '#fffbeb',
            border: `1px solid ${qualified ? '#e2e8f0' : '#fde68a'}`,
            boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)',
        }}>
            <span style={{ color: qualified ? '#16a34a' : '#d97706', marginTop: 1 }}>
                {qualified ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}
            </span>
            <div style={{ minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 13, fontWeight: 800, color: '#0f172a' }}>{stripEvidenceIds(item.requirement)}</span>
                    <span style={{ fontSize: 10, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{item.category}</span>
                </div>
                {summary && (
                    <div style={{ fontSize: 12.5, color: '#334155', lineHeight: 1.55, marginTop: 3 }}>{summary}</div>
                )}
                {qualified && quote && (
                    <blockquote style={{
                        margin: '7px 0 0', padding: '6px 10px', borderLeft: '3px solid #c7d2fe',
                        background: '#f8fafc', borderRadius: '0 8px 8px 0', fontSize: 12, color: '#475569',
                        fontStyle: 'italic', lineHeight: 1.5, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere',
                    }}>
                        “{truncateText(quote, 220)}”
                    </blockquote>
                )}
                {!qualified && item.not_counted?.length > 0 && (
                    <div style={{ fontSize: 11.5, color: '#92400e', marginTop: 5 }}>
                        Not counted: {item.not_counted.slice(0, 2).map(stripEvidenceIds).join(' · ')}
                    </div>
                )}
                {lead && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 7, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 10.5, color: '#94a3b8', fontWeight: 700 }}>from</span>
                        <ProvenanceChip provenance={lead.provenance} where={lead.where} sources={lead.sources?.length ? lead.sources : item.sources} />
                        {item.profile_evidence.length > 1 && (
                            <span style={{ fontSize: 10.5, color: '#94a3b8', fontWeight: 700 }}>+{item.profile_evidence.length - 1} more</span>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}

function RequirementDetails({ breakdown }) {
    return (
        <div style={{ display: 'grid', gap: 12, marginTop: 12, paddingTop: 12, borderTop: '1px solid rgba(99,102,241,0.15)' }}>
            {breakdown.map((item) => (
                <div key={item.key} style={{ display: 'grid', gap: 8 }}>
                    <div style={{ fontSize: 11, fontWeight: 800, color: '#4338ca', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                        {stripEvidenceIds(item.requirement)} · {item.status === 'qualified' ? 'met' : 'not verified'}
                    </div>
                    {(item.profile_evidence || []).map((group, index) => (
                        <div key={`${item.key}-${index}`} style={{
                            display: 'grid', gap: 6, padding: '10px 12px', borderRadius: 12,
                            background: '#ffffff', border: '1px solid #e2e8f0',
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
                                <div style={{ minWidth: 0 }}>
                                    <div style={{ fontSize: 13, fontWeight: 800, color: '#0f172a' }}>{stripEvidenceIds(group.display_title)}</div>
                                    <div style={{ fontSize: 11.5, color: '#64748b' }}>{stripEvidenceIds(group.display_subtitle)}</div>
                                </div>
                                <ProvenanceChip provenance={group.provenance} where={group.where} sources={group.sources} />
                            </div>
                            {group.matched_terms?.length > 0 && (
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                    {group.matched_terms.map((term) => (
                                        <span key={term} style={{ fontSize: 10.5, fontWeight: 700, padding: '2px 8px', borderRadius: 999, background: '#eef2ff', color: '#4338ca', border: '1px solid #c7d2fe' }}>
                                            {stripEvidenceIds(term)}
                                        </span>
                                    ))}
                                </div>
                            )}
                            {group.quote && (
                                <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.55 }}>
                                    <EvidenceText value={stripEvidenceIds(group.quote)} />
                                </div>
                            )}
                        </div>
                    ))}
                    {item.cross_check?.length > 0 && (
                        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#334155', lineHeight: 1.55, display: 'grid', gap: 3 }}>
                            {item.cross_check.slice(0, 4).map((line, index) => <li key={index}>{stripEvidenceIds(line)}</li>)}
                        </ul>
                    )}
                    {item.not_counted?.length > 0 && (
                        <div style={{ fontSize: 11.5, color: '#92400e', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 10, padding: '6px 10px', lineHeight: 1.5 }}>
                            <strong>Not counted:</strong> {item.not_counted.map(stripEvidenceIds).join(' · ')}
                        </div>
                    )}
                    {item.sources?.length > 0 && (
                        <div>
                            <div style={{ fontSize: 10.5, fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>Web sources</div>
                            <SourcesList sources={item.sources} />
                        </div>
                    )}
                </div>
            ))}
        </div>
    )
}

function MatchExplanation({ candidate, expanded, onToggle }) {
    const breakdown = Array.isArray(candidate.requirement_breakdown) ? candidate.requirement_breakdown : []
    const claims = Array.isArray(candidate.audit_claims) ? candidate.audit_claims : []
    const isPending = candidate.shortlist_status === 'pending_reasoning'
    const audited = candidate.auditor_status === 'passed' || candidate.auditor_status === 'citations_repaired'
    const met = breakdown.filter(item => item.status === 'qualified').length
    const firstName = String(candidate.name || '').trim().split(/\s+/)[0] || 'This candidate'
    const narrativeVerdict = breakdown.length
        ? (met === breakdown.length
            ? `${firstName} meets all ${breakdown.length} requirement${breakdown.length === 1 ? '' : 's'} of this search.`
            : `${firstName} meets ${met} of ${breakdown.length} requirements; the rest could not be verified from the stored data.`)
        : firstSentence(candidate.decision_narrative || candidate.answer)
    const verdict = audited && claims.length ? firstSentence(candidate.answer) : narrativeVerdict
    const claimByKey = {}
    claims.forEach(claim => { if (claim?.requirement_key && !claimByKey[claim.requirement_key]) claimByKey[claim.requirement_key] = claim.text })
    const auditNote = candidate.auditor_status === 'audit_unavailable'
        ? 'AI review unavailable — matched on stored data only'
        : candidate.auditor_status === 'overridden_ungrounded_rejection' ? 'Matched on stored data' : ''

    return (
        <div style={{
            margin: '10px 0 0', background: 'linear-gradient(135deg, #faf5ff, #eff6ff)',
            border: '1px solid #e0e7ff', borderRadius: 14, padding: '12px 14px',
        }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, minWidth: 0, flex: 1 }}>
                    <span style={{ color: '#4f46e5', marginTop: 2 }}><Sparkles size={15} /></span>
                    <div style={{ minWidth: 0 }}>
                        {isPending ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#6366f1', fontSize: 12.5, fontWeight: 700 }}>
                                <Loader2 size={13} className="animate-spin" /> Writing the match summary…
                            </div>
                        ) : (
                            <div style={{ fontSize: 13.5, fontWeight: 700, color: '#1e1b4b', lineHeight: 1.5 }}>{verdict || 'Matched on stored evidence.'}</div>
                        )}
                        {breakdown.length > 0 && (
                            <div style={{ fontSize: 11.5, color: '#6366f1', fontWeight: 700, marginTop: 2 }}>
                                {met} of {breakdown.length} requirement{breakdown.length === 1 ? '' : 's'} met{auditNote ? ` · ${auditNote}` : ''}
                            </div>
                        )}
                    </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <ShortlistStatusBadge candidate={candidate} />
                    {!isPending && <ConfidenceDot confidence={candidate.confidence || 'medium'} />}
                </div>
            </div>

            {breakdown.length > 0 && (
                <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
                    {breakdown.map(item => (
                        <RequirementRow key={item.key} item={item} claimText={claimByKey[item.key]} />
                    ))}
                </div>
            )}

            {breakdown.length > 0 && (
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}>
                    <button
                        onClick={onToggle}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 11.5, color: '#6366f1', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4, padding: 0 }}
                    >
                        {expanded ? <><ChevronUp size={13} /> Hide details</> : <><ChevronDown size={13} /> Show the evidence</>}
                    </button>
                </div>
            )}
            {expanded && breakdown.length > 0 && <RequirementDetails breakdown={breakdown} />}
        </div>
    )
}

function QuickAddButton({ candidateId, alreadyAdded, addedRoleName }) {
    const [busy, setBusy] = useState(false)
    const [selectedRole, setSelectedRole] = useState('')
    const { roles, quickAddCandidateToRole } = useAppStore(
        useShallow(state => ({ roles: state.roles, quickAddCandidateToRole: state.quickAddCandidateToRole }))
    )
    const activeRoles = useMemo(() => (roles || []).filter(r => r.activation_status === 'active'), [roles])

    // Default to first active role
    const effectiveRole = selectedRole || (activeRoles[0]?.id ? String(activeRoles[0].id) : '')

    const handleAdd = async (e) => {
        e.stopPropagation()
        if (!effectiveRole) { toast.error('No active role found — activate a role first'); return }
        setBusy(true)
        const result = await quickAddCandidateToRole(candidateId, effectiveRole)
        setBusy(false)
        if (result.success) {
            const d = result.data || {}
            const roleName = activeRoles.find(r => String(r.id) === String(effectiveRole))?.name || 'role'
            const enrichingNote = d.enriching_count > 0 ? ' · enriching contact info' : d.email_queued_count > 0 ? ' · email queued' : ''
            toast.success(`Added to ${roleName}${enrichingNote}`)
        } else {
            toast.error(result.error || 'Could not add to role')
        }
    }

    if (alreadyAdded) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: '#16a34a', fontWeight: 700, marginTop: 10 }}>
                <CheckCircle2 size={13} /> Shortlisted{addedRoleName ? ` · ${addedRoleName}` : ''}
            </div>
        )
    }

    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 10 }} onClick={e => e.stopPropagation()}>
            {activeRoles.length > 1 && (
                <select
                    value={effectiveRole}
                    onChange={e => setSelectedRole(e.target.value)}
                    disabled={busy}
                    style={{
                        fontSize: 11, padding: '3px 6px', borderRadius: 7,
                        border: '1px solid #cbd5e1', background: '#f8fafc',
                        color: '#334155', fontWeight: 600, cursor: 'pointer',
                    }}
                >
                    {activeRoles.map(r => (
                        <option key={r.id} value={r.id}>{r.name}</option>
                    ))}
                </select>
            )}
            <button
                onClick={handleAdd}
                disabled={busy || !effectiveRole}
                style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5,
                    fontSize: 11, fontWeight: 700, padding: '4px 10px',
                    borderRadius: 7, border: '1px solid #6366f1',
                    background: busy ? '#e0e7ff' : '#f5f3ff',
                    color: '#4f46e5', cursor: busy ? 'not-allowed' : 'pointer',
                    transition: 'background 0.15s',
                }}
            >
                {busy ? <Loader2 size={11} className="animate-spin" /> : <UserPlus size={11} />}
                {busy ? 'Adding...' : activeRoles.length === 1 ? `Add to ${activeRoles[0]?.name || 'Role'}` : 'Add to Role'}
            </button>
        </div>
    )
}

// Collapsible section with a pill-count badge in the header
function CollapsibleSection({ icon, label, items, colorScheme, defaultOpen = false }) {
    const [open, setOpen] = useState(defaultOpen)
    if (!items || items.length === 0) return null
    const { bg, color, border } = colorScheme
    return (
        <div style={{ marginTop: 7 }}>
            <button
                onClick={() => setOpen(o => !o)}
                style={{
                    display: 'flex', alignItems: 'center', gap: 6, background: 'none',
                    border: 'none', cursor: 'pointer', padding: '2px 0', width: '100%',
                }}
            >
                <span style={{ fontSize: 10, fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: 4 }}>
                    {icon} {label}
                </span>
                <span style={{ fontSize: 10, fontWeight: 700, background: bg, color, border: `1px solid ${border}`, borderRadius: 99, padding: '1px 6px' }}>
                    {items.length}
                </span>
                <span style={{ marginLeft: 'auto', color: '#94a3b8', display: 'flex', alignItems: 'center' }}>
                    {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                </span>
            </button>
            {open && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 5, paddingLeft: 2 }}>
                    {items.map((item, i) => (
                        <span key={i} style={{ fontSize: 11, fontWeight: 600, padding: '2px 9px', borderRadius: 99, background: bg, color, border: `1px solid ${border}`, whiteSpace: 'nowrap' }}>
                            {item}
                        </span>
                    ))}
                </div>
            )}
        </div>
    )
}

// Parse matched/missing criteria into structured groups
function parseCriteriaGroups(matched, missing) {
    const groups = { industries: [], functions: [], geographies: [], tenure: [], other_matched: [], other_missing: [] }

    const splitValues = (val) => String(val || '').split(',').map(s => s.trim()).filter(Boolean)

    matched.forEach(m => {
        const criterion = typeof m === 'string' ? '' : (m.criterion || '')
        const value = typeof m === 'string' ? m : (m.value || m.criterion || '')
        const lower = criterion.toLowerCase()

        if (lower.includes('industr') || lower.includes('saas') || lower.includes('product')) {
            // Split comma-separated industry list into individual chips
            splitValues(value).forEach(v => groups.industries.push(v))
        } else if (lower.includes('function') || lower.includes('role') || lower.includes('sales dev') || lower.includes('tenure')) {
            // Split comma-separated function list into individual chips
            splitValues(value).forEach(v => groups.functions.push(v))
        } else if (lower.includes('geograph') || lower.includes('location') || lower.includes('region') || lower.includes('country')) {
            splitValues(value).forEach(v => groups.geographies.push(v))
        } else if (lower.includes('experience') || lower.includes('years') || lower.includes('managed')) {
            groups.tenure.push(`${criterion}: ${value}`.replace(/^:\s*/, ''))
        } else {
            groups.other_matched.push(`${criterion}${value ? ': ' + value : ''}`.replace(/^:\s*/, '') || String(m))
        }
    })

    missing.slice(0, 4).forEach(m => {
        const label = typeof m === 'string' ? m : (m.criterion && m.value ? `${m.criterion}: ${m.value}` : m.criterion || JSON.stringify(m))
        groups.other_missing.push(label)
    })
    return groups
}

function CandidateCard({ candidate }) {
    const [expanded, setExpanded] = useState(false)
    const { selectedCandidates, toggleCandidateSelection } = useAppStore(
        useShallow(state => ({ selectedCandidates: state.selectedCandidates, toggleCandidateSelection: state.toggleCandidateSelection }))
    )

    const primaryRole = candidate.roles?.[0] || {}
    const id = candidate.id
    const isSelected = !!selectedCandidates[id]
    const title = primaryRole.title || candidate.headline || 'Current role unavailable'
    const company = primaryRole.company || ''
    const location = candidate.location || candidate.city || 'Location unavailable'

    const isPending = candidate.shortlist_status === 'pending_reasoning'
    const answer = candidate.answer || candidate.reasoning || candidate.summary || ''
    const reasoning = candidate.reasoning || ''
    const hasDetailedReasoning = reasoning && reasoning !== answer && reasoning.length > 30
    const sources = Array.isArray(candidate.sources) ? candidate.sources : []
    const score = candidate.match_score
    const confidence = candidate.confidence || 'medium'
    const matched = candidate.matched_criteria || []
    const missing = candidate.missing_criteria || []
    const hasEvidenceDetails = (candidate.evidence_log?.length || 0) > 0 || (candidate.scoped_tenure?.length || 0) > 0

    const groups = parseCriteriaGroups(matched, missing)
    const hasGroups = matched.length > 0 || missing.length > 0
    const hasBreakdown = Array.isArray(candidate.requirement_breakdown) && candidate.requirement_breakdown.length > 0

    const scoreColor = score > 80 ? { bg: '#e7f6ec', color: '#166534' }
        : score > 60 ? { bg: '#f7f0e4', color: '#8b6b44' }
        : { bg: '#edf2f7', color: '#475569' }

    return (
        <article className={`shortlist-card ${isSelected ? 'selected' : ''}`}>
            <div className="shortlist-card-top">
                <label className="shortlist-check">
                    <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleCandidateSelection(id)}
                        aria-label={`Select ${candidate.name || 'candidate'}`}
                    />
                </label>

                <div className="shortlist-person">
                    <h3>{candidate.name || 'Unnamed candidate'}</h3>
                    <div>{company ? `${title} at ${company}` : title}</div>
                </div>

                <div className="shortlist-meta">
                    <span>{formatExperience(candidate.total_experience_years)}</span>
                    <span>
                        <MapPin size={14} />
                        {location}
                    </span>
                    {score != null && (
                        <span style={{
                            display: 'inline-flex', padding: '3px 9px', borderRadius: 99,
                            background: scoreColor.bg, color: scoreColor.color,
                            fontSize: 11, fontWeight: 700,
                        }}>
                            {score}% evidence fit
                        </span>
                    )}
                    {candidate.linkedin && (
                        <a href={candidate.linkedin} target="_blank" rel="noopener noreferrer">
                            LinkedIn <ExternalLink size={13} />
                        </a>
                    )}
                </div>
            </div>

            {/* Verdict → requirement checklist → details. Payloads from a
                backend that predates requirement_breakdown fall back to the
                plain answer + evidence drawer (ids never shown). */}
            {hasBreakdown && (
                <MatchExplanation candidate={candidate} expanded={expanded} onToggle={() => setExpanded(e => !e)} />
            )}
            {!hasBreakdown && isPending && <PendingReasoningShimmer />}
            {!hasBreakdown && !isPending && answer && (
                <div style={{
                    margin: '8px 0 0',
                    background: 'linear-gradient(135deg, #faf5ff, #eff6ff)',
                    border: '1px solid #e0e7ff',
                    borderRadius: 10,
                    padding: '10px 14px',
                    fontSize: 13,
                    color: '#1e1b4b',
                    lineHeight: 1.65,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                }}>
                    {renderTextWithLinks(stripEvidenceIds(answer))}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                            <span style={{ fontSize: 10, color: '#6366f1', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                                AI Match Analysis
                            </span>
                            <ShortlistStatusBadge candidate={candidate} />
                            <ConfidenceDot confidence={confidence} />
                            <SourceBadge sources={sources} />
                        </div>
                        {(hasDetailedReasoning || hasEvidenceDetails) && (
                            <button
                                onClick={() => setExpanded(e => !e)}
                                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 11, color: '#6366f1', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 3, padding: 0 }}
                            >
                                {expanded ? <><ChevronUp size={13} /> Less</> : <><ChevronDown size={13} /> Evidence</>}
                            </button>
                        )}
                    </div>
                    {expanded && <EvidenceDrawer candidate={candidate} />}
                </div>
            )}

            {/* Legacy chips: only when the checklist is not available */}
            {!hasBreakdown && !isPending && hasGroups && (
                <div style={{ marginTop: 8, padding: '8px 12px', background: '#f8fafc', border: '1px solid #e8edf3', borderRadius: 10 }}>
                    <CollapsibleSection
                        icon="🏭" label="Industries"
                        items={groups.industries}
                        colorScheme={{ bg: '#eff6ff', color: '#1e40af', border: '#bfdbfe' }}
                        defaultOpen={groups.industries.length <= 4}
                    />
                    <CollapsibleSection
                        icon="⚡" label="Functions"
                        items={groups.functions}
                        colorScheme={{ bg: '#f0fdf4', color: '#166534', border: '#bbf7d0' }}
                        defaultOpen={groups.functions.length <= 4}
                    />
                    <CollapsibleSection
                        icon="📍" label="Geographies"
                        items={groups.geographies}
                        colorScheme={{ bg: '#fef3c7', color: '#92400e', border: '#fde68a' }}
                        defaultOpen={true}
                    />
                    {groups.tenure.length > 0 && (
                        <CollapsibleSection
                            icon="⏱" label="Experience"
                            items={groups.tenure}
                            colorScheme={{ bg: '#ecfdf5', color: '#047857', border: '#a7f3d0' }}
                            defaultOpen={true}
                        />
                    )}
                    {groups.other_matched.length > 0 && (
                        <CollapsibleSection
                            icon="✓" label="Other Matches"
                            items={groups.other_matched}
                            colorScheme={{ bg: '#ecfdf5', color: '#065f46', border: '#a7f3d0' }}
                            defaultOpen={false}
                        />
                    )}
                    {groups.other_missing.length > 0 && (
                        <CollapsibleSection
                            icon="⚠" label="Gaps"
                            items={groups.other_missing}
                            colorScheme={{ bg: '#fff7ed', color: '#9a3412', border: '#fed7aa' }}
                            defaultOpen={true}
                        />
                    )}
                </div>
            )}

            <ScopedTenureChips items={candidate.scoped_tenure} />

            {candidate.contributing_roles_details?.roles?.length > 0 && (
                <div className="shortlist-role-strip">
                    <BriefcaseBusiness size={14} />
                    {candidate.contributing_roles_details.roles.slice(0, 3).map((role) => (
                        <span key={`${role.company}-${role.title}`}>
                            {role.title || 'Role'}{role.company ? `, ${role.company}` : ''}
                        </span>
                    ))}
                </div>
            )}

            <QuickAddButton
                candidateId={id}
                alreadyAdded={candidate.status === 'Shortlisted' || !!candidate._addedToRole}
                addedRoleName={candidate._addedToRole || null}
            />
        </article>
    )
}

function SearchResults() {
    const searchResults = useAppStore(state => state.searchResults)
    const searchDebug = useAppStore(state => state.searchDebug)
    const [page, setPage] = useState(1)
    const [pageSize, setPageSize] = useState(25)

    const rankedResults = useMemo(() => {
        const statusRank = { verified_match: 0, shortlisted: 0, pending_reasoning: 99 }
        return [...searchResults].sort((left, right) => {
            const leftStatus = left.shortlist_status || 'shortlisted'
            const rightStatus = right.shortlist_status || 'shortlisted'
            const leftRank = statusRank[leftStatus] ?? 1
            const rightRank = statusRank[rightStatus] ?? 1
            if (leftRank !== rightRank) return leftRank - rightRank
            const scoreDelta = Number(right.match_score || 0) - Number(left.match_score || 0)
            if (scoreDelta !== 0) return scoreDelta
            return Number(right.total_experience_years || 0) - Number(left.total_experience_years || 0)
        })
    }, [searchResults])

    const totalPages = Math.max(1, Math.ceil(rankedResults.length / pageSize))

    useEffect(() => {
        if (page > totalPages) setPage(totalPages)
    }, [totalPages, page])

    const startIndex = (page - 1) * pageSize
    const endIndex = startIndex + pageSize
    const displayedResults = rankedResults.slice(startIndex, endIndex)

    return (
        <div className="shortlist-results">
            <div className="shortlist-results-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px' }}>
                    <div className="section-label">Qualified Matches ({searchResults.length})</div>
                    {searchResults.length > 0 && (
                        <div className="shortlist-sort-note">
                            {searchDebug?.semantic_pool_count
                                ? `${searchDebug.returned ?? searchResults.length} returned · ${searchDebug.passed ?? searchResults.length} qualified from ${searchDebug.semantic_pool_count} reviewed`
                                : 'Sorted by strict fit score'}
                        </div>
                    )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '13px', color: '#64748b' }}>Cards per page:</span>
                    <select
                        value={pageSize}
                        onChange={e => {
                            setPageSize(Number(e.target.value))
                            setPage(1)
                        }}
                        style={{
                            padding: '4px 8px', borderRadius: '8px', border: '1px solid rgba(203, 213, 225, 0.9)',
                            fontSize: '12px', fontWeight: 600, color: '#0f172a', outline: 'none',
                            background: '#fff', cursor: 'pointer'
                        }}
                    >
                        <option value={10}>10</option>
                        <option value={25}>25</option>
                        <option value={50}>50</option>
                        <option value={100}>100</option>
                        <option value={200}>200</option>
                        <option value={500}>500</option>
                    </select>
                </div>
            </div>

            <div className="shortlist-card-list">
                {displayedResults.map((candidate) => (
                    <CandidateCard key={candidate.id} candidate={candidate} />
                ))}
            </div>

            {searchResults.length > 0 && (
                <div style={{ padding: '14px 18px', background: 'rgba(248,250,252,0.78)', borderTop: '1px solid #eef2f7', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottomLeftRadius: '12px', borderBottomRightRadius: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                        <span style={{ fontSize: 13, color: '#64748b' }}>
                            Showing {Math.min(startIndex + 1, searchResults.length)}–{Math.min(endIndex, searchResults.length)} of <strong style={{ color: '#0f172a' }}>{searchResults.length}</strong> qualified matches
                        </span>
                    </div>
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                            style={{ width: 34, height: 34, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)', borderRadius: 10, cursor: page === 1 ? 'not-allowed' : 'pointer', opacity: page === 1 ? 0.4 : 1 }}>
                            <ChevronLeft size={14} color="#64748b" />
                        </button>

                        {(() => {
                            const pages = [];
                            const range = 2;
                            for (let i = 1; i <= totalPages; i++) {
                                if (i === 1 || i === totalPages || (i >= page - range && i <= page + range)) {
                                    pages.push(
                                        <button
                                            key={i}
                                            onClick={() => setPage(i)}
                                            style={{
                                                width: 34, height: 34, borderRadius: 10, fontSize: 13, fontWeight: i === page ? 700 : 600,
                                                background: i === page ? '#f97316' : '#fff', color: i === page ? '#fff' : '#64748b',
                                                border: i === page ? 'none' : '1px solid rgba(203, 213, 225, 0.9)', cursor: 'pointer'
                                            }}
                                        >
                                            {i}
                                        </button>
                                    );
                                } else if (i === page - range - 1 || i === page + range + 1) {
                                    pages.push(<span key={i} style={{ color: '#94a3b8', margin: '0 4px' }}>...</span>);
                                }
                            }
                            return pages;
                        })()}

                        <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
                            style={{ width: 34, height: 34, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)', borderRadius: 10, cursor: page === totalPages ? 'not-allowed' : 'pointer', opacity: page === totalPages ? 0.4 : 1 }}>
                            <ChevronRight size={14} color="#64748b" />
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

export default SearchResults
