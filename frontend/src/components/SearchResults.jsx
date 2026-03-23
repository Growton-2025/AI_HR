import { useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import { Linkedin } from 'lucide-react'

function SearchResults() {
    const {
        searchResults,
        selectedCandidates,
        toggleCandidateSelection,
        candidatePriorities,
        setCandidatePriority,
        candidateFeedback,
        setCandidateFeedback
    } = useAppStore()

    const [expandedSummary, setExpandedSummary] = useState(null)

    const priorityOptions = ['--', 'High', 'Medium', 'Low']

    const getPriorityBadge = (priority) => {
        if (!priority || priority === '--') return null
        const classes = {
            'High': 'priority-badge priority-high',
            'Medium': 'priority-badge priority-medium',
            'Low': 'priority-badge priority-low'
        }
        return <span className={classes[priority]}>{priority}</span>
    }

    const selectedCount = Object.keys(selectedCandidates).length

    return (
        <div>
            <div className="section-label">Screening Results</div>

            <table className="data-table">
                <thead>
                    <tr>
                        <th style={{ width: '40px' }}>#</th>
                        <th>Name</th>
                        <th>Title / Company</th>
                        <th>LinkedIn</th>
                        <th style={{ width: '200px' }}>Summary</th>
                        <th style={{ width: '100px' }}>Priority</th>
                        <th style={{ width: '60px' }}>Select</th>
                    </tr>
                </thead>
                <tbody>
                    {searchResults.map((candidate, idx) => {
                        const primaryRole = candidate.roles?.[0] || {}
                        const id = candidate.id
                        const name = candidate.name || 'N/A'
                        const nameParts = name.split(' ')
                        const firstName = nameParts[0] || 'N/A'
                        const lastName = nameParts.slice(1).join(' ') || ''
                        const summary = candidate.reasoning || 'N/A'
                        const truncatedSummary = summary.length > 60 ? summary.substring(0, 60) + '...' : summary
                        const isSelected = !!selectedCandidates[id]
                        const priority = candidatePriorities[id] || '--'

                        return (
                            <tr key={id} className="animate-slide-up" style={{ animationDelay: `${idx * 0.05}s` }}>
                                <td>{idx + 1}</td>
                                <td>{name}</td>
                                <td>
                                    <div style={{ fontWeight: 600 }}>{primaryRole.title || 'N/A'}</div>
                                    <div style={{ fontSize: '13px', color: '#64748b' }}>{primaryRole.company || 'N/A'}</div>
                                </td>
                                <td>
                                    {candidate.linkedin ? (
                                        <a href={candidate.linkedin} target="_blank" rel="noopener noreferrer" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                                            <Linkedin size={16} /> Profile
                                        </a>
                                    ) : '—'}
                                </td>
                                <td>
                                    <span
                                        style={{ cursor: 'pointer' }}
                                        onClick={() => setExpandedSummary(expandedSummary === id ? null : id)}
                                        title="Click to expand"
                                    >
                                        {truncatedSummary}
                                    </span>
                                </td>
                                <td>
                                    <select
                                        className="select-field"
                                        value={priority}
                                        onChange={(e) => setCandidatePriority(id, e.target.value)}
                                    >
                                        {priorityOptions.map(opt => (
                                            <option key={opt} value={opt}>{opt}</option>
                                        ))}
                                    </select>
                                </td>
                                <td style={{ textAlign: 'center' }}>
                                    <input
                                        type="checkbox"
                                        checked={isSelected}
                                        onChange={() => toggleCandidateSelection(id)}
                                    />
                                </td>
                            </tr>
                        )
                    })}
                </tbody>
            </table>

            {/* Expanded Summary Modal */}
            {expandedSummary && (
                <div className="modal-overlay" onClick={() => setExpandedSummary(null)}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <h3 style={{ marginBottom: '16px' }}>Full Summary</h3>
                        <p style={{ lineHeight: 1.6 }}>
                            {searchResults.find(c => c.id === expandedSummary)?.reasoning || 'N/A'}
                        </p>
                        <button
                            className="btn btn-secondary"
                            style={{ marginTop: '16px' }}
                            onClick={() => setExpandedSummary(null)}
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}

            {/* Feedback Section for Selected Candidates */}
            {selectedCount > 0 && (
                <div style={{ marginTop: '24px' }}>
                    <div className="section-label">Feedback for Selected Candidates</div>

                    <div style={{
                        background: '#f5f3ff',
                        padding: '12px 16px',
                        borderRadius: '8px',
                        borderLeft: '4px solid #7c3aed',
                        marginBottom: '16px'
                    }}>
                        <span style={{ fontWeight: 500, color: '#7c3aed' }}>
                            {selectedCount} candidate(s) selected
                        </span> — Add your feedback below
                    </div>

                    {Object.entries(selectedCandidates).map(([id, candidate]) => {
                        const numId = parseInt(id)
                        const priority = candidatePriorities[numId] || '--'
                        const feedback = candidateFeedback[numId] || ''

                        return (
                            <div key={id} style={{ marginBottom: '16px' }}>
                                <div style={{ fontWeight: 600, marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    {candidate.name}
                                    {getPriorityBadge(priority)}
                                </div>
                                <textarea
                                    className="textarea-field"
                                    placeholder="Enter your feedback about this candidate..."
                                    value={feedback}
                                    onChange={(e) => setCandidateFeedback(numId, e.target.value)}
                                    style={{ minHeight: '80px' }}
                                />
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

export default SearchResults
