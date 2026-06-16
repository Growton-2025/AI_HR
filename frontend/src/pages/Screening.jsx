import { startTransition, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { useAppStore } from '../store/useAppStore'
import SearchResults from '../components/SearchResults'
import { AlertCircle, Check, Cpu, Database, Globe, Loader2, RotateCcw, Search, SearchCode, SearchX, X } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'

function buildRetrySuggestions(query) {
    const trimmedQuery = (query || '').trim()
    if (!trimmedQuery) return []

    const suggestions = []
    const companyMatch = trimmedQuery.match(/\b(?:working|worked|currently working|currently at|from|at|in)\s+([a-z0-9&.' -]+)$/i)
    const companyHint = companyMatch?.[1]?.trim().replace(/[.,]+$/, '')

    if (companyHint && companyHint.length >= 3) {
        suggestions.push(`People from ${companyHint}`)
        suggestions.push(`Candidates with experience at ${companyHint}`)
        suggestions.push(`${companyHint} sales candidates`)
    } else {
        suggestions.push(trimmedQuery.replace(/\b(senior|lead|principal|staff|director)\b/gi, '').replace(/\s+/g, ' ').trim())
        suggestions.push(`${trimmedQuery} in India`)
        suggestions.push(`Candidates similar to ${trimmedQuery}`)
    }

    return [...new Set(suggestions.map(suggestion => suggestion.trim()).filter(suggestion => suggestion && suggestion.toLowerCase() !== trimmedQuery.toLowerCase()))].slice(0, 3)
}

function sourceValueToPayload(value) {
    if (!value || value === 'master') {
        return { sourceType: 'master', sourceRoleId: null }
    }
    const roleId = Number(String(value).replace('role:', ''))
    return {
        sourceType: 'role',
        sourceRoleId: Number.isFinite(roleId) ? roleId : null,
    }
}

function Screening() {
    const {
        searchQuery,
        setSearchQuery,
        isSearching,
        searchProgress,
        statusMessage,
        searchResults,
        searchDebug,
        usage,
        searchOutcome,
        lastSearchError,
        searchCandidatesStream,
        stopSearch,
        pauseSearch,
        resumeSearch,
        isSearchPaused,
        clearSearch,
        selectedCandidates,
        clearSelections,
        roles,
        fetchRoles,
        assignCandidatesToRole,
    } = useAppStore(useShallow((state) => ({
        searchQuery: state.searchQuery,
        setSearchQuery: state.setSearchQuery,
        isSearching: state.isSearching,
        searchProgress: state.searchProgress,
        statusMessage: state.statusMessage,
        searchResults: state.searchResults,
        searchDebug: state.searchDebug,
        usage: state.usage,
        searchOutcome: state.searchOutcome,
        lastSearchError: state.lastSearchError,
        searchCandidatesStream: state.searchCandidatesStream,
        stopSearch: state.stopSearch,
        pauseSearch: state.pauseSearch,
        resumeSearch: state.resumeSearch,
        isSearchPaused: state.isSearchPaused,
        clearSearch: state.clearSearch,
        selectedCandidates: state.selectedCandidates,
        clearSelections: state.clearSelections,
        roles: state.roles,
        fetchRoles: state.fetchRoles,
        assignCandidatesToRole: state.assignCandidatesToRole,
    })))

    const [inputQuery, setInputQuery] = useState(searchQuery)
    // Track what the user EXPLICITLY picked; null means "use the first available role"
    const [userPickedSource, setUserPickedSource] = useState(null)
    const [destinationRole, setDestinationRole] = useState('')
    const [isAddingToRole, setIsAddingToRole] = useState(false)
    const [useWebSearch, setUseWebSearch] = useState(false)
    const selectedCount = Object.keys(selectedCandidates).length

    // Derive the effective source: default to first role, fall back to master only if no roles exist
    const sourceValue = userPickedSource ?? (roles.length > 0 ? `role:${roles[0].id}` : 'master')

    const handleSourceChange = (val) => {
        setUserPickedSource(val)
    }

    useEffect(() => {
        fetchRoles({ force: true })
    }, [fetchRoles])

    useEffect(() => {
        if (!destinationRole && roles.length > 0) {
            setDestinationRole(roles[0].name)
        }
    }, [destinationRole, roles])

    const selectedSourceLabel = useMemo(() => {
        if (sourceValue === 'master') return 'Master List'
        const roleId = Number(String(sourceValue).replace('role:', ''))
        return roles.find(role => Number(role.id) === roleId)?.name || 'Selected role'
    }, [roles, sourceValue])

    const handleSearch = (queryOverride) => {
        const nextQuery = typeof queryOverride === 'string'
            ? queryOverride.trim()
            : inputQuery.trim()
        if (!nextQuery) return

        if (nextQuery !== inputQuery) {
            setInputQuery(nextQuery)
        }

        clearSelections()
        setSearchQuery(nextQuery)
        const sourcePayload = sourceValueToPayload(sourceValue)
        startTransition(() => {
            searchCandidatesStream(nextQuery, {
                ...sourcePayload,
                useWebSearch,
            })
        })
    }

    const handleClear = () => {
        setInputQuery('')
        clearSelections()
        startTransition(() => {
            clearSearch()
        })
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSearch()
        }
    }

    const handleAddToRole = async () => {
        if (!destinationRole) {
            toast.error('Choose a destination role')
            return
        }
        const candidateIds = Object.keys(selectedCandidates).map(id => Number(id)).filter(Number.isFinite)
        if (!candidateIds.length) {
            toast.error('Select at least one profile')
            return
        }

        setIsAddingToRole(true)
        const assignments = candidateIds.map(candidateId => ({
            candidate_id: candidateId,
            priority: '--',
            feedback: '',
        }))
        const result = await assignCandidatesToRole(destinationRole, assignments)
        setIsAddingToRole(false)

        if (!result.success) {
            toast.error(result.error || 'Failed to add profiles')
            return
        }

        toast.success(result.data?.message || `Added ${candidateIds.length} profile(s) to ${destinationRole}`)
        clearSelections()
    }

    const suggestions = [
        'BDRs with US experience and 5+ years in SaaS',
        'Business development executives in enterprise SaaS',
        'Sales leaders with fintech product experience',
        'Account executives with outbound US market experience',
    ]

    const retrySuggestions = useMemo(() => buildRetrySuggestions(searchQuery || inputQuery), [inputQuery, searchQuery])
    const hasResults = searchResults.length > 0
    const reviewedCount = Number(searchDebug?.semantic_pool_count || searchDebug?.total_reviewed || 0)
    const qualifiedCount = Number(searchDebug?.passed ?? searchResults.length)
    const failedCount = Number(searchDebug?.failed || 0)
    const hasSettledSearch = hasResults || ['empty', 'error', 'cancelled'].includes(searchOutcome)
    const shouldShowStage = !hasResults
    const steps = [
        { label: 'Analyzing Requirements', icon: Search },
        { label: 'Evaluating Candidate Pool', icon: Database },
        { label: 'Scoring Candidate Relevance', icon: Cpu },
        { label: 'Finalizing Shortlist', icon: Check },
    ]

    let currentStepIndex = 0
    if (searchProgress > 25) currentStepIndex = 1
    if (searchProgress > 50) currentStepIndex = 2
    if (searchProgress > 85) currentStepIndex = 3

    return (
        <div className="screening-workspace">
            <div className="screening-title-row">
                <div>
                    <h2 className="screen-header">
                        AI Shortlisting Engine
                    </h2>
                    <p className="subtitle">
                        Scan a selected candidate list or the master pool to isolate top-matching profiles.
                    </p>
                </div>
            </div>

            <div className="shortlist-search-shell">
                <div className="shortlist-query-panel">
                    <label className="shortlist-field-label">Search Query Requirements</label>
                    <div className="shortlist-query-box">
                        <textarea
                            className="search-textarea shortlist-textarea"
                            placeholder="Describe the profile you're looking for..."
                            value={inputQuery}
                            onChange={(e) => setInputQuery(e.target.value)}
                            onKeyDown={handleKeyDown}
                        />
                        {inputQuery && (
                            <button
                                className="shortlist-clear-input"
                                onClick={() => setInputQuery('')}
                                aria-label="Clear query"
                            >
                                <X size={18} />
                            </button>
                        )}
                    </div>
                    <label
                        style={{
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: 10,
                            marginTop: 10,
                            padding: '10px 12px',
                            borderRadius: 8,
                            border: '1px solid #dbe3ee',
                            background: useWebSearch ? '#f5f3ff' : '#fff',
                            cursor: isSearching ? 'not-allowed' : 'pointer',
                        }}
                    >
                        <input
                            type="checkbox"
                            checked={useWebSearch}
                            disabled={isSearching}
                            onChange={(event) => setUseWebSearch(event.target.checked)}
                            style={{ marginTop: 3 }}
                        />
                        <Globe size={16} color="#4338ca" style={{ flexShrink: 0, marginTop: 1 }} />
                        <span style={{ fontSize: 12, color: '#334155', lineHeight: 1.45 }}>
                            <span style={{ fontWeight: 800, color: '#0f172a' }}>Use web search</span>
                            <span style={{ display: 'block', marginTop: 3, color: '#64748b' }}>
                                Adds live company research for competitors, funding, offices, category, and final reasoning. Slower but broader.
                            </span>
                        </span>
                    </label>
                </div>

                <div className="shortlist-source-panel">
                    <label className="shortlist-field-label">Candidate List</label>
                    <select
                        className="select-field shortlist-source-select"
                        value={sourceValue}
                        onChange={(event) => handleSourceChange(event.target.value)}
                    >
                        <option value="master">Master List</option>
                        {roles.map(role => (
                            <option key={role.id || role.name} value={`role:${role.id}`}>
                                {role.name}{Number.isFinite(Number(role.candidate_count)) ? ` (${role.candidate_count})` : ''}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="shortlist-search-actions">
                    <div className="suggestion-chips">
                        {suggestions.map((suggestion) => (
                            <button key={suggestion} type="button" className="chip" onClick={() => setInputQuery(suggestion)}>
                                {suggestion}
                            </button>
                        ))}
                    </div>
                    <div className="shortlist-button-row">
                        <button className="btn btn-secondary" onClick={handleClear}>
                            Clear
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={handleSearch}
                            disabled={!inputQuery.trim() || isSearching}
                        >
                            {isSearching ? <Loader2 size={16} className="animate-spin" /> : <SearchCode size={16} />}
                            {isSearching ? 'Evaluating Pool...' : 'Run AI Shortlist'}
                        </button>
                    </div>
                </div>
            </div>

            {shouldShowStage && (
                <div className="screening-stage-shell">
                    {isSearching && (
                        <div className="screening-stage-card screening-stage-loading">
                            <div className="loader-container" style={{ minHeight: 'auto' }}>
                                <div className="ai-brain-pulse">
                                    <SearchCode size={40} />
                                </div>

                                <h2 style={{ fontSize: '24px', fontWeight: '700', color: '#1e293b', marginBottom: '8px' }}>
                                    Evaluating {selectedSourceLabel}
                                </h2>

                                <div className="screening-status-chip">
                                    <SearchCode size={14} />
                                    <span>{statusMessage || 'Evaluating candidate pool...'}</span>
                                </div>

                                <div className="screening-progress-track">
                                    <div
                                        className="screening-progress-fill"
                                        style={{ width: `${Math.max(searchProgress, 8)}%` }}
                                    />
                                </div>

                                <div className="loading-steps" style={{ marginTop: '8px' }}>
                                    {steps.map((step, index) => {
                                        const Icon = step.icon
                                        let statusClass = ''
                                        if (index < currentStepIndex) statusClass = 'completed'
                                        else if (index === currentStepIndex) statusClass = 'active'

                                        return (
                                            <div key={step.label} className={`step-item ${statusClass}`}>
                                                <div className="step-icon">
                                                    {index < currentStepIndex ? <Check size={14} /> : <Icon size={14} />}
                                                </div>
                                                <div style={{ flex: 1, fontSize: '14px', fontWeight: index === currentStepIndex ? '600' : '500' }}>
                                                    {step.label}
                                                </div>
                                                {index === currentStepIndex && <Loader2 size={16} className="animate-spin" style={{ color: '#7c3aed' }} />}
                                            </div>
                                        )
                                    })}
                                </div>

                                <div style={{ display: 'flex', gap: '12px', marginTop: '28px', justifyContent: 'center' }}>
                                    <button
                                        className="btn btn-secondary"
                                        onClick={isSearchPaused ? resumeSearch : pauseSearch}
                                        style={{ fontSize: '13px', padding: '8px 16px', background: isSearchPaused ? '#e0e7ff' : '#fff' }}
                                    >
                                        {isSearchPaused ? 'Resume Screening' : 'Pause Screening'}
                                    </button>
                                    <button
                                        className="btn btn-secondary"
                                        onClick={stopSearch}
                                        style={{ fontSize: '13px', padding: '8px 16px' }}
                                    >
                                        Cancel Screening
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {!isSearching && !hasSettledSearch && (
                        <div className="screening-stage-card screening-stage-idle">
                            <div className="screening-idle-header">
                                <div className="screening-idle-badge">
                                    <SearchCode size={14} />
                                    <span>Role-aware AI screening</span>
                                </div>
                                <h3>Search inside a role list or the complete master pool</h3>
                                <p>
                                    Choose a candidate list, describe the requirement, and review explainable matches before adding profiles into a role.
                                </p>
                            </div>
                        </div>
                    )}

                    {!isSearching && searchOutcome === 'empty' && (
                        <div className="screening-stage-card screening-stage-empty">
                            <div className="screening-empty-icon">
                                <SearchX size={28} />
                            </div>
                            <h3>No close matches yet</h3>
                            <p>
                                I couldn&apos;t find a strong match for <strong>{searchQuery}</strong> inside <strong>{selectedSourceLabel}</strong>.
                            </p>
                            {retrySuggestions.length > 0 && (
                                <div className="screening-empty-actions">
                                    {retrySuggestions.map((suggestion) => (
                                        <button
                                            key={suggestion}
                                            type="button"
                                            className="screening-suggestion-button"
                                            onClick={() => handleSearch(suggestion)}
                                        >
                                            {suggestion}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {!isSearching && searchOutcome === 'error' && (
                        <div className="screening-stage-card screening-stage-error">
                            <div className="screening-empty-icon error">
                                <AlertCircle size={28} />
                            </div>
                            <h3>Screening hit a snag</h3>
                            <p>{lastSearchError || statusMessage || 'The search could not be completed right now.'}</p>
                            <div className="screening-empty-actions">
                                <button
                                    type="button"
                                    className="screening-suggestion-button"
                                    onClick={() => handleSearch(searchQuery || inputQuery)}
                                >
                                    <RotateCcw size={14} />
                                    Retry Search
                                </button>
                                <button
                                    type="button"
                                    className="screening-suggestion-button"
                                    onClick={handleClear}
                                >
                                    Start Fresh
                                </button>
                            </div>
                        </div>
                    )}

                    {!isSearching && searchOutcome === 'cancelled' && (
                        <div className="screening-stage-card screening-stage-cancelled">
                            <div className="screening-idle-badge">
                                <AlertCircle size={14} />
                                <span>Screening stopped</span>
                            </div>
                            <h3>Your previous search was cancelled</h3>
                            <p>Update the query and run it again whenever you&apos;re ready.</p>
                        </div>
                    )}
                </div>
            )}

            {hasResults && (
                <div className="screening-results-panel">
                    {isSearching && (
                        <div style={{ padding: '16px 20px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', marginBottom: '20px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '13px', fontWeight: '600', color: '#6366f1', marginBottom: '10px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <button 
                                        onClick={isSearchPaused ? resumeSearch : pauseSearch} 
                                        style={{ 
                                            padding: '4px 8px', fontSize: '11px', fontWeight: 'bold', 
                                            borderRadius: '6px', border: '1px solid #c7d2fe', 
                                            background: isSearchPaused ? '#e0e7ff' : '#fff',
                                            color: '#4f46e5', cursor: 'pointer'
                                        }}
                                    >
                                        {isSearchPaused ? 'Resume' : 'Pause'}
                                    </button>
                                    <Loader2 size={16} className={isSearchPaused ? "" : "animate-spin"} style={{ color: isSearchPaused ? '#94a3b8' : '#6366f1' }} />
                                    <span style={{ color: isSearchPaused ? '#64748b' : '#6366f1' }}>
                                        {isSearchPaused ? 'Evaluation paused.' : (statusMessage || 'Evaluating more candidates in the background...')}
                                    </span>
                                </div>
                                <span>{searchProgress}%</span>
                            </div>
                            <div className="screening-progress-track" style={{ height: '6px', margin: 0, borderRadius: '99px' }}>
                                <div className="screening-progress-fill" style={{ width: `${Math.max(searchProgress, 5)}%`, transition: 'width 0.3s ease', borderRadius: '99px' }} />
                            </div>
                        </div>
                    )}
                    <div className="result-banner">
                        <div className="result-banner-content">
                            <div className="result-banner-title">{searchResults.length} Qualified Match{searchResults.length === 1 ? '' : 'es'}</div>
                            <div className="result-banner-subtitle">
                                <strong>{selectedSourceLabel}</strong> · {searchQuery}
                                {reviewedCount > 0 && (
                                    <span> · {qualifiedCount} qualified after strict filters from {reviewedCount} reviewed{failedCount > 0 ? ` (${failedCount} rejected)` : ''}</span>
                                )}
                            </div>
                        </div>
                        {usage && (
                            <div style={{ color: '#64748b', fontSize: '12px', textAlign: 'right' }}>
                                <div>{usage.total_tokens} tokens</div>
                                <div>${usage.total_cost} USD</div>
                            </div>
                        )}
                    </div>

                    <SearchResults />
                    <div style={{ height: selectedCount > 0 ? '96px' : '32px' }} />
                </div>
            )}

            {selectedCount > 0 && (
                <div className="shortlist-action-bar">
                    <div className="shortlist-selected-count">{selectedCount}</div>
                    <div className="shortlist-selected-label">profiles selected</div>
                    <select
                        className="select-field shortlist-destination-select"
                        value={destinationRole}
                        onChange={(event) => setDestinationRole(event.target.value)}
                    >
                        {roles.map(role => (
                            <option key={role.id || role.name} value={role.name}>
                                {role.name}{Number.isFinite(Number(role.candidate_count)) ? ` (${role.candidate_count} profiles)` : ''}
                            </option>
                        ))}
                    </select>
                    <button
                        className="btn btn-primary"
                        onClick={handleAddToRole}
                        disabled={isAddingToRole || !destinationRole}
                    >
                        {isAddingToRole ? <Loader2 size={16} className="animate-spin" /> : null}
                        <span>{isAddingToRole ? 'Adding...' : 'Add Selected to Role'}</span>
                    </button>
                    <button className="btn btn-secondary" onClick={clearSelections} disabled={isAddingToRole}>
                        Cancel
                    </button>
                </div>
            )}
        </div>
    )
}

export default Screening
