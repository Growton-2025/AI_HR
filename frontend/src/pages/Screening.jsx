import { startTransition, useMemo, useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import SearchResults from '../components/SearchResults'
import RoleAssignment from '../components/RoleAssignment'
import { Search, SearchCode, X, Cpu, Database, Check, Loader2, SearchX, AlertCircle, Activity, Sparkles, RotateCcw } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'

function buildRetrySuggestions(query) {
    const trimmedQuery = (query || '').trim()
    if (!trimmedQuery) {
        return []
    }

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

function Screening() {
    const {
        searchQuery,
        setSearchQuery,
        isSearching,
        searchProgress,
        statusMessage,
        searchResults,
        usage,
        searchOutcome,
        lastSearchError,
        searchCandidatesStream,
        stopSearch,
        clearSearch,
        selectedCandidates,
        screenStep,
        setScreenStep,
    } = useAppStore(useShallow((state) => ({
        searchQuery: state.searchQuery,
        setSearchQuery: state.setSearchQuery,
        isSearching: state.isSearching,
        searchProgress: state.searchProgress,
        statusMessage: state.statusMessage,
        searchResults: state.searchResults,
        usage: state.usage,
        searchOutcome: state.searchOutcome,
        lastSearchError: state.lastSearchError,
        searchCandidatesStream: state.searchCandidatesStream,
        stopSearch: state.stopSearch,
        clearSearch: state.clearSearch,
        selectedCandidates: state.selectedCandidates,
        screenStep: state.screenStep,
        setScreenStep: state.setScreenStep,
    })))

    const [inputQuery, setInputQuery] = useState(searchQuery)
    const selectedCount = Object.keys(selectedCandidates).length

    const handleSearch = (queryOverride) => {
        const nextQuery = typeof queryOverride === 'string'
            ? queryOverride.trim()
            : inputQuery.trim()
        if (!nextQuery) {
            return
        }

        if (nextQuery !== inputQuery) {
            setInputQuery(nextQuery)
        }

        setSearchQuery(nextQuery)
        startTransition(() => {
            searchCandidatesStream(nextQuery)
        })
    }

    const handleClear = () => {
        setInputQuery('')
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

    const suggestions = [
        "Sales Manager in London with SaaS experience",
        "Python Developer with 3 years experience",
        "Marketing Director in New York",
        "Product Manager for Fintech"
    ]

    const retrySuggestions = useMemo(() => buildRetrySuggestions(searchQuery || inputQuery), [inputQuery, searchQuery])
    const hasResults = searchResults.length > 0
    const hasSettledSearch = hasResults || ['empty', 'error', 'cancelled'].includes(searchOutcome)
    const shouldShowStage = isSearching || !hasResults
    const steps = [
        { label: "Analyzing Requirements", icon: Search },
        { label: "Scanning Candidate Database", icon: Database },
        { label: "Evaluating Matches with AI", icon: Cpu },
        { label: "Finalizing Results", icon: Check }
    ]

    let currentStepIndex = 0
    if (searchProgress > 25) currentStepIndex = 1
    if (searchProgress > 50) currentStepIndex = 2
    if (searchProgress > 85) currentStepIndex = 3

    // Step 3: Role Assignment
    if (screenStep === 3) {
        return (
            <div style={{ width: '100%', position: 'relative', minHeight: '100vh' }}>
                <RoleAssignment />
            </div>
        )
    }

    return (
        <div style={{ width: '100%', position: 'relative', minHeight: '100vh', padding: '40px 40px 40px' }}>
            <h2 className="screen-header">Candidate Screening</h2>
            <p className="subtitle">
                Find your perfect candidate using natural language. Describe the role, skills, and requirements.
            </p>

            <div className="hero-search-container" style={{ position: 'relative', zIndex: 20 }}>
                <div style={{ position: 'relative' }}>
                    <textarea
                        className="search-textarea"
                        placeholder="Describe your ideal candidate..."
                        value={inputQuery}
                        onChange={(e) => setInputQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                    {inputQuery && (
                        <button
                            onClick={() => setInputQuery('')}
                            style={{
                                position: 'absolute',
                                right: '16px',
                                top: '16px',
                                background: 'none',
                                border: 'none',
                                cursor: 'pointer',
                                color: '#94a3b8'
                            }}
                        >
                            <X size={18} />
                        </button>
                    )}

                    {/* Recent History Dropdown */}

                </div>

                <div className="search-actions">
                    <div className="suggestion-chips">
                        {suggestions.map((s, i) => (
                            <div key={i} className="chip" onClick={() => setInputQuery(s)}>
                                {s}
                            </div>
                        ))}
                    </div>

                    <div style={{ display: 'flex', gap: '12px' }}>
                        <button className="btn btn-secondary" onClick={handleClear}>
                            Clear
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={handleSearch}
                            disabled={!inputQuery.trim()}
                        >
                            {isSearching ? <Loader2 size={16} className="animate-spin" /> : <SearchCode size={16} />}
                            {isSearching ? 'Screening...' : 'Screen Candidates'}
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
                                    <Activity size={40} />
                                </div>

                                <h2 style={{ fontSize: '24px', fontWeight: '700', color: '#1e293b', marginBottom: '8px' }}>
                                    AI Talent Scout
                                </h2>
                                <p style={{ color: '#64748b', marginBottom: '16px', textAlign: 'center', maxWidth: '520px' }}>
                                    Finding the strongest match for your role without freezing the screen.
                                </p>

                                <div className="screening-status-chip">
                                    <Sparkles size={14} />
                                    <span>{statusMessage || 'Screening candidates...'}</span>
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
                                            <div key={index} className={`step-item ${statusClass}`}>
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

                                <button
                                    className="btn btn-secondary"
                                    onClick={stopSearch}
                                    style={{ marginTop: '28px', fontSize: '13px', padding: '8px 16px' }}
                                >
                                    Cancel Screening
                                </button>
                            </div>
                        </div>
                    )}

                    {!isSearching && !hasSettledSearch && (
                        <div className="screening-stage-card screening-stage-idle">
                            <div className="screening-idle-header">
                                <div className="screening-idle-badge">
                                    <Sparkles size={14} />
                                    <span>Fast natural-language search</span>
                                </div>
                                <h3>Search by company, title, geography, product focus, or experience</h3>
                                <p>
                                    Screening works best when you write the request the way you would brief a recruiter.
                                    The page will keep the search box visible and transition results in underneath it.
                                </p>
                            </div>
                            <div className="screening-idle-grid">
                                {suggestions.map((suggestion) => (
                                    <button
                                        key={suggestion}
                                        type="button"
                                        className="screening-idle-chip"
                                        onClick={() => setInputQuery(suggestion)}
                                    >
                                        {suggestion}
                                    </button>
                                ))}
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
                                I couldn&apos;t find a strong match for <strong>{searchQuery}</strong>.
                                Try a broader phrase, or rerun one of these relaxed searches.
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

                            <div className="screening-empty-footer">
                                <button
                                    className="btn btn-secondary"
                                    onClick={handleClear}
                                >
                                    Clear Screening
                                </button>
                            </div>
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

            {hasResults && !isSearching && (
                <div className="screening-results-panel">
                    <div className="result-banner">
                        <div className="result-banner-content">
                            <div className="result-banner-title">{searchResults.length} Candidates Found</div>
                            <div className="result-banner-subtitle">
                                <strong>Query:</strong> {searchQuery}
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

                    <div style={{ height: '40px' }}></div>

                    <div className="section-label">Actions</div>
                    <div className="button-row" style={{ alignItems: 'center', background: 'white', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                        <button
                            className="btn btn-primary"
                            disabled={selectedCount === 0}
                            onClick={() => setScreenStep(3)}
                        >
                            Assign to Role
                        </button>
                        {selectedCount > 0 ? (
                            <span className="selection-badge">{selectedCount} candidate(s) selected</span>
                        ) : (
                            <span style={{ color: '#94a3b8', fontSize: '14px' }}>Select candidates from the list above</span>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}

export default Screening
