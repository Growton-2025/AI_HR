import { useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import SearchResults from '../components/SearchResults'
import RoleAssignment from '../components/RoleAssignment'
import { Search, SearchCode, X, Cpu, Database, Check, Loader2, SearchX, AlertCircle, Activity } from 'lucide-react'

function Screening() {
    const {
        searchQuery,
        setSearchQuery,
        isSearching,
        searchProgress,
        statusMessage,
        searchResults,
        usage,
        searchCandidatesStream,
        stopSearch,
        clearSearch,
        selectedCandidates,
        screenStep,
        setScreenStep,
    } = useAppStore()

    const [inputQuery, setInputQuery] = useState(searchQuery)
    const [showHistory, setShowHistory] = useState(false)

    const handleSearch = () => {
        if (inputQuery.trim()) {
            setSearchQuery(inputQuery)

            searchCandidatesStream(inputQuery)
            setShowHistory(false)
        }
    }

    const handleClear = () => {
        setInputQuery('')
        clearSearch()
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleSearch()
        }
    }

    const selectedCount = Object.keys(selectedCandidates).length

    const suggestions = [
        "Sales Manager in London with SaaS experience",
        "Python Developer with 3 years experience",
        "Marketing Director in New York",
        "Product Manager for Fintech"
    ]

    // Step 3: Role Assignment
    if (screenStep === 3) {
        return (
            <div style={{ width: '100%', position: 'relative', minHeight: '100vh' }}>
                <RoleAssignment />
            </div>
        )
    }

    // Step 2: Generating/Loading
    if (isSearching) {
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

        return (
            <div style={{ width: '100%', position: 'relative', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div className="loader-container">
                    {/* Pulsing Brain */}
                    <div className="ai-brain-pulse">
                        <Activity size={40} />
                    </div>

                    <h2 style={{ fontSize: '24px', fontWeight: '700', color: '#1e293b', marginBottom: '8px' }}>
                        AI Talent Scout
                    </h2>
                    <p style={{ color: '#64748b', marginBottom: '40px' }}>
                        Finding the perfect match for your role...
                    </p>

                    {/* Steps */}
                    <div className="loading-steps">
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
                        style={{ marginTop: '40px', fontSize: '13px', padding: '8px 16px' }}
                    >
                        Cancel Screening
                    </button>
                </div>
            </div>
        )
    }

    // Step 1: Search Interface + Results
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
                        onFocus={() => setShowHistory(true)}
                        onBlur={() => setTimeout(() => setShowHistory(false), 200)}
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
                            <SearchCode size={16} /> Screen Candidates
                        </button>
                    </div>
                </div>
            </div>

            {/* Show results if available */}
            {searchResults.length > 0 && (
                <>
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
                </>
            )}

            {/* Status message when no results */}
            {/* Professional No Results State */}
            {!isSearching && searchResults.length === 0 && statusMessage && (
                <div style={{
                    marginTop: '32px',
                    padding: '40px',
                    background: '#fff5f5', /* Red-50 */
                    border: '1px solid #fed7d7', /* Red-200 */
                    borderRadius: '16px',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    textAlign: 'center',
                    boxShadow: '0 4px 6px -1px rgba(229, 62, 62, 0.05)'
                }}>
                    <div style={{
                        width: '64px',
                        height: '64px',
                        background: '#fff',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginBottom: '16px',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                        border: '1px solid #fed7d7'
                    }}>
                        <SearchX size={32} color="#e53e3e" />
                    </div>

                    <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#c53030', marginBottom: '8px' }}>
                        No Matches Found
                    </h3>

                    <p style={{ color: '#718096', maxWidth: '400px', marginBottom: '24px', lineHeight: '1.6' }}>
                        We couldn't find any candidates matching "<strong>{searchQuery}</strong>".
                        The criteria might be too specific.
                    </p>

                    <div style={{ textAlign: 'left', background: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #edf2f7', width: '100%', maxWidth: '450px' }}>
                        <div style={{ fontSize: '12px', fontWeight: '600', color: '#718096', textTransform: 'uppercase', marginBottom: '12px', letterSpacing: '0.5px' }}>
                            Suggestions
                        </div>
                        <ul style={{ margin: 0, paddingLeft: '20px', color: '#4a5568', fontSize: '14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            <li>Try using broader terms (e.g., "Developer" instead of "Senior Python Developer")</li>
                            <li>Check for spelling errors in your query</li>
                            <li>Remove specific location or experience constraints</li>
                        </ul>
                    </div>

                    <button
                        className="btn btn-secondary"
                        onClick={handleClear}
                        style={{ marginTop: '24px' }}
                    >
                        Clear Screening & Try Again
                    </button>
                </div>
            )}
        </div>
    )
}

export default Screening
