import { useState, useEffect } from 'react'
import { useAppStore } from '../store/useAppStore'
import { toast } from 'sonner'
import { useShallow } from 'zustand/react/shallow'

function RoleAssignment() {
    const {
        selectedCandidates,
        candidatePriorities,
        candidateFeedback,
        roles,
        fetchRoles,
        assignCandidatesToRole,
        setScreenStep,
        clearSelections
    } = useAppStore(useShallow((state) => ({
        selectedCandidates: state.selectedCandidates,
        candidatePriorities: state.candidatePriorities,
        candidateFeedback: state.candidateFeedback,
        roles: state.roles,
        fetchRoles: state.fetchRoles,
        assignCandidatesToRole: state.assignCandidatesToRole,
        setScreenStep: state.setScreenStep,
        clearSelections: state.clearSelections,
    })))

    const [selectedRoles, setSelectedRoles] = useState([])
    const [isAssigning, setIsAssigning] = useState(false)

    useEffect(() => {
        fetchRoles()
    }, [fetchRoles])

    const toggleRole = (roleName) => {
        setSelectedRoles(prev => {
            if (prev.includes(roleName)) {
                return prev.filter(r => r !== roleName)
            }
            return [...prev, roleName]
        })
    }

    const handleAssign = () => {
        if (selectedRoles.length === 0) {
            toast.error('Please select at least one role')
            return
        }

        setIsAssigning(true)
        const candidateIds = Object.keys(selectedCandidates).map(id => parseInt(id))

        // Trigger all assignments in parallel, no await
        selectedRoles.forEach(roleName => {
            const assignments = candidateIds.map(cid => ({
                candidate_id: cid,
                priority: candidatePriorities[cid] || '--',
                feedback: candidateFeedback[cid] || ''
            }))
            assignCandidatesToRole(roleName, assignments)
        })

        toast.success(`Assigning ${candidateIds.length} candidate(s) to ${selectedRoles.length} role(s)...`)

        // Instant transition back
        setTimeout(() => {
            clearSelections()
            setScreenStep(1)
            setIsAssigning(false)
        }, 300)
    }


    const selectedCount = Object.keys(selectedCandidates).length

    return (
        <>
            <h2 className="screen-header">Role Assignment</h2>

            <div className="result-banner">
                <div className="result-banner-title">{selectedCount} Candidate(s) Selected</div>
                <div className="result-banner-subtitle">Choose one or more roles to assign these candidates to</div>
            </div>

            <div className="section-label">Available Roles</div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '16px' }}>
                Select the roles for candidate assignment
            </p>

            {roles.map(role => {
                const isChecked = selectedRoles.includes(role.name)

                return (
                    <div key={role.name} style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                        <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => toggleRole(role.name)}
                        />
                        <div style={{
                            flex: 1,
                            background: 'white',
                            padding: '14px 18px',
                            borderRadius: '8px',
                            border: `1px solid ${isChecked ? '#1a73e8' : '#e8eaed'}`
                        }}>
                            <div style={{ fontWeight: 600, fontSize: '15px', color: '#1a1f2e', marginBottom: '4px' }}>
                                {role.name}
                            </div>
                            <div style={{ fontSize: '12px', color: '#5f6368' }}>
                                Candidates: <span style={{ color: '#7c3aed', fontWeight: 500 }}>{role.candidate_count}</span>
                            </div>
                        </div>
                    </div>
                )
            })}

            {roles.length === 0 && (
                <div className="alert alert-info">
                    No roles available. Create roles in the Roles page first.
                </div>
            )}

            <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: '24px 0' }} />

            <div className="button-row">
                <button
                    className="btn btn-primary"
                    onClick={handleAssign}
                    disabled={isAssigning || selectedRoles.length === 0}
                >
                    {isAssigning ? 'Assigning...' : 'Assign Candidates'}
                </button>
                <button
                    className="btn btn-secondary"
                    onClick={() => setScreenStep(1)}
                >
                    Back to Results
                </button>
            </div>
        </>
    )
}

export default RoleAssignment
