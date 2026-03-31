import { useState, useEffect, useCallback, useRef } from 'react'
import { useAppStore } from '../store/useAppStore'
import { Plus, Trash2, Folder, Linkedin, ArrowLeft, User, Loader2, Mail, Copy, Send, RefreshCcw } from 'lucide-react'
import { toast } from 'sonner'
import StatusDropdown from '../components/StatusDropdown'

function Roles() {
    const {
        roles,
        fetchRoles,
        createRole,
        deleteRole,
        viewingRole,
        fetchRoleDetails,
        clearViewingRole,
        openRole,
        // Global Outreach Cache
        outreachStatusCache,
        fetchOutreachStatus,
        triggerHeyReachOutreach,
        removeCandidateFromRole
    } = useAppStore()

    const [newRoleName, setNewRoleName] = useState('')
    const [expandedSummary, setExpandedSummary] = useState(null)

    // Derived state for instant access
    const outreachStatus = (viewingRole?.id && outreachStatusCache[viewingRole.id]) ? outreachStatusCache[viewingRole.id] : {}

    const [isSendingOutreach, setIsSendingOutreach] = useState(false)
    const [isSendingLI, setIsSendingLI] = useState(false)
    const [isSyncing, setIsSyncing] = useState(false)
    const [isRefreshing, setIsRefreshing] = useState(false)
    const [isLoadingRole, setIsLoadingRole] = useState(false)
    const { heyreachCampaignId, setHeyreachCampaignId, lookupHeyReachCampaign } = useAppStore()


    // Initial Load
    useEffect(() => {
        fetchRoles()
    }, [fetchRoles])

    // No auto-refresh - user controls when to refresh via manual button

    // Show loading when opening a new role (only if not cached)
    useEffect(() => {
        if (viewingRole) {
            // If candidates array doesn't exist yet, we're loading
            if (!viewingRole.candidates) {
                setIsLoadingRole(true)
            } else {
                setIsLoadingRole(false)
            }
        } else {
            setIsLoadingRole(false)
        }
    }, [viewingRole])

    // Silent sync function with useCallback to prevent stale closures
    const silentSync = useCallback(async () => {
        if (!viewingRole?.id) return
        try {
            const token = localStorage.getItem('token')
            const response = await fetch(`/api/outreach/sync-responses/${viewingRole.id}`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            })
            if (response.ok) {
                fetchOutreachStatus(viewingRole.id)
            }
        } catch (error) {
            console.error("Silent sync failed", error)
        }
    }, [viewingRole?.id, fetchOutreachStatus])

    // Fetch outreach status when viewing role
    useEffect(() => {
        if (viewingRole?.id) {
            fetchOutreachStatus(viewingRole.id)
        }
    }, [viewingRole?.id])

    // Local fetchOutreachStatus removed in favor of global store action

    const handleManualRefresh = async () => {
        if (!viewingRole?.name) return
        setIsRefreshing(true)
        try {
            await fetchRoleDetails(viewingRole.name)
            await fetchOutreachStatus(viewingRole.id)
            toast.success('Refreshed role data')
        } catch (error) {
            toast.error('Failed to refresh')
        } finally {
            setIsRefreshing(false)
        }
    }

    const handleLookupHrCampaign = async () => {
        if (!viewingRole?.name) return
        const res = await lookupHeyReachCampaign(viewingRole.name)
        if (res.success) {
            toast.success(`Found campaign ID: ${res.campaign_id}`)
        } else {
            toast.error(res.error || 'No matching campaign found in HeyReach')
        }
    }

    const handleSendOutreach = async () => {

        if (!viewingRole?.candidates || viewingRole.candidates.length === 0) {
            toast.error('No candidates to send outreach to')
            return
        }

        setIsSendingOutreach(true)
        try {
            const token = localStorage.getItem('token')

            // Filter invalid candidates just in case
            const validCandidates = viewingRole.candidates.filter(c => c && c.id)
            if (validCandidates.length === 0) {
                toast.error('No valid candidates found to contact')
                setIsSendingOutreach(false)
                return
            }

            const candidateIds = validCandidates.map(c => parseInt(c.id)).filter(id => !isNaN(id))
            const roleId = parseInt(viewingRole.id)

            if (isNaN(roleId)) {
                toast.error('Invalid Role ID')
                setIsSendingOutreach(false)
                return
            }

            const response = await fetch('/api/outreach/trigger', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    candidate_ids: candidateIds.map(id => parseInt(id)),
                    role_id: parseInt(viewingRole.id),
                    role_name: viewingRole.name
                })
            })

            if (response.ok) {
                const data = await response.json()
                toast.success(`Campaign created! Sending to ${data.candidates_count} candidates`)
                // Refresh status
                const roleId = parseInt(viewingRole.id)
                setTimeout(() => fetchOutreachStatus(roleId), 1000)
            } else {
                const error = await response.json()
                toast.error(error.detail || 'Failed to send outreach')
            }
        } catch (error) {
            toast.error('Failed to send outreach')
            console.error(error)
        } finally {
            setIsSendingOutreach(false)
        }
    }

    const handleSendLinkedIn = async () => {
        if (!viewingRole?.candidates || viewingRole.candidates.length === 0) {
            toast.error('No candidates to send LinkedIn outreach to')
            return
        }

        const campaignId = parseInt(heyreachCampaignId, 10)
        if (isNaN(campaignId) || campaignId <= 0) {
            toast.error('Enter a valid HeyReach campaign ID')
            return
        }

        setIsSendingLI(true)
        try {
            const validCandidates = viewingRole.candidates.filter(c => c && c.id && c.linkedin)
            if (validCandidates.length === 0) {
                toast.error('No candidates with LinkedIn profiles found')
                setIsSendingLI(false)
                return
            }

            const candidateIds = validCandidates.map(c => parseInt(c.id))
            const roleId = parseInt(viewingRole.id)

            const res = await triggerHeyReachOutreach({
                candidate_ids: candidateIds,
                role_id: roleId,
                role_name: viewingRole.name,
                campaign_id: campaignId,
                sender_account_id: 113572 // From user snippet
            })



            if (res.success) {
                toast.success(`LinkedIn Outreach triggered for ${res.data.success_count} candidates`)
                setTimeout(() => fetchOutreachStatus(roleId), 1000)
            } else {
                toast.error(res.error || 'Failed to trigger LinkedIn outreach')
            }
        } catch (error) {
            toast.error('Failed to send LinkedIn outreach')
            console.error(error)
        } finally {
            setIsSendingLI(false)
        }
    }


    const handleSyncResponses = async () => {
        if (!viewingRole?.id) return

        setIsSyncing(true)
        try {
            const token = localStorage.getItem('token')
            const response = await fetch(`/api/outreach/sync-responses/${viewingRole.id}`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            })

            if (response.ok) {
                const data = await response.json()
                if (data.updated_count > 0) {
                    toast.success(`Synced ${data.updated_count} new responses`)
                } else {
                    toast.info('No new responses found')
                }
                fetchOutreachStatus(viewingRole.id)
            } else {
                toast.error('Failed to sync responses')
            }
        } catch (error) {
            toast.error('Failed to sync responses')
            console.error(error)
        } finally {
            setIsSyncing(false)
        }
    }

    const handleCopy = (text) => {
        if (!text) return
        navigator.clipboard.writeText(text)
        toast.success('Copied to clipboard')
    }

    const handleCreateRole = () => {
        if (!newRoleName.trim()) return
        const name = newRoleName.trim()
        setNewRoleName('')

        // Instant Toast
        toast.success(`Role "${name}" created`, { duration: 1000 })

        // Optimistic UI handled by store
        createRole(name).then(res => {
            if (!res.success) {
                setNewRoleName(name)
                toast.error(res.error || 'Failed to create role')
            }
        })
    }

    const handleDeleteRole = (roleName) => {
        if (!window.confirm(`Are you sure you want to delete "${roleName}"?`)) return

        // Instant Toast
        toast.success(`Role "${roleName}" removed`, { duration: 1000 })

        deleteRole(roleName).then(res => {
            if (!res.success) {
                toast.error(res.error || 'Failed to delete role')
            }
        })
    }

    const handleRemoveCandidate = (candidateId, candidateName) => {
        if (!window.confirm(`Are you sure you want to remove ${candidateName} from this role?`)) return

        toast.success(`${candidateName} removed from role`, { duration: 1000 })
        removeCandidateFromRole(viewingRole.name, candidateId).then(res => {
            if (!res.success) {
                toast.error(res.error || 'Failed to remove candidate')
            }
        })
    }

    const getPriorityBadge = (priority) => {
        if (!priority || priority === '--') return null
        const colors = {
            'High': '#ea4335',
            'Medium': '#fbbc04',
            'Low': '#34a853'
        }
        return (
            <span style={{ color: colors[priority], fontWeight: 600 }}>{priority}</span>
        )
    }

    const getStatusBadge = (status) => {
        const styles = {
            'in_campaign': { bg: '#f5f3ff', border: '#ddd6fe', color: '#6d28d9', text: 'In Campaign' },
            'sent': { bg: '#eff6ff', border: '#bfdbfe', color: '#1d4ed8', text: 'Sent' },
            'replied': { bg: '#f0fdf4', border: '#bbf7d0', color: '#15803d', text: 'Replied' },
            'bounced': { bg: '#fef2f2', border: '#fecaca', color: '#b91c1c', text: 'Bounced' },
            'pending': { bg: '#f8fafc', border: '#e2e8f0', color: '#64748b', text: 'Pending' }

        }
        const style = styles[status] || { bg: '#f8fafc', border: '#e2e8f0', color: '#64748b', text: 'Unknown' }
        return (
            <span style={{
                padding: '4px 10px',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: 600,
                background: style.bg,
                border: `1px solid ${style.border}`,
                color: style.color,
                display: 'inline-flex',
                alignItems: 'center',
                whiteSpace: 'nowrap'
            }}>
                {style.text}
            </span>
        )
    }



    const getLiStatusBadge = (status, sentCount = 0) => {
        const styles = {
            'in_campaign': { bg: '#f5f3ff', border: '#ddd6fe', color: '#6d28d9', text: 'In Campaign' },
            'connection_sent': { bg: '#f5f3ff', border: '#ddd6fe', color: '#6d28d9', text: 'Request Sent' },
            'connection_accepted': { bg: '#f0fdf4', border: '#bbf7d0', color: '#15803d', text: 'Connected' },
            'message_sent': { bg: '#fdf4ff', border: '#f5d0fe', color: '#a21caf', text: 'Message Sent' },
            'replied': { bg: '#fff7ed', border: '#ffedd5', color: '#c2410c', text: 'Replied' }

        }
        const style = styles[status] || { bg: '#f8fafc', border: '#e2e8f0', color: '#64748b', text: '--' }
        return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{
                    padding: '4px 10px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: 600,
                    background: style.bg,
                    border: `1px solid ${style.border}`,
                    color: style.color,
                    display: 'inline-flex',
                    alignItems: 'center',
                    whiteSpace: 'nowrap'
                }}>
                    {style.text}
                </span>
                {sentCount > 0 && <span style={{ fontSize: '10px', color: '#64748b', fontWeight: 500 }}>{sentCount} message(s) sent</span>}
            </div>
        )
    }



    const [chattingWith, setChattingWith] = useState(null)
    const [chatPlatform, setChatPlatform] = useState('email') // 'email' or 'linkedin'
    const [chatMessages, setChatMessages] = useState([])
    const [isFetchingChat, setIsFetchingChat] = useState(false)
    const [replyMessage, setReplyMessage] = useState('')
    const [isSendingReply, setIsSendingReply] = useState(false)
    const chatEndRef = useRef(null)

    useEffect(() => {
        if (chatEndRef.current) {
            chatEndRef.current.scrollIntoView({ behavior: 'smooth' })
        }
    }, [chatMessages])

    const { fetchChatHistory, sendChatReply } = useAppStore()

    const handleOpenChat = async (candidate, platform = 'email') => {
        setChattingWith(candidate)
        setChatPlatform(platform)
        setChatMessages([])
        setIsFetchingChat(true)
        try {
            const res = await fetchChatHistory(viewingRole.id, candidate.id, platform)
            if (res.success) {
                // Messages are already chronological (oldest first) from backend sync/fetch logic
                setChatMessages(res.messages)
            } else {
                toast.error(res.error || `Failed to load ${platform} chat history`)
            }
        } catch (error) {
            toast.error(`Failed to fetch ${platform} chat`)
        } finally {
            setIsFetchingChat(false)
        }
    }

    const handleSendReply = async () => {
        if (!replyMessage.trim() || isSendingReply) return
        setIsSendingReply(true)
        try {
            const res = await sendChatReply(viewingRole.id, chattingWith.id, replyMessage, chatPlatform)
            if (res.success) {
                toast.success('Reply sent!')
                setReplyMessage('')
                // Refresh chat
                handleOpenChat(chattingWith, chatPlatform)
            } else {
                toast.error(res.error || 'Failed to send reply')
            }
        } catch (error) {
            toast.error('Failed to send reply')
        } finally {
            setIsSendingReply(false)
        }
    }

    // Role Detail View
    if (viewingRole) {
        return (
            <div className="roles-page" style={{ width: '100%', position: 'relative', minHeight: '100vh', animation: 'fadeIn 0.2s ease-out' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
                    <button className="btn btn-secondary" onClick={clearViewingRole}>
                        <ArrowLeft size={16} /> Back to Roles
                    </button>
                    <h2 style={{ margin: 0, fontSize: '24px', fontWeight: 700, color: '#1e293b' }}>{viewingRole.name}</h2>
                </div>

                <div className="result-banner">
                    <div className="result-banner-title">
                        {viewingRole.candidates?.length || 0} Shortlisted Candidate(s)
                    </div>
                </div>

                {/* Optimized Action Bar */}
                <div style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    alignItems: 'stretch',
                    gap: '12px',
                    marginBottom: '20px'
                }}>
                    {/* EMAIL OUTREACH SECTION */}
                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                        padding: '12px',
                        background: '#ffffff',
                        borderRadius: '10px',
                        border: '1px solid #e2e8f0',
                        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                        flex: '0 0 auto'
                    }}>
                        <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Email Outreach</span>
                        <button
                            className="btn btn-primary btn-sm"
                            onClick={handleSendOutreach}
                            disabled={isSendingOutreach || !viewingRole.candidates?.length}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                padding: '8px 20px',
                                height: '36px'
                            }}
                        >
                            <Mail size={14} /> {isSendingOutreach ? 'Processing...' : 'Send Outreach'}
                        </button>
                    </div>

                    {/* LINKEDIN OUTREACH SECTION */}
                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                        padding: '12px',
                        background: '#f0f7ff',
                        borderRadius: '10px',
                        border: '1px solid #bae6fd',
                        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                        flex: '1 0 auto'
                    }}>
                        <span style={{ fontSize: '11px', fontWeight: 700, color: '#0369a1', textTransform: 'uppercase', letterSpacing: '0.05em' }}>LinkedIn Outreach (HeyReach)</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                background: '#ffffff',
                                padding: '4px 10px',
                                borderRadius: '6px',
                                border: '1px solid #cbd5e1'
                            }}>
                                <span style={{ fontSize: '11px', fontWeight: 600, color: '#475569' }}>Campaign ID:</span>
                                <input
                                    type="text"
                                    value={heyreachCampaignId}
                                    onChange={(e) => setHeyreachCampaignId(e.target.value)}
                                    placeholder="ID"
                                    style={{
                                        padding: '2px 6px',
                                        border: 'none',
                                        fontSize: '13px',
                                        width: '60px',
                                        outline: 'none',
                                        fontWeight: 'bold',
                                        color: '#0369a1'
                                    }}
                                />
                                <button
                                    className="btn-link"
                                    onClick={handleLookupHrCampaign}
                                    title="Auto-find campaign by role name"
                                    style={{
                                        padding: '4px',
                                        color: '#0369a1',
                                        display: 'flex',
                                        alignItems: 'center',
                                        cursor: 'pointer',
                                        border: 'none',
                                        background: 'none'
                                    }}
                                >
                                    <RefreshCcw size={12} />
                                </button>
                            </div>

                            <button
                                className="btn btn-secondary btn-sm"
                                onClick={handleSendLinkedIn}
                                disabled={isSendingLI || !viewingRole.candidates?.length}
                                style={{
                                    background: '#0a66c2',
                                    color: 'white',
                                    border: 'none',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    padding: '8px 24px',
                                    height: '36px'
                                }}
                            >
                                <Linkedin size={14} /> {isSendingLI ? 'Sending...' : 'Send LinkedIn'}
                            </button>
                        </div>
                    </div>

                    {/* UTILITIES SECTION */}
                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                        padding: '12px',
                        background: '#ffffff',
                        borderRadius: '10px',
                        border: '1px solid #e2e8f0',
                        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                        flex: '0 0 auto'
                    }}>
                        <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Utilities</span>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <button
                                className="btn btn-secondary btn-sm"
                                onClick={handleSyncResponses}
                                disabled={isSyncing}
                                title="Sync replies from both platforms"
                                style={{ height: '36px', padding: '0 16px' }}
                            >
                                <RefreshCcw size={14} className={isSyncing ? 'animate-spin' : ''} />
                                {isSyncing ? 'Sync' : 'Sync Responses'}
                            </button>
                            <button
                                className="btn btn-secondary btn-sm"
                                onClick={handleManualRefresh}
                                disabled={isRefreshing}
                                title="Refresh role data"
                                style={{ height: '36px', padding: '0 12px' }}
                            >
                                <RefreshCcw size={14} className={isRefreshing ? 'animate-spin' : ''} />
                            </button>
                        </div>
                    </div>
                </div>




                <div className="role-table-note">
                    Use Delivery to track campaign stage. Use Open Hub to read full conversations and reply.
                </div>

                {isLoadingRole ? (
                    <div className="empty-state" style={{ marginTop: '24px' }}>
                        <Loader2 size={48} className="animate-spin" style={{ color: '#94a3b8', marginBottom: '16px' }} />
                        <p>Loading candidates...</p>
                    </div>
                ) : (!viewingRole.candidates || viewingRole.candidates.length === 0) ? (
                    <div className="empty-state" style={{ marginTop: '24px' }}>
                        <User size={48} style={{ color: '#e2e8f0', marginBottom: '16px' }} />
                        <p>No candidates assigned yet. Start by screening for talent!</p>
                    </div>
                ) : (
                    <div className="table-wrapper" style={{ maxHeight: '600px', overflowY: 'auto' }}>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th style={{ width: '30px', minWidth: '30px' }}>#</th>
                                    <th style={{ minWidth: '140px', maxWidth: '180px' }}>Candidate</th>
                                    <th style={{ minWidth: '150px', maxWidth: '180px' }}>Role</th>
                                    <th style={{ minWidth: '140px' }}>Email</th>
                                    <th style={{ minWidth: '100px' }}>Phone</th>
                                    <th style={{ minWidth: '120px', width: '120px' }}>Details</th>
                                    <th style={{ minWidth: '70px', width: '70px' }}>Priority</th>
                                    <th style={{ minWidth: '140px', maxWidth: '160px' }}>Feedback</th>
                                    <th style={{ minWidth: '150px' }}>Status</th>
                                    <th style={{ minWidth: '120px' }}><div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Mail size={14} /> Delivery</div></th>
                                    <th style={{ minWidth: '180px' }}><div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Mail size={14} /> Hub</div></th>
                                    <th style={{ minWidth: '120px' }}><div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Linkedin size={14} /> Delivery</div></th>
                                    <th style={{ minWidth: '180px' }}><div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Linkedin size={14} /> Hub</div></th>
                                    <th style={{ minWidth: '60px', width: '60px' }}>Actions</th>

                                </tr>
                            </thead>
                            <tbody>
                                {viewingRole.candidates.map((candidate, idx) => {
                                    const primaryRole = candidate.roles?.[0] || {}
                                    const summary = candidate.reasoning || candidate.summary || "N/A"
                                    const truncatedSummary = summary.length > 30 ? summary.substring(0, 30) + "..." : summary
                                    const feedback = candidate.feedback || ""
                                    const truncatedFeedback = feedback.length > 80 ? feedback.substring(0, 80) + "..." : feedback

                                    return (
                                        <tr key={candidate.id || idx}>
                                            <td>{idx + 1}</td>
                                            <td>
                                                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                                    <span style={{ fontWeight: 600, color: "#1e293b" }}>{candidate.name || "N/A"}</span>
                                                    {candidate.linkedin && (
                                                        <a href={candidate.linkedin} target="_blank" rel="noopener noreferrer" className="linkedin-link" title="LinkedIn">
                                                            <Linkedin size={12} />
                                                        </a>
                                                    )}
                                                </div>
                                            </td>
                                            <td>
                                                <div style={{ lineHeight: "1.2" }}>
                                                    <div>{primaryRole.title || "N/A"}</div>
                                                    <div style={{ fontSize: "11px", color: "#64748b" }}>{primaryRole.company || "N/A"}</div>
                                                </div>
                                            </td>
                                            <td>
                                                {candidate.email ? (
                                                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                                        <span className="contact-cell" title={candidate.email}>{candidate.email}</span>
                                                        <button className="icon-btn" onClick={() => handleCopy(candidate.email)} title="Copy Email">
                                                            <Copy size={12} />
                                                        </button>
                                                    </div>
                                                ) : <span className="empty-val">Not available</span>}
                                            </td>
                                            <td>
                                                {candidate.mobile_phone ? (
                                                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                                        <span className="contact-cell" title={candidate.mobile_phone}>{candidate.mobile_phone}</span>
                                                        <button className="icon-btn" onClick={() => handleCopy(candidate.mobile_phone)} title="Copy Phone">
                                                            <Copy size={12} />
                                                        </button>
                                                    </div>
                                                ) : <span className="empty-val">Not available</span>}
                                            </td>
                                            <td>
                                                <span className="summary-trigger" onClick={() => setExpandedSummary(candidate.id)}>
                                                    {truncatedSummary}
                                                </span>
                                            </td>
                                            <td>{getPriorityBadge(candidate.priority)}</td>
                                            <td title={feedback} style={{ fontSize: "13px", color: "#64748b", whiteSpace: "normal", lineHeight: "1.4" }}>
                                                {truncatedFeedback || "—"}
                                            </td>

                                            {/* Candidate Status */}
                                            <td>
                                                <StatusDropdown
                                                    status={candidate.status}
                                                    candidateId={candidate.id}
                                                    onUpdate={(id, newStatus) => {
                                                        useAppStore.setState(state => ({
                                                            viewingRole: {
                                                                ...state.viewingRole,
                                                                candidates: state.viewingRole.candidates.map(c =>
                                                                    c.id === id ? { ...c, status: newStatus } : c
                                                                )
                                                            }
                                                        }))
                                                    }}
                                                />
                                            </td>

                                            {/* Email Delivery */}
                                            <td>
                                                {outreachStatus[candidate.id] ?
                                                    getStatusBadge(outreachStatus[candidate.id].status) :
                                                    <span className="empty-val">Not started</span>
                                                }
                                            </td>

                                            {/* Email Hub */}
                                            <td>
                                                {outreachStatus[candidate.id] ? (
                                                    <div className="hub-cell">
                                                        <button className="hub-link" onClick={() => handleOpenChat(candidate, 'email')}>
                                                            <Mail size={12} /> Open Hub
                                                        </button>
                                                        {outreachStatus[candidate.id].response_text ? (
                                                            <span className="hub-preview" title={outreachStatus[candidate.id].response_text}>
                                                                {outreachStatus[candidate.id].response_text}
                                                            </span>
                                                        ) : (
                                                            <span className="hub-muted">No reply yet</span>
                                                        )}
                                                    </div>
                                                ) : <span className="empty-val">Not started</span>}
                                            </td>

                                            {/* LinkedIn Status */}
                                            <td>
                                                {outreachStatus[candidate.id] ?
                                                    getLiStatusBadge(outreachStatus[candidate.id].li_status, outreachStatus[candidate.id].li_sent_count) :
                                                    <span className="empty-val">Not started</span>
                                                }
                                            </td>

                                            {/* LinkedIn Hub */}
                                            <td>
                                                {outreachStatus[candidate.id]?.li_status ? (
                                                    <div className="hub-cell">
                                                        <button className="hub-link hub-link-linkedin" onClick={() => handleOpenChat(candidate, 'linkedin')}>
                                                            <Linkedin size={12} /> Open Hub
                                                        </button>
                                                        {outreachStatus[candidate.id].li_response_text ? (
                                                            <span className="hub-preview" title={outreachStatus[candidate.id].li_response_text}>
                                                                {outreachStatus[candidate.id].li_response_text}
                                                            </span>
                                                        ) : (
                                                            <span className="hub-muted">No LinkedIn reply yet</span>
                                                        )}
                                                        {outreachStatus[candidate.id].li_response_received_at && (
                                                            <span className="hub-time">
                                                                {new Date(outreachStatus[candidate.id].li_response_received_at).toLocaleDateString()}
                                                            </span>
                                                        )}
                                                    </div>
                                                ) : <span className="empty-val">Not started</span>}
                                            </td>

                                            {/* Actions */}
                                            <td style={{ textAlign: 'center' }}>
                                                <button
                                                    className="icon-btn"
                                                    onClick={(e) => { e.stopPropagation(); handleRemoveCandidate(candidate.id, candidate.name); }}
                                                    title="Remove from role"
                                                    style={{ color: '#ef4444' }}
                                                    onMouseEnter={e => { e.currentTarget.style.background = '#fee2e2'; }}
                                                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                                                >
                                                    <Trash2 size={16} />
                                                </button>
                                            </td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>
                )}

                {/* Modals */}
                {expandedSummary && (
                    <div className="modal-overlay" onClick={() => setExpandedSummary(null)}>
                        <div className="modal-content" onClick={e => e.stopPropagation()}>
                            <h3 className="modal-title">{typeof expandedSummary === 'object' ? expandedSummary.type : 'Matching Analysis'}</h3>
                            <div className="modal-scroll-area">
                                {typeof expandedSummary === 'object' ?
                                    expandedSummary.content :
                                    (viewingRole.candidates.find(c => c.id === expandedSummary)?.reasoning ||
                                        viewingRole.candidates.find(c => c.id === expandedSummary)?.summary || 'N/A')
                                }
                            </div>
                            <div className="modal-footer">
                                <button className="btn btn-secondary" onClick={() => setExpandedSummary(null)}>Close</button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Chat Modal */}
                {chattingWith && (
                    <div className="modal-overlay" onClick={() => setChattingWith(null)}>
                        <div className="modal-content" style={{ maxWidth: '700px', display: 'flex', flexDirection: 'column', maxHeight: '90vh' }} onClick={e => e.stopPropagation()}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    {chatPlatform === 'email' ? <Mail size={24} style={{ color: '#7c3aed' }} /> : <Linkedin size={24} style={{ color: '#0284c7' }} />}
                                    <h3 className="modal-title" style={{ margin: 0 }}>
                                        {chatPlatform === 'email' ? 'Email Hub' : 'LinkedIn Hub'} with {chattingWith.name}
                                    </h3>
                                </div>
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    <button
                                        className="icon-btn"
                                        onClick={() => handleOpenChat(chattingWith, chatPlatform)}
                                        disabled={isFetchingChat}
                                        title="Refresh conversation"
                                    >
                                        <RefreshCcw size={16} className={isFetchingChat ? 'animate-spin' : ''} />
                                    </button>
                                    <button className="icon-btn" onClick={() => setChattingWith(null)}><Plus size={16} style={{ transform: 'rotate(45deg)' }} /></button>
                                </div>
                            </div>

                            <div className="chat-area" style={{
                                flex: 1,
                                overflowY: 'auto',
                                padding: '16px',
                                background: '#f8fafc',
                                borderRadius: '12px',
                                border: '1px solid #e2e8f0',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '12px',
                                minHeight: '300px'
                            }}>
                                {isFetchingChat ? (
                                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
                                        <Loader2 size={32} className="animate-spin" style={{ color: '#94a3b8' }} />
                                    </div>
                                ) : chatMessages.length === 0 ? (
                                    <div style={{ textAlign: 'center', color: '#94a3b8', marginTop: '40px' }}>No messages found.</div>
                                ) : (
                                    chatMessages.map((msg, i) => {
                                        const type = (msg.type || '').toUpperCase();
                                        const isIncoming = ['INBOX', 'REPLY', 'REPLIED', 'LEAD', 'INCOMING'].includes(type);
                                        const isOutgoing = !isIncoming;

                                        let body = msg.email_body || msg.text || msg.message || msg.content || msg.html_body || msg.body || '';

                                        if (isIncoming) {
                                            const quotedIndex = body.search(/On\s+.*,\s+.*wrote:/i);
                                            if (quotedIndex !== -1) body = body.substring(0, quotedIndex);
                                            const originalMsgIndex = body.search(/---+\s*Original\s*Message\s*---+/i);
                                            if (originalMsgIndex !== -1) body = body.substring(0, originalMsgIndex);
                                            const signatureMarkers = ["Best regards", "Regards", "Thanks,", "Thank you,", "Kind regards", "Sincerely"];
                                            for (const marker of signatureMarkers) {
                                                const sigIndex = body.lastIndexOf(marker);
                                                if (sigIndex > body.length / 2) {
                                                    body = body.substring(0, sigIndex);
                                                    break;
                                                }
                                            }
                                            body = body.trim();
                                        }

                                        return (
                                            <div key={i} style={{
                                                alignSelf: isOutgoing ? 'flex-end' : 'flex-start',
                                                maxWidth: '85%',
                                                padding: '12px 16px',
                                                borderRadius: isOutgoing ? '16px 16px 2px 16px' : '16px 16px 16px 2px',
                                                background: isOutgoing ? '#7c3aed' : '#ffffff',
                                                color: isOutgoing ? 'white' : '#1e293b',
                                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                                border: isOutgoing ? 'none' : '1px solid #e2e8f0'
                                            }}>
                                                <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '4px', fontWeight: 600 }}>
                                                    {isOutgoing ? 'Me' : (msg.sender_name || chattingWith.name)} • {msg.time ? new Date(msg.time).toLocaleString() : ''}
                                                </div>
                                                <div style={{ fontSize: '14px', lineHeight: '1.5', whiteSpace: 'pre-wrap' }} dangerouslySetInnerHTML={{ __html: body }} />
                                            </div>
                                        );
                                    })
                                )}
                                <div ref={chatEndRef} />
                            </div>

                            <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                <textarea
                                    className="input-field"
                                    placeholder="Write your reply..."
                                    style={{ width: '100%', minHeight: '80px', resize: 'vertical', padding: '12px' }}
                                    value={replyMessage}
                                    onChange={(e) => setReplyMessage(e.target.value)}
                                />
                                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
                                    <button className="btn btn-secondary" onClick={() => setChattingWith(null)}>Cancel</button>
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleSendReply}
                                        disabled={!replyMessage.trim() || isSendingReply}
                                        style={{ minWidth: '120px' }}
                                    >
                                        {isSendingReply ? <Loader2 size={16} className="animate-spin" /> : <><Send size={16} style={{ marginRight: '8px' }} /> Send Reply</>}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

            </div>
        )
    }

    // Role List View
    return (
        <div className="roles-page" style={{ width: '100%', position: 'relative', minHeight: '100vh', animation: 'fadeIn 0.2s ease-out' }}>
            <h2 className="screen-header">Role Management</h2>

            <div className="result-banner">
                <div className="result-banner-title">{roles.length} Active Role(s)</div>
                <div className="result-banner-subtitle">Organize and manage your top talent by role</div>
            </div>

            {/* Quick Add UI */}
            <div className="quick-add-container">
                <input
                    type="text"
                    className="input-field"
                    placeholder="New Role Name (e.g. Senior Sales Director)"
                    value={newRoleName}
                    onChange={(e) => setNewRoleName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCreateRole()}
                    style={{ flex: 1 }}
                />
                <button
                    className="btn btn-primary"
                    onClick={handleCreateRole}
                    disabled={!newRoleName.trim()}
                >
                    <Plus size={18} /> Add Role
                </button>
            </div>


            {/* Roles Grid */}
            <div className="roles-grid">
                {roles.length === 0 ? (
                    <div className="empty-state">
                        <Folder size={48} style={{ color: '#e2e8f0', marginBottom: '16px' }} />
                        <p>No recruitment roles created yet.</p>
                    </div>
                ) : (
                    roles.map(role => (
                        <div key={role.name} className="role-card">
                            <div className="role-card-main" onClick={() => openRole(role)}>
                                <div className="role-card-icon">
                                    <Folder size={20} />
                                </div>
                                <div className="role-card-content">
                                    <div className="role-card-name">{role.name}</div>
                                    <div className="role-card-meta">
                                        <User size={12} /> {role.candidate_count} assigned candidates
                                    </div>
                                </div>
                            </div>
                            <button className="role-delete-btn" onClick={() => handleDeleteRole(role.name)} title="Remove Role">
                                <Trash2 size={16} />
                            </button>
                        </div>
                    ))
                )}
            </div>
        </div>
    )
}

export default Roles
