import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import axios from 'axios'
import { API_BASE, useAppStore } from '../store/useAppStore'
import { Plus, Trash2, Folder, Linkedin, ArrowLeft, User, Loader2, Mail, Copy, RefreshCcw, FileUp, Settings2, PowerOff, Filter, Briefcase, Building2, MapPin, BarChart2, X, ChevronDown, ChevronUp, MessageSquare, Phone, Check, Search, UserPlus, Play, Edit2, ExternalLink } from 'lucide-react'
import { toast } from 'sonner'
import StatusDropdown, { RECRUITMENT_STAGES, STATUS_STYLES } from '../components/StatusDropdown'
import { useShallow } from 'zustand/react/shallow'
import CsvMappingModal from '../components/CsvMappingModal'
import RoleEmailSendModal from '../components/RoleEmailSendModal'
import RoleCreateModal from '../components/RoleCreateModal'
import { TagFilterInput, SelectFilter, RangeSlider } from '../components/FilterComponents'
import CandidateConversationModal from '../components/CandidateConversationModal'
import EditableCandidateNotes from '../components/EditableCandidateNotes'
import AddToListModal from '../components/AddToListModal'
import AiColumnConfigModal from '../components/AiColumnConfigModal'
import AiColumnCellDrawer from '../components/AiColumnCellDrawer'
import HayasaBrand from '../components/HayasaBrand'
import { longOperationAxios } from '../api/longTimeoutAxios'
import { renderFriendlyAiValue, resolveAiCellValue, summarizeAiRun, createOptimisticAiColumn, mergeAiColumnDefinitions } from './TalentPool'

const RESPONDED_TAB = '__responded__'
const RESPONSE_TAB_LABEL = 'Responded'

const NEW_BADGE_DAYS = 7

function isRecentlyAdded(isoString) {
    if (!isoString) return false
    const added = new Date(isoString)
    if (Number.isNaN(added.getTime())) return false
    return (Date.now() - added.getTime()) < NEW_BADGE_DAYS * 86400000
}

function addedTooltip(isoString) {
    if (!isoString) return ''
    const added = new Date(isoString)
    if (Number.isNaN(added.getTime())) return ''
    const day = added.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })
    const time = added.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    return `Added on ${day} at ${time}`
}

function candidateResponseSnapshot(candidate = {}, outreach = {}, options = {}) {
    const countsReady = options.countsReady !== false
    const candidateCountsReady = countsReady || candidate.outreach_counts_loaded === true || candidate.outreach_counts_included === true
    const rawMessageCount = candidateCountsReady ? (
        outreach.message_count
        ?? outreach.total_message_count
        ?? outreach.li_message_count
        ?? outreach.email_message_count
        ?? candidate.message_count
        ?? ((Number(candidate.message_sent_count || 0) || 0) + (Number(candidate.li_sent_count || 0) || 0))
    ) : null
    const messageCount = rawMessageCount == null ? null : (Number(rawMessageCount) || 0)
    const responseText = (
        outreach.li_response_text
        || outreach.response_text
        || candidate.li_response_text
        || candidate.response_text
        || candidate.response
        || ''
    )
    const responseChannel = outreach.li_response_text || candidate.li_response_text
        ? 'LinkedIn'
        : (outreach.response_text || candidate.response_text || candidate.response ? 'Email' : '')
    return {
        responseText,
        responseChannel,
        messageCount,
        hasResponse: Boolean(String(responseText || '').trim()) || Number(messageCount || 0) > 0 || Boolean(outreach.li_conversation_id),
        hasUnread: Boolean(outreach.has_unread_response),
    }
}


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
        removeCandidateFromRole,
        invalidateTalentPoolCaches,
        rolesLastFetchedAt,
        fetchTalentPoolSummary,
        fetchAnalytics,
        deactivateRole,
        assignCandidatesToRole,
        user
    } = useAppStore(useShallow((state) => ({
        rolesLastFetchedAt: state.rolesLastFetchedAt,
        roles: state.roles,
        fetchRoles: state.fetchRoles,
        createRole: state.createRole,
        deleteRole: state.deleteRole,
        viewingRole: state.viewingRole,
        fetchRoleDetails: state.fetchRoleDetails,
        clearViewingRole: state.clearViewingRole,
        openRole: state.openRole,
        outreachStatusCache: state.outreachStatusCache,
        fetchOutreachStatus: state.fetchOutreachStatus,
        removeCandidateFromRole: state.removeCandidateFromRole,
        invalidateTalentPoolCaches: state.invalidateTalentPoolCaches,
        fetchTalentPoolSummary: state.fetchTalentPoolSummary,
        fetchAnalytics: state.fetchAnalytics,
        deactivateRole: state.deactivateRole,
        assignCandidatesToRole: state.assignCandidatesToRole,
        user: state.user
    })))

    const [createModalOpen, setCreateModalOpen] = useState(false)
    const [activationRole, setActivationRole] = useState(null)
    const [uploadRole, setUploadRole] = useState(null)
    const [uploadFile, setUploadFile] = useState(null)
    const [uploadHeaders, setUploadHeaders] = useState([])
    const [uploadMapping, setUploadMapping] = useState({})
    const [uploadMappingDetails, setUploadMappingDetails] = useState({})
    const [uploadRequiredTargets, setUploadRequiredTargets] = useState(['first_name', 'last_name', 'linkedin', 'city', 'title'])
    const [uploadTargetOptions, setUploadTargetOptions] = useState([])
    const [uploadPreviewBusy, setUploadPreviewBusy] = useState('')
    const [uploadCommitBusy, setUploadCommitBusy] = useState(false)
    const [uploadRowCount, setUploadRowCount] = useState(0)
    const [uploadProgress, setUploadProgress] = useState(null)
    const [uploadEnrichmentMode, setUploadEnrichmentMode] = useState('none')
    const uploadFileRef = useRef(null)

    // Selection State
    const [selectedIds, setSelectedIds] = useState(new Set())
    const [allFilteredSelected, setAllFilteredSelected] = useState(false)
    const [showAddToListModal, setShowAddToListModal] = useState(false)

    // Filtering State
    const [showFilters, setShowFilters] = useState(true)
    const [showOutreach, setShowOutreach] = useState(false)
    const [activeStatusTab, setActiveStatusTab] = useState('')
    const [roleSearch, setRoleSearch] = useState('')
    const [filters, setFilters] = useState({
        title: [], titleInput: '',
        company: [], companyInput: '',
        city: [], cityInput: '',
        product_service: [], productInput: '',
        status: [], statusInput: '',
        min_exp: 0, max_exp: 40,
        added_from: '', added_to: ''
    })
    const [roleFilteredCandidates, setRoleFilteredCandidates] = useState([])
    const [roleStatusCounts, setRoleStatusCounts] = useState({})
    const [roleFilterMeta, setRoleFilterMeta] = useState({
        titles: [], companies: [], cities: [], products: [], statuses: []
    })
    const [isFilteringRole, setIsFilteringRole] = useState(false)
    const roleFilterRequestSeqRef = useRef(0)
    const roleFilterDebounceRef = useRef(null)
    // Holds the role id whose saved filters have been loaded, so the persist
    // effect never overwrites storage with defaults before hydration runs.
    const roleFiltersHydratedRef = useRef(null)

    // ── URL ↔ open-role sync ─────────────────────────────────────────────
    // The open role lives in the URL (?role=<name>) so a refresh, browser
    // back/forward, or a shared link restores the exact same view.
    const [searchParams, setSearchParams] = useSearchParams()
    const roleParamName = searchParams.get('role') || ''
    const roleUrlInitRef = useRef(false)

    useEffect(() => {
        const currentViewing = useAppStore.getState().viewingRole
        if (!roleUrlInitRef.current) {
            roleUrlInitRef.current = true
            if (!roleParamName && currentViewing?.name) {
                // Mounted with a role already open (in-app navigation back to
                // /roles) — reflect it in the URL instead of closing it.
                setSearchParams(
                    currentViewing.id
                        ? { role: currentViewing.name, role_id: String(currentViewing.id) }
                        : { role: currentViewing.name },
                    { replace: true },
                )
                return
            }
        }
        if (roleParamName) {
            if (currentViewing?.name !== roleParamName) {
                const known = (useAppStore.getState().roles || []).find(r => r.name === roleParamName)
                // openRole handles unknown names: it fetches by name and
                // clears the view on 404 (deleted roles).
                openRole(known || { name: roleParamName })
            }
        } else if (currentViewing) {
            clearViewingRole()
        }
    }, [roleParamName, openRole, clearViewingRole, setSearchParams])

    const openRoleView = useCallback((role) => {
        openRole(role)
        setSearchParams(role?.id ? { role: role.name, role_id: String(role.id) } : { role: role?.name || '' })
    }, [openRole, setSearchParams])

    const closeRoleView = useCallback(() => {
        clearViewingRole()
        setSearchParams({})
    }, [clearViewingRole, setSearchParams])
    const [addCandidateOpen, setAddCandidateOpen] = useState(false)
    const [candidateSearch, setCandidateSearch] = useState('')
    const [candidateSearchResults, setCandidateSearchResults] = useState([])
    const [candidateSearchLoading, setCandidateSearchLoading] = useState(false)
    // 'talent_pool' searches the whole master pool; 'owner_pool' narrows to the role owner's uploads (admin only)
    const [addCandidateScope, setAddCandidateScope] = useState('talent_pool')
    const [selectedCandidateAdds, setSelectedCandidateAdds] = useState(new Set())
    const [isAddingCandidates, setIsAddingCandidates] = useState(false)
    const candidateSearchDebounceRef = useRef(null)
    const [outreachCountLoadedRoles, setOutreachCountLoadedRoles] = useState(() => new Set())

    const splitFilterValues = useCallback((values = [], _inputValue = '') => {
        // Tags are kept whole: suggestion values like "Bhubaneswar, Odisha, India"
        // contain commas and must not be split into separate filter terms.
        const list = Array.isArray(values) ? values : values == null || values === '' ? [] : [values]
        return [...list]
            .map(value => String(value || '').trim())
            .filter(Boolean)
    }, [])

    // Serialize filter values for the browse API. "||" is used when any value
    // contains a comma (the backend splits on "||" when present); a trailing
    // "||" marks a single comma-containing value as already-joined.
    const joinFilterParam = useCallback((values = []) => {
        if (!values.length) return ''
        if (!values.some(value => value.includes(','))) return values.join(',')
        const joined = values.join('||')
        return values.length === 1 ? `${joined}||` : joined
    }, [])

    const roleScopeParams = useCallback(() => {
        const params = new URLSearchParams()
        if (String(user?.role || '').toLowerCase() === 'admin' && viewingRole?.owner_user_id) {
            params.set('view_scope', 'recruiter_pools')
            params.set('recruiter_filter_id', viewingRole.owner_user_id)
        }
        if (viewingRole?.id) params.set('role_id', viewingRole.id)
        return params
    }, [user?.role, viewingRole?.id, viewingRole?.owner_user_id])

    const hasActiveRoleFilters = useMemo(() => (
        roleSearch.trim().length > 0
        || splitFilterValues(filters.title, filters.titleInput).length > 0
        || splitFilterValues(filters.company, filters.companyInput).length > 0
        || splitFilterValues(filters.city, filters.cityInput).length > 0
        || splitFilterValues(filters.product_service, filters.productInput).length > 0
        || splitFilterValues(filters.status, filters.statusInput).length > 0
        || Boolean(activeStatusTab)
        || Number(filters.min_exp || 0) > 0
        || Number(filters.max_exp ?? 40) < 40
        || Boolean(filters.added_from)
        || Boolean(filters.added_to)
    ), [activeStatusTab, filters, roleSearch, splitFilterValues])

    const hasServerRoleFilters = useMemo(() => (
        roleSearch.trim().length > 0
        || splitFilterValues(filters.title, filters.titleInput).length > 0
        || splitFilterValues(filters.company, filters.companyInput).length > 0
        || splitFilterValues(filters.city, filters.cityInput).length > 0
        || splitFilterValues(filters.product_service, filters.productInput).length > 0
        || splitFilterValues(filters.status, filters.statusInput).length > 0
        || Boolean(activeStatusTab && activeStatusTab !== RESPONDED_TAB)
        || Number(filters.min_exp || 0) > 0
        || Number(filters.max_exp ?? 40) < 40
        || Boolean(filters.added_from)
        || Boolean(filters.added_to)
    ), [activeStatusTab, filters, roleSearch, splitFilterValues])

    const setFilter = useCallback((key, val) => {
        setFilters(prev => {
            const next = { ...prev, [key]: val }
            if (key === 'status') {
                const nextValues = Array.isArray(val) ? val : (val ? [val] : [])
                setActiveStatusTab(nextValues.length === 1 ? nextValues[0] : '')
            }
            return next
        })
    }, [])

    const clearFilters = useCallback(() => {
        setFilters({
            title: [], titleInput: '',
            company: [], companyInput: '',
            city: [], cityInput: '',
            product_service: [], productInput: '',
            status: [], statusInput: '',
            min_exp: 0, max_exp: 40,
            added_from: '', added_to: ''
        })
        setActiveStatusTab('')
        setRoleSearch('')
        try {
            if (viewingRole?.id) window.localStorage.removeItem(`role_filters_v1_${viewingRole.id}`)
        } catch { /* storage unavailable */ }
        const baseCandidates = viewingRole?.candidates || []
        setRoleFilteredCandidates(baseCandidates)
        setRoleStatusCounts(baseCandidates.reduce((counts, candidate) => {
            const status = candidate.status || 'To be started'
            counts[status] = (counts[status] || 0) + 1
            return counts
        }, {}))
    }, [viewingRole?.candidates, viewingRole?.id])

    const statusCounts = roleStatusCounts
    const roleStatusOptions = useMemo(
        () => [...new Set([...(roleFilterMeta.statuses || []), ...RECRUITMENT_STAGES])],
        [roleFilterMeta.statuses]
    )
    const roleFilteredTotal = useMemo(
        () => Object.values(statusCounts).reduce((total, count) => total + Number(count || 0), 0),
        [statusCounts]
    )

    // Derived state for instant access
    const outreachStatus = (viewingRole?.id && outreachStatusCache[viewingRole.id]) ? outreachStatusCache[viewingRole.id] : {}
    const profileOutreachCountsLoaded = Array.isArray(viewingRole?.candidates)
        && viewingRole.candidates.length > 0
        && viewingRole.candidates.every(candidate => candidate.outreach_counts_loaded === true || candidate.outreach_counts_included === true)
    const outreachCountsLoaded = Boolean(
        (viewingRole?.id && outreachCountLoadedRoles.has(String(viewingRole.id)))
        || profileOutreachCountsLoaded
    )
    const filteredCandidates = useMemo(() => (
        activeStatusTab === RESPONDED_TAB
            ? roleFilteredCandidates.filter(candidate => candidateResponseSnapshot(candidate, outreachStatus[candidate.id] || {}, { countsReady: outreachCountsLoaded }).hasResponse)
            : roleFilteredCandidates
    ), [activeStatusTab, outreachCountsLoaded, outreachStatus, roleFilteredCandidates])
    const respondedCount = useMemo(() => (
        outreachCountsLoaded
            ? roleFilteredCandidates.filter(candidate => candidateResponseSnapshot(candidate, outreachStatus[candidate.id] || {}, { countsReady: true }).hasResponse).length
            : null
    ), [outreachCountsLoaded, outreachStatus, roleFilteredCandidates])
    const visibleStatusTabs = useMemo(() => (
        roleStatusOptions.filter(status => Number(statusCounts[status] || 0) > 0 || RECRUITMENT_STAGES.includes(status))
    ), [roleStatusOptions, statusCounts])

    // ── Smart Columns (role-scoped AI columns, mirrors the Talent Pool feature) ──
    const [roleAiColumns, setRoleAiColumns] = useState([])
    const [aiColumnModal, setAiColumnModal] = useState(null)
    const [aiCellDrawer, setAiCellDrawer] = useState({ open: false, loading: false, detail: null, title: '' })
    const roleAiInFlightKeyRef = useRef('')
    const roleAiRequestSeqRef = useRef(0)

    const roleViewScope = viewingRole?.id ? `role_${viewingRole.id}` : ''
    const roleAiCandidateIdsKey = useMemo(
        () => filteredCandidates.map(candidate => candidate.id).filter(Boolean).join(','),
        [filteredCandidates]
    )

    const fetchRoleAiColumns = useCallback(async () => {
        if (!roleViewScope || !roleAiCandidateIdsKey) return
        const params = new URLSearchParams()
        params.set('candidate_ids', roleAiCandidateIdsKey)
        params.set('view_scope', roleViewScope)
        params.set('role_id', roleViewScope.replace('role_', ''))
        const requestKey = params.toString()
        if (roleAiInFlightKeyRef.current === requestKey) return
        const requestId = ++roleAiRequestSeqRef.current
        roleAiInFlightKeyRef.current = requestKey
        try {
            const res = await longOperationAxios.get(`${API_BASE}/ai-columns?${requestKey}`, { timeout: 120000 })
            if (requestId !== roleAiRequestSeqRef.current) return
            setRoleAiColumns(prev => mergeAiColumnDefinitions(prev || [], res.data?.columns || []))
        } catch (error) {
            console.error('Failed to fetch role smart columns', error)
        } finally {
            if (roleAiInFlightKeyRef.current === requestKey) roleAiInFlightKeyRef.current = ''
        }
    }, [roleViewScope, roleAiCandidateIdsKey])

    useEffect(() => { setRoleAiColumns([]) }, [roleViewScope])

    useEffect(() => {
        const timer = window.setTimeout(() => { void fetchRoleAiColumns() }, 400)
        return () => window.clearTimeout(timer)
    }, [fetchRoleAiColumns])

    useEffect(() => {
        const hasActiveRuns = (roleAiColumns || []).some(column => {
            const status = column.latest_run?.status
            if (status === 'queued' || status === 'running') return true
            return Object.values(column.cells_by_candidate || {}).some(cell => cell.status === 'queued' || cell.status === 'running')
        })
        if (!hasActiveRuns) return undefined
        const timer = window.setInterval(() => { void fetchRoleAiColumns() }, 2000)
        return () => window.clearInterval(timer)
    }, [roleAiColumns, fetchRoleAiColumns])

    const dynamicRoleAiCols = useMemo(() => (
        (roleAiColumns || []).flatMap(definition => {
            const outputs = Array.isArray(definition.output_schema) && definition.output_schema.length
                ? definition.output_schema
                : [{ key: 'result', label: 'Result', primary: true }]
            return outputs.map(output => ({
                key: `__ai__${definition.id}__${output.key}`,
                label: outputs.length === 1 ? definition.name : `${definition.name} · ${output.label}`,
                definitionId: definition.id,
                definition,
                outputKey: output.key,
                isPrimaryOutput: Boolean(output.primary),
            }))
        })
    ), [roleAiColumns])

    const upsertOptimisticRoleAiColumn = useCallback((runInfo = {}) => {
        const candidateIdArray = Array.isArray(runInfo.candidateIds)
            ? runInfo.candidateIds.map(Number).filter(Number.isFinite)
            : []
        const rawDefinition = runInfo.definition || {}
        const definition = {
            ...rawDefinition,
            id: rawDefinition.id ?? runInfo.columnDefinitionId,
            name: rawDefinition.name || runInfo.columnName || 'Smart Column',
        }
        if (!definition.id || !candidateIdArray.length) return
        const optimisticColumn = createOptimisticAiColumn(definition, candidateIdArray, runInfo.runId || null)
        setRoleAiColumns(prev => {
            const existingIndex = (prev || []).findIndex(col => String(col.id) === String(definition.id))
            if (existingIndex === -1) return [...(prev || []), optimisticColumn]
            return (prev || []).map((col, index) => (
                index === existingIndex ? mergeAiColumnDefinitions([col], [optimisticColumn])[0] : col
            ))
        })
    }, [])

    const revertOptimisticRoleAiColumn = useCallback((runInfo = {}) => {
        if (!runInfo?.columnDefinitionId) return
        if (runInfo.isEditMode) {
            void fetchRoleAiColumns()
            return
        }
        setRoleAiColumns(prev => (prev || []).filter(col => (
            String(col.id) !== String(runInfo.columnDefinitionId) || !col.__optimistic
        )))
    }, [fetchRoleAiColumns])

    const attachRoleAiColumnRun = useCallback((runInfo = {}) => {
        if (!runInfo?.columnDefinitionId || !runInfo?.runId) return
        setRoleAiColumns(prev => (prev || []).map(col => {
            if (String(col.id) !== String(runInfo.columnDefinitionId)) return col
            return {
                ...col,
                latest_run: {
                    ...(col.latest_run || {}),
                    id: runInfo.runId,
                    run_id: runInfo.runId,
                    status: 'queued',
                },
            }
        }))
    }, [])

    const deleteRoleAiColumn = async (definition) => {
        if (!window.confirm(`Delete smart column "${definition?.name}"?`)) return
        const id = Number(definition?.id)
        if (!Number.isFinite(id)) {
            toast.error('Invalid smart column id — try refreshing the page')
            return
        }
        const previousColumns = roleAiColumns
        setRoleAiColumns(prev => (prev || []).filter(col => Number(col.id) !== id))
        try {
            await longOperationAxios.delete(`${API_BASE}/ai-columns/${id}`, { timeout: 120000 })
            toast.success('Smart column removed')
            void fetchRoleAiColumns()
        } catch (error) {
            setRoleAiColumns(previousColumns || [])
            toast.error(error.response?.data?.detail || 'Failed to delete smart column')
        }
    }

    const rerunRoleAiColumn = async (definition) => {
        const selectedIdArray = Array.from(selectedIds || [])
        if (selectedIdArray.length === 0) {
            toast.error('Select at least one row first, then run the smart column')
            return
        }
        const previousColumns = roleAiColumns
        setRoleAiColumns(prev => (prev || []).map(col => {
            if (col.id !== definition.id) return col
            const updatedCells = { ...(col.cells_by_candidate || {}) }
            selectedIdArray.forEach(id => {
                updatedCells[id] = { ...(updatedCells[id] || {}), status: 'queued', primary_output: '', outputs: {} }
            })
            return {
                ...col,
                cells_by_candidate: updatedCells,
                latest_run: { ...(col.latest_run || {}), status: 'queued', total: selectedIdArray.length, completed: 0, failed: 0, skipped: 0 },
            }
        }))
        try {
            await longOperationAxios.post(`${API_BASE}/ai-columns/run`, {
                column_definition_id: definition.id,
                selection_mode: 'selected_ids',
                selected_ids: selectedIdArray,
                view_scope: roleViewScope,
                role_id: viewingRole?.id || null,
            })
            toast.success(`Running "${definition.name}" on ${selectedIdArray.length} row(s)…`)
            void fetchRoleAiColumns()
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Failed to run smart column')
            setRoleAiColumns(previousColumns)
        }
    }

    const openRoleAiCellDrawer = async (definition, candidate, outputKey) => {
        const candidateName = (candidate?.name || `${candidate?.first_name || ''} ${candidate?.last_name || ''}`).trim()
        setAiCellDrawer({ open: true, loading: true, detail: null, title: `${definition?.name || 'Smart Column'} • ${candidateName}` })
        try {
            const res = await longOperationAxios.get(`${API_BASE}/ai-columns/${definition.id}/cells/${candidate.id}`, { timeout: 60000 })
            const detail = res.data || null
            if (detail && outputKey && detail.outputs && detail.outputs[outputKey] && outputKey !== 'primary') {
                detail.primary_output = detail.outputs[outputKey]
            }
            setAiCellDrawer(current => ({ ...current, loading: false, detail }))
        } catch (error) {
            if (error.response?.status !== 404) {
                toast.error(error.response?.data?.detail || 'Failed to load AI cell details')
            }
            setAiCellDrawer(current => ({ ...current, loading: false, detail: null }))
        }
    }

    const [editingCell, setEditingCell] = useState({ candidateId: null, field: null })
    const [editValue, setEditValue] = useState('')
    const editOriginalRef = useRef('')

    const handleContactEdit = async (candidateId, field, value) => {
        const trimmed = String(value || '').trim()
        setEditingCell({ candidateId: null, field: null })
        setEditValue('')
        if (trimmed === editOriginalRef.current) return
        try {
            await axios.patch(`${API_BASE}/candidates/${candidateId}`, { [field]: trimmed })
            const patch = field === 'email'
                ? { email: trimmed }
                : { mobile_phone: trimmed, phone: trimmed }
            updateCandidateInRole(candidateId, patch)
            toast.success('Contact updated')
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Failed to update contact')
        }
    }

    const startEdit = (candidateId, field, currentValue) => {
        editOriginalRef.current = currentValue || ''
        setEditingCell({ candidateId, field })
        setEditValue(currentValue || '')
    }

    const cancelEdit = () => {
        setEditingCell({ candidateId: null, field: null })
        setEditValue('')
    }

    const [emailSetup, setEmailSetup] = useState(null)
    const [emailSetupLoading, setEmailSetupLoading] = useState(false)
    const [emailSetupModalOpen, setEmailSetupModalOpen] = useState(false)
    const [isSyncing, setIsSyncing] = useState(false)
    const [isDeactivating, setIsDeactivating] = useState(false)
    const [isRefreshing, setIsRefreshing] = useState(false)
    const [refreshingProfileIds, setRefreshingProfileIds] = useState({})
    const [isLoadingRole, setIsLoadingRole] = useState(false)
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

    useEffect(() => {
        if (!viewingRole?.id) {
            setRoleFilteredCandidates([])
            setRoleStatusCounts({})
            setRoleFilterMeta({ titles: [], companies: [], cities: [], products: [], statuses: [] })
            return
        }
        const baseCandidates = viewingRole.candidates || []
        setRoleFilteredCandidates(baseCandidates)
        setRoleStatusCounts(baseCandidates.reduce((counts, candidate) => {
            const status = candidate.status || 'To be started'
            counts[status] = (counts[status] || 0) + 1
            return counts
        }, {}))
        // Restore this role's saved filters (survives page refresh); fall back
        // to a clean slate when nothing was saved.
        let saved = null
        try {
            saved = JSON.parse(window.localStorage.getItem(`role_filters_v1_${viewingRole.id}`) || 'null')
        } catch { saved = null }
        setFilters({
            title: [], titleInput: '',
            company: [], companyInput: '',
            city: [], cityInput: '',
            product_service: [], productInput: '',
            status: [], statusInput: '',
            min_exp: 0, max_exp: 40,
            added_from: '', added_to: '',
            ...(saved && typeof saved.filters === 'object' ? saved.filters : {}),
        })
        setActiveStatusTab(typeof saved?.activeStatusTab === 'string' ? saved.activeStatusTab : '')
        setRoleSearch(typeof saved?.roleSearch === 'string' ? saved.roleSearch : '')
        roleFiltersHydratedRef.current = viewingRole.id
        setSelectedIds(new Set())
        setAllFilteredSelected(false)
    }, [viewingRole?.id])

    // Persist filters per role so a page refresh keeps the exact same view.
    useEffect(() => {
        if (!viewingRole?.id || roleFiltersHydratedRef.current !== viewingRole.id) return
        try {
            window.localStorage.setItem(
                `role_filters_v1_${viewingRole.id}`,
                JSON.stringify({ filters, roleSearch, activeStatusTab }),
            )
        } catch { /* storage unavailable */ }
    }, [viewingRole?.id, filters, roleSearch, activeStatusTab])

    useEffect(() => {
        if (!viewingRole?.id || !Array.isArray(viewingRole.candidates)) return
        const currentById = new Map(viewingRole.candidates.map(candidate => [Number(candidate.id), candidate]))
        setRoleFilteredCandidates(previous => {
            if (!hasActiveRoleFilters) return viewingRole.candidates
            return previous
                .filter(candidate => currentById.has(Number(candidate.id)))
                .map(candidate => ({ ...candidate, ...currentById.get(Number(candidate.id)) }))
        })
        if (!hasActiveRoleFilters) {
            setRoleStatusCounts(viewingRole.candidates.reduce((counts, candidate) => {
                const status = candidate.status || 'To be started'
                counts[status] = (counts[status] || 0) + 1
                return counts
            }, {}))
        }
    }, [hasActiveRoleFilters, viewingRole?.candidates, viewingRole?.id])

    useEffect(() => {
        if (!viewingRole?.id) return undefined
        let cancelled = false
        const params = roleScopeParams()
        axios.get(`${API_BASE}/candidates/browse/meta?${params.toString()}&cb=${Date.now()}`, { timeout: 60000 })
            .then(response => {
                if (cancelled) return
                const data = response.data || {}
                setRoleFilterMeta({
                    titles: Array.isArray(data.titles) ? data.titles : [],
                    companies: Array.isArray(data.companies) ? data.companies : [],
                    cities: Array.isArray(data.cities) ? data.cities : [],
                    products: Array.isArray(data.products) ? data.products : [],
                    statuses: Array.isArray(data.statuses) ? data.statuses : [],
                })
            })
            .catch(error => {
                if (!cancelled) console.error('Failed to load role filter metadata:', error)
            })
        return () => { cancelled = true }
    }, [roleScopeParams, viewingRole?.id])

    useEffect(() => {
        if (!viewingRole?.id) return undefined
        window.clearTimeout(roleFilterDebounceRef.current)
        const requestSeq = ++roleFilterRequestSeqRef.current
        const params = roleScopeParams()
        params.set('page', '1')
        params.set('page_size', '5000')
        params.set('sort_by', 'name')
        params.set('sort_dir', 'asc')

        const query = roleSearch.trim()
        const title = joinFilterParam(splitFilterValues(filters.title, filters.titleInput))
        const company = joinFilterParam(splitFilterValues(filters.company, filters.companyInput))
        const city = joinFilterParam(splitFilterValues(filters.city, filters.cityInput))
        const product = joinFilterParam(splitFilterValues(filters.product_service, filters.productInput))
        const statusValues = splitFilterValues(filters.status, filters.statusInput)
        const status = joinFilterParam(statusValues)
        if (query) params.set('q', query)
        if (title) params.set('title', title)
        if (company) params.set('company', company)
        if (city) params.set('city', city)
        if (product) params.set('product_service', product)
        if (status) params.set('status', status)
        if (Number(filters.min_exp || 0) > 0) params.set('min_exp', filters.min_exp)
        if (Number(filters.max_exp ?? 40) < 40) params.set('max_exp', filters.max_exp)
        if (filters.added_from) params.set('added_from', filters.added_from)
        if (filters.added_to) params.set('added_to', filters.added_to)

        if (!hasServerRoleFilters) {
            const baseCandidates = viewingRole.candidates || []
            setRoleFilteredCandidates(baseCandidates)
            setRoleStatusCounts(baseCandidates.reduce((counts, candidate) => {
                const stat = candidate.status || 'To be started'
                counts[stat] = (counts[stat] || 0) + 1
                return counts
            }, {}))
            setIsFilteringRole(false)
            return undefined
        }

        roleFilterDebounceRef.current = window.setTimeout(async () => {
            setIsFilteringRole(true)
            try {
                const response = await axios.get(`${API_BASE}/candidates/browse?${params.toString()}&cb=${Date.now()}`, { timeout: 60000 })
                if (requestSeq !== roleFilterRequestSeqRef.current) return
                // Browse rows don't carry role-link fields (added_to_role_at) —
                // merge them in from the role's own candidate list.
                const roleById = new Map((viewingRole.candidates || []).map(c => [Number(c.id), c]))
                setRoleFilteredCandidates((response.data?.candidates || []).map(row => {
                    const roleRow = roleById.get(Number(row.id))
                    return roleRow?.added_to_role_at ? { ...row, added_to_role_at: roleRow.added_to_role_at } : row
                }))
                setRoleStatusCounts(response.data?.status_counts || {})
            } catch (error) {
                if (requestSeq === roleFilterRequestSeqRef.current) {
                    console.error('Failed to filter role candidates:', error)
                }
            } finally {
                if (requestSeq === roleFilterRequestSeqRef.current) setIsFilteringRole(false)
            }
        }, 120)

        return () => window.clearTimeout(roleFilterDebounceRef.current)
    }, [
        activeStatusTab,
        filters,
        roleSearch,
        roleScopeParams,
        splitFilterValues,
        joinFilterParam,
        viewingRole?.id,
        hasServerRoleFilters,
        viewingRole?.candidates,
    ])

    useEffect(() => {
        if (!viewingRole?.id) {
            setEmailSetup(null)
            return undefined
        }
        let cancelled = false
        setEmailSetupLoading(true)
        axios.get(`${API_BASE}/outreach/roles/${viewingRole.id}/email-setup?cb=${Date.now()}`)
            .then(res => { if (!cancelled) setEmailSetup(res.data) })
            .catch(error => {
                if (!cancelled) {
                    setEmailSetup(null)
                    console.error('Failed to load role email setup:', error)
                }
            })
            .finally(() => { if (!cancelled) setEmailSetupLoading(false) })
        return () => { cancelled = true }
    }, [viewingRole?.id])

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
            let cancelled = false
            const roleId = viewingRole.id
            fetchOutreachStatus(roleId).finally(() => {
                if (cancelled) return
                setOutreachCountLoadedRoles(prev => {
                    const next = new Set(prev)
                    next.add(String(roleId))
                    return next
                })
            })
            return () => { cancelled = true }
        }
        return undefined
    }, [fetchOutreachStatus, viewingRole?.id])

    // Keep contact cells in sync with Clay without reloading the full role.
    // This polls one lightweight role-scoped endpoint and stops after two minutes.
    useEffect(() => {
        const candidates = viewingRole?.candidates
        if (!viewingRole?.name || !Array.isArray(candidates) || candidates.length === 0) return undefined
        if (!candidates.some(candidate => !candidate.email || !candidate.mobile_phone
            || ['scheduled', 'waiting_for_email', 'email_enrolling'].includes(candidate.email_outreach_status)
            || ['scheduled', 'enrolling'].includes(candidate.linkedin_outreach_status))) return undefined

        let cancelled = false
        let timer = null
        let attempts = 0

        const pollContacts = async () => {
            attempts += 1
            try {
                const res = await axios.get(
                    `${API_BASE}/roles/${encodeURIComponent(viewingRole.name)}/contacts?cb=${Date.now()}`,
                    { headers: { 'Cache-Control': 'no-cache' } }
                )
                if (cancelled) return

                const contactById = new Map(
                    (res.data?.contacts || []).map(contact => [Number(contact.id), contact])
                )
                let hasMissingContacts = false

                useAppStore.setState(state => {
                    if (state.viewingRole?.name !== viewingRole.name) return state

                    const roleOutreach = { ...(state.outreachStatusCache[viewingRole.id] || {}) }
                    for (const contact of (res.data?.contacts || [])) {
                        roleOutreach[contact.id] = {
                            ...(roleOutreach[contact.id] || {}),
                            status: contact.email_status,
                            li_status: contact.linkedin_status,
                        }
                    }

                    const updatedCandidates = (state.viewingRole.candidates || []).map(candidate => {
                        const contact = contactById.get(Number(candidate.id))
                        const updated = contact ? {
                            ...candidate,
                            email: contact.email || candidate.email || '',
                            mobile_phone: contact.mobile_phone || candidate.mobile_phone || '',
                            email_outreach_status: contact.email_status || candidate.email_outreach_status,
                            linkedin_outreach_status: contact.linkedin_status || candidate.linkedin_outreach_status,
                        } : candidate
                        if (!updated.email || !updated.mobile_phone) hasMissingContacts = true
                        return updated
                    })
                    const updatedRole = { ...state.viewingRole, candidates: updatedCandidates }

                    return {
                        viewingRole: updatedRole,
                        roleDetailsCache: {
                            ...state.roleDetailsCache,
                            [viewingRole.name]: updatedRole,
                        },
                        outreachStatusCache: {
                            ...state.outreachStatusCache,
                            [viewingRole.id]: roleOutreach,
                        },
                    }
                })

                const stillPending = (res.data?.contacts || []).some(contact =>
                    ['scheduled', 'waiting_for_email', 'email_enrolling'].includes(contact.email_status)
                    || ['scheduled', 'enrolling'].includes(contact.linkedin_status))
                if ((hasMissingContacts || stillPending) && attempts < 150 && !cancelled) {
                    timer = setTimeout(pollContacts, 2000)
                }
            } catch {
                if (attempts < 150 && !cancelled) timer = setTimeout(pollContacts, 2000)
            }
        }

        timer = setTimeout(pollContacts, 1500)
        return () => {
            cancelled = true
            if (timer) clearTimeout(timer)
        }
    }, [viewingRole?.id, viewingRole?.candidates?.length])

    // Local fetchOutreachStatus removed in favor of global store action

    const mergeCandidateContact = (candidateId, contact = {}) => {
        useAppStore.setState(state => {
            if (!state.viewingRole?.candidates) return state
            const updatedCandidates = state.viewingRole.candidates.map(candidate => (
                Number(candidate.id) === Number(candidateId)
                    ? {
                        ...candidate,
                        email: contact.email || candidate.email || '',
                        mobile_phone: contact.mobile_phone || contact.phone || candidate.mobile_phone || '',
                        email_outreach_status: contact.email_outreach_status || candidate.email_outreach_status,
                        linkedin_outreach_status: contact.linkedin_outreach_status || candidate.linkedin_outreach_status,
                    }
                    : candidate
            ))
            const updatedRole = { ...state.viewingRole, candidates: updatedCandidates }
            return {
                viewingRole: updatedRole,
                roleDetailsCache: {
                    ...state.roleDetailsCache,
                    [updatedRole.name]: updatedRole,
                },
            }
        })
    }

    const handleRefreshProfile = async (candidate) => {
        if (!candidate?.id || refreshingProfileIds[candidate.id]) return
        setRefreshingProfileIds(current => ({ ...current, [candidate.id]: true }))

        try {
            const enrichment = await axios.post(`${API_BASE}/enrich/${candidate.id}`)
            const immediateEmail = enrichment.data?.email || ''
            const immediatePhone = enrichment.data?.phone || ''
            if (immediateEmail || immediatePhone) {
                mergeCandidateContact(candidate.id, { email: immediateEmail, phone: immediatePhone })
                toast.success(`Refreshed ${candidate.name || 'candidate'} profile`)
                return
            }

            toast.info(`Refreshing ${candidate.name || 'candidate'} from Clay…`)
            for (let attempt = 0; attempt < 20; attempt += 1) {
                await new Promise(resolve => setTimeout(resolve, 2000))
                const contacts = await axios.get(
                    `${API_BASE}/roles/${encodeURIComponent(viewingRole.name)}/contacts?cb=${Date.now()}`,
                    { headers: { 'Cache-Control': 'no-cache' } }
                )
                const contact = (contacts.data?.contacts || []).find(
                    item => Number(item.id) === Number(candidate.id)
                )
                if (contact?.email || contact?.mobile_phone) {
                    mergeCandidateContact(candidate.id, contact)
                    toast.success(`Clay data updated for ${candidate.name || 'candidate'}`)
                    return
                }
            }

            toast.warning('Clay finished, but its result has not reached Hayasa yet. Check the Clay callback step.')
        } catch (error) {
            toast.error(error.response?.data?.detail || `Failed to refresh ${candidate.name || 'candidate'}`)
        } finally {
            setRefreshingProfileIds(current => ({ ...current, [candidate.id]: false }))
        }
    }

    const handleManualRefresh = async () => {
        if (!viewingRole?.name) return
        setIsRefreshing(true)
        try {
            const refreshed = await fetchRoleDetails(viewingRole.name, { force: true })
            if (!refreshed?.success) throw new Error(refreshed?.error || 'Refresh failed')
            await fetchOutreachStatus(viewingRole.id)
            await fetchRoles({ force: true })
            toast.success('Refreshed role data')
        } catch (error) {
            toast.error('Failed to refresh')
        } finally {
            setIsRefreshing(false)
        }
    }

    const handleDeactivate = async () => {
        if (!viewingRole) return
        setIsDeactivating(true)
        const res = await deactivateRole(viewingRole.id)
        if (res.success) {
            toast.success("Role deactivated successfully.")
        } else {
            toast.error(res.error || "Failed to deactivate role")
        }
        setIsDeactivating(false)
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

    useEffect(() => {
        if (!addCandidateOpen || !viewingRole?.id) return undefined
        window.clearTimeout(candidateSearchDebounceRef.current)
        candidateSearchDebounceRef.current = window.setTimeout(async () => {
            const params = new URLSearchParams()
            params.set('page', '1')
            params.set('page_size', '25')
            params.set('sort_by', 'name')
            params.set('sort_dir', 'asc')
            const query = candidateSearch.trim()
            if (query) params.set('q', query)
            if (String(user?.role || '').toLowerCase() === 'admin') {
                if (addCandidateScope === 'owner_pool' && viewingRole?.owner_user_id) {
                    params.set('view_scope', 'recruiter_pools')
                    params.set('recruiter_filter_id', viewingRole.owner_user_id)
                } else {
                    params.set('view_scope', 'master')
                }
            }
            setCandidateSearchLoading(true)
            try {
                const res = await axios.get(`${API_BASE}/candidates/browse?${params.toString()}&cb=${Date.now()}`, { timeout: 60000 })
                const assignedIds = new Set((viewingRole.candidates || []).map(candidate => Number(candidate.id)))
                setCandidateSearchResults((res.data?.candidates || []).filter(candidate => !assignedIds.has(Number(candidate.id))))
            } catch (error) {
                console.error('Failed to search candidates for role assignment:', error)
                setCandidateSearchResults([])
            } finally {
                setCandidateSearchLoading(false)
            }
        }, 180)
        return () => window.clearTimeout(candidateSearchDebounceRef.current)
    }, [addCandidateOpen, addCandidateScope, candidateSearch, user?.role, viewingRole?.candidates, viewingRole?.id, viewingRole?.owner_user_id])

    const toggleCandidateAdd = useCallback((candidateId) => {
        setSelectedCandidateAdds(previous => {
            const next = new Set(previous)
            if (next.has(candidateId)) next.delete(candidateId)
            else next.add(candidateId)
            return next
        })
    }, [])

    const handleAddCandidatesToRole = async () => {
        if (!viewingRole?.name || selectedCandidateAdds.size === 0) return
        setIsAddingCandidates(true)
        const assignments = Array.from(selectedCandidateAdds).map(candidateId => ({
            candidate_id: Number(candidateId),
            priority: '--',
            feedback: '',
        }))
        const result = await assignCandidatesToRole(viewingRole.name, assignments)
        setIsAddingCandidates(false)
        if (!result.success) {
            toast.error(result.error || 'Failed to add candidates')
            return
        }
        toast.success(result.data?.message || `Added ${assignments.length} candidate${assignments.length === 1 ? '' : 's'}`)
        setSelectedCandidateAdds(new Set())
        setCandidateSearch('')
        setAddCandidateOpen(false)
    }

    const toggleSelection = useCallback((id) => {
        if (allFilteredSelected) return; // Prevent individual toggles when 'select all' is active
        setSelectedIds(prev => {
            const next = new Set(prev)
            if (next.has(id)) next.delete(id)
            else next.add(id)
            return next
        })
    }, [allFilteredSelected])

    const handleSelectAll = useCallback((e) => {
        if (e.target.checked) {
            setAllFilteredSelected(true)
            setSelectedIds(new Set(filteredCandidates.map(c => c.id)))
        } else {
            setAllFilteredSelected(false)
            setSelectedIds(new Set())
        }
    }, [filteredCandidates])


    const handleCreateRole = async (setup) => {
        const res = await createRole(setup)
        if (!res.success) return res
        const createdRole = { ...res.data, candidate_count: 0, upload_count: 0 }
        if (res.data?.activation_status === 'active') toast.success(`Role "${setup.name}" created and activated`)
        else toast.warning(`Role "${setup.name}" was saved but needs activation: ${res.data?.activation_error || 'setup failed'}`)
        return { ...res, createdRole }
    }

    const handleRetryActivation = async (role) => {
        setActivationRole(role)
    }

    const handleActivateExisting = async (setup) => {
        try {
            const role = activationRole
            const res = await axios.put(`${API_BASE}/roles/id/${role.id}/activation`, {
                heyreach_campaign_id: setup.heyreach_campaign_id,
                smartlead_sender_account_id: setup.smartlead_sender_account_id,
                smartlead_campaign_id: setup.smartlead_campaign_id,
            })
            await fetchRoles({ force: true })
            if (viewingRole?.id === role.id) await fetchRoleDetails(role.name, { force: true })
            if (res.data?.activation_status === 'active') {
                toast.success(`${role.name} is active`)
                return { success: true, data: res.data }
            }
            const message = res.data?.activation_error || 'Activation is still incomplete'
            return { success: false, error: message }
        } catch (error) {
            // Provisioning touches two external systems and may outlive the browser
            // connection. Verify durable state before reporting a false failure.
            try {
                const verification = await axios.get(`${API_BASE}/roles/id/${activationRole.id}/activation`)
                if (verification.data?.activation_status === 'active') {
                    await fetchRoles({ force: true })
                    if (viewingRole?.id === activationRole.id) {
                        await fetchRoleDetails(activationRole.name, { force: true })
                    }
                    toast.success(`${activationRole.name} is active`)
                    return { success: true, data: verification.data }
                }
                return {
                    success: false,
                    error: verification.data?.activation_error
                        || error.response?.data?.detail
                        || error.message
                        || 'Activation failed',
                }
            } catch (verificationError) {
                return {
                    success: false,
                    error: error.response?.data?.detail
                        || verificationError.response?.data?.detail
                        || error.message
                        || 'Activation failed',
                }
            }
        }
    }

    const handleRoleStatusUpdate = async (candidateId, newStatus) => {
        if (newStatus !== 'Shortlisted') {
            return axios.post(`${API_BASE}/candidates/${candidateId}/status`, { status: newStatus })
        }
        try {
            const res = await axios.post(
                `${API_BASE}/outreach/roles/${viewingRole.id}/candidates/${candidateId}/shortlist`,
                {},
                { timeout: 60000 },
            )
            mergeCandidateContact(candidateId, {
                email: res.data?.email,
                phone: res.data?.phone,
                email_outreach_status: res.data?.email_outreach,
                linkedin_outreach_status: res.data?.linkedin_outreach,
            })
            if (res.data?.already_processed) {
                toast.success('Candidate is already actively pushing in the outreach queue')
            } else {
                toast.success(res.data?.contact_enriching ? 'Shortlisted · retrieving contact details' : 'Shortlisted · outreach queued')
            }
            setTimeout(() => fetchOutreachStatus(viewingRole.id), 1200)
            return res
        } catch (error) {
            const apiError = error.response?.data
            toast.error(apiError?.detail || (typeof apiError === 'string' ? apiError : '') || error.message || 'Could not start shortlist outreach')
            throw error
        }
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

        setRoleFilteredCandidates(previous => previous.filter(candidate => Number(candidate.id) !== Number(candidateId)))
        toast.success(`${candidateName} removed from role`, { duration: 1000 })
        removeCandidateFromRole(viewingRole.name, candidateId).then(res => {
            if (!res.success) {
                const restored = useAppStore.getState().viewingRole?.candidates || []
                setRoleFilteredCandidates(restored)
                toast.error(res.error || 'Failed to remove candidate')
            }
        })
    }

    const openRoleUploadPicker = (role) => {
        setUploadRole(role)
        uploadFileRef.current?.click()
    }

    const runRoleUploadPreview = async (role, file) => {
        if (!role?.name || !file) return
        const fd = new FormData()
        fd.append('file', file)
        fd.append('use_llm', 'true')
        setUploadPreviewBusy(role.name)
        try {
            const res = await axios.post(`${API_BASE}/roles/${encodeURIComponent(role.name)}/upload/preview`, fd, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000,
            })
            const headers = res.data.headers || []
            const sm = res.data.suggested_mapping || {}
            const init = {}
            for (const h of headers) init[h] = sm[h] || 'ignore'
            setUploadRole(role)
            setUploadFile(file)
            setUploadHeaders(headers)
            setUploadMapping(init)
            setUploadMappingDetails(res.data.mapping_details || {})
            setUploadRequiredTargets(res.data.required_targets || ['first_name', 'last_name', 'linkedin', 'city', 'title'])
            setUploadTargetOptions(res.data.target_options || [])
            setUploadRowCount(Number(res.data.row_count) || 0)
            setUploadProgress(null)
            // Bug-4 fix: auto-enable verified enrichment for Apify/LinkedIn-style CSVs
            // that already contain structured work-history columns (experiences/*).
            // The user can still override this in the modal.
            const hasExperienceCols = headers.some(h => /^experiences\/\d+\//i.test(String(h || '')))
            setUploadEnrichmentMode(hasExperienceCols ? 'verified_profile' : 'none')
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Preview failed')
        } finally {
            setUploadPreviewBusy('')
        }
    }

    const pollUploadStatus = async (uploadId) => {
        for (let attempt = 0; attempt < 900; attempt += 1) {
            const res = await axios.get(`${API_BASE}/candidates/uploads/${uploadId}`, { timeout: 60000 })
            const next = res.data || {}
            setUploadProgress(next)
            if (['completed', 'completed_with_errors', 'failed'].includes(String(next.status || '').toLowerCase())) {
                return next
            }
            await new Promise(resolve => window.setTimeout(resolve, 700))
        }
        throw new Error('Upload is still running. Check recent uploads for its status.')
    }

    const commitRoleUpload = async () => {
        if (!uploadRole?.name || !uploadFile) return
        const fd = new FormData()
        fd.append('file', uploadFile)
        fd.append('mapping_json', JSON.stringify(uploadMapping))
        fd.append('enrichment_mode', uploadEnrichmentMode || 'none')
        setUploadCommitBusy(true)
        try {
            const res = await axios.post(`${API_BASE}/roles/${encodeURIComponent(uploadRole.name)}/upload/commit`, fd, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000,
            })
            setUploadProgress(res.data || {})
            const d = await pollUploadStatus(res.data?.upload_id)
            if (String(d.status || '').toLowerCase() === 'failed') {
                toast.error(d.error_message || 'Upload failed')
                return
            }
            const rows = Number(d.row_count) || 0
            const assigned = Number(d.role_assigned_count) || 0
            const skipped = Number(d.skipped) || 0
            toast.success(`Import complete: ${rows} rows processed, ${assigned} new role assignment${assigned === 1 ? '' : 's'}${skipped ? `, ${skipped} skipped` : ''}.`)
            if (Array.isArray(d.errors) && d.errors.length > 0) {
                toast.warning(`Some rows had issues: ${d.errors.slice(0, 3).join(' ')}`)
            }
            invalidateTalentPoolCaches()
            await fetchRoles({ force: true })
            fetchTalentPoolSummary({ force: true, freshnessMs: 0 })
            fetchAnalytics({ force: true })
            if (viewingRole?.name === uploadRole.name) {
                await fetchRoleDetails(uploadRole.name)
            }
        } catch (e) {
            toast.error(e.response?.data?.detail || e.message || 'Upload failed')
        } finally {
            setUploadCommitBusy(false)
        }
    }

    const uploadUi = (
        <>
            <input
                ref={uploadFileRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                style={{ display: 'none' }}
                onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f && uploadRole) runRoleUploadPreview(uploadRole, f)
                    e.target.value = ''
                }}
            />
            {uploadRole && uploadFile && (
                <CsvMappingModal
                    title={`Map columns for ${uploadRole.name}`}
                    subtitle="Review smart suggestions, adjust any field, then import this CSV into the role."
                    headers={uploadHeaders}
                    mapping={uploadMapping}
                    details={uploadMappingDetails}
                    requiredTargets={uploadRequiredTargets}
                    targetOptions={uploadTargetOptions}
                    rowCount={uploadRowCount}
                    progress={uploadProgress}
                    enrichmentMode={uploadEnrichmentMode}
                    busy={uploadCommitBusy}
                    onEnrichmentModeChange={setUploadEnrichmentMode}
                    onChange={(header, value) => setUploadMapping(prev => ({ ...prev, [header]: value }))}
                    onCancel={() => {
                        setUploadRole(null)
                        setUploadFile(null)
                        setUploadHeaders([])
                        setUploadMapping({})
                        setUploadMappingDetails({})
                        setUploadProgress(null)
                        setUploadRowCount(0)
                        setUploadEnrichmentMode('none')
                    }}
                    onImport={commitRoleUpload}
                />
            )}
        </>
    )



    const [chattingWith, setChattingWith] = useState(null)

    const updateCandidateInRole = useCallback((candidateId, patch) => {
        if (patch.status) {
            const storeCandidate = useAppStore.getState().viewingRole?.candidates?.find(candidate => Number(candidate.id) === Number(candidateId))
            const oldStatus = storeCandidate?.status || 'To be started'
            if (oldStatus !== patch.status) {
                setRoleStatusCounts(previous => ({
                    ...previous,
                    [oldStatus]: Math.max(0, Number(previous[oldStatus] || 0) - 1),
                    [patch.status]: Number(previous[patch.status] || 0) + 1,
                }))
            }
            setRoleFilteredCandidates(previous => {
                const activeStatuses = (Array.isArray(filters.status) ? filters.status : []).filter(Boolean)
                const hasStatusFilter = activeStatusTab !== RESPONDED_TAB && activeStatuses.length > 0
                const matchesFilter = activeStatuses.includes(patch.status)
                const current = previous.find(candidate => Number(candidate.id) === Number(candidateId))
                if (!current) {
                    if (hasStatusFilter && matchesFilter && storeCandidate) {
                        return [...previous, { ...storeCandidate, ...patch }]
                            .sort((a, b) => Number(a.id || 0) - Number(b.id || 0))
                    }
                    return previous
                }
                if (hasStatusFilter && !matchesFilter) {
                    return previous.filter(candidate => Number(candidate.id) !== Number(candidateId))
                }
                return previous.map(candidate =>
                    Number(candidate.id) === Number(candidateId) ? { ...candidate, ...patch } : candidate
                )
            })
        }
        useAppStore.setState(state => {
            if (!state.viewingRole?.candidates) return state
            const updatedRole = {
                ...state.viewingRole,
                candidates: state.viewingRole.candidates.map(candidate =>
                    Number(candidate.id) === Number(candidateId) ? { ...candidate, ...patch } : candidate
                ),
            }
            return {
                viewingRole: updatedRole,
                roleDetailsCache: {
                    ...state.roleDetailsCache,
                    [updatedRole.name]: updatedRole,
                },
            }
        })
        setChattingWith(current =>
            Number(current?.id) === Number(candidateId) ? { ...current, ...patch } : current
        )
    }, [activeStatusTab, filters.status])

    // Role Detail View
    if (viewingRole) {
        return (
            <div className="roles-page" style={{ width: '100%', position: 'relative', minHeight: '100vh', animation: 'fadeIn 0.2s ease-out' }}>
                {activationRole && <RoleCreateModal role={activationRole} onClose={() => setActivationRole(null)} onSubmit={handleActivateExisting} />}
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
                    <button className="btn btn-secondary" onClick={closeRoleView}>
                        <ArrowLeft size={16} /> Back to Roles
                    </button>
                    <h2 style={{ margin: 0, fontSize: '24px', fontWeight: 700, color: '#1e293b' }}>{viewingRole.name}</h2>
                    <button
                        className="btn btn-secondary"
                        onClick={() => setAddCandidateOpen(true)}
                        style={{ marginLeft: 'auto' }}
                    >
                        <UserPlus size={16} /> Add Candidate
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={() => openRoleUploadPicker(viewingRole)}
                        disabled={uploadPreviewBusy === viewingRole.name}
                    >
                        <FileUp size={16} /> {uploadPreviewBusy === viewingRole.name ? 'Reading...' : 'Upload CSV'}
                    </button>
                </div>
                {uploadUi}
                {emailSetupModalOpen && (
                    <RoleEmailSendModal
                        roleId={viewingRole.id}
                        roleName={viewingRole.name}
                        onClose={() => setEmailSetupModalOpen(false)}
                        onSaved={setEmailSetup}
                    />
                )}
                {addCandidateOpen && (
                    <div className="modal-overlay" onClick={() => !isAddingCandidates && setAddCandidateOpen(false)}>
                        <div className="role-add-candidate-modal" onClick={event => event.stopPropagation()}>
                            <div className="role-add-candidate-header">
                                <div>
                                    <strong>Add Candidate</strong>
                                    <span>{viewingRole.name}</span>
                                </div>
                                <button type="button" className="icon-btn" disabled={isAddingCandidates} onClick={() => setAddCandidateOpen(false)}>
                                    <X size={18} />
                                </button>
                            </div>
                            <div className="role-add-search">
                                <Search size={16} />
                                <input
                                    autoFocus
                                    value={candidateSearch}
                                    onChange={event => setCandidateSearch(event.target.value)}
                                    placeholder="Search candidates by name, company, title, or location"
                                />
                                {candidateSearchLoading && <Loader2 size={15} className="animate-spin" />}
                            </div>
                            {String(user?.role || '').toLowerCase() === 'admin' && (
                                <div style={{ display: 'flex', gap: 6, padding: '8px 16px 0' }}>
                                    {[['talent_pool', 'Talent Pool'], ['owner_pool', "Role owner's pool"]].map(([value, label]) => (
                                        <button
                                            key={value}
                                            type="button"
                                            onClick={() => setAddCandidateScope(value)}
                                            style={{
                                                padding: '4px 12px', borderRadius: 999, fontSize: 11, fontWeight: 700,
                                                cursor: 'pointer', fontFamily: 'inherit',
                                                border: addCandidateScope === value ? '1px solid #0f172a' : '1px solid #e2e8f0',
                                                background: addCandidateScope === value ? '#0f172a' : '#fff',
                                                color: addCandidateScope === value ? '#fff' : '#64748b',
                                            }}
                                        >
                                            {label}
                                        </button>
                                    ))}
                                </div>
                            )}
                            <div className="role-add-results">
                                {candidateSearchResults.map(candidate => {
                                    const checked = selectedCandidateAdds.has(candidate.id)
                                    return (
                                        <button
                                            key={candidate.id}
                                            type="button"
                                            className={`role-add-result ${checked ? 'selected' : ''}`}
                                            onClick={() => toggleCandidateAdd(candidate.id)}
                                        >
                                            <input type="checkbox" checked={checked} onChange={() => toggleCandidateAdd(candidate.id)} onClick={event => event.stopPropagation()} />
                                            <div>
                                                <strong>{candidate.name || `${candidate.first_name || ''} ${candidate.last_name || ''}`.trim() || 'Unnamed candidate'}</strong>
                                                <span>{candidate.title || candidate.headline || 'No title'}{candidate.company || candidate.current_company ? ` · ${candidate.company || candidate.current_company}` : ''}</span>
                                            </div>
                                            <small>{candidate.city || candidate.location || ''}</small>
                                        </button>
                                    )
                                })}
                                {!candidateSearchLoading && candidateSearchResults.length === 0 && (
                                    <div className="role-add-empty">No available candidates found.</div>
                                )}
                            </div>
                            <div className="role-add-footer">
                                <span>{selectedCandidateAdds.size} selected</span>
                                <div>
                                    <button type="button" className="btn btn-secondary" disabled={isAddingCandidates} onClick={() => setAddCandidateOpen(false)}>Cancel</button>
                                    <button type="button" className="btn btn-primary" disabled={isAddingCandidates || selectedCandidateAdds.size === 0} onClick={handleAddCandidatesToRole}>
                                        {isAddingCandidates ? <Loader2 size={15} className="animate-spin" /> : <UserPlus size={15} />}
                                        {isAddingCandidates ? 'Adding...' : 'Add to Role'}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* ── Compact toolbar bar ── */}
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap',
                    padding: '12px 18px', marginBottom: '16px',
                    background: '#fff', borderRadius: '14px',
                    border: '1px solid #e2e8f0',
                    boxShadow: '0 1px 3px rgba(15,23,42,0.04)'
                }}>
                    {/* Candidate count */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: 22, fontWeight: 800, color: '#0f172a', lineHeight: 1 }}>
                            {viewingRole.candidates === null ? '…' : (viewingRole.candidate_count ?? viewingRole.candidates?.length ?? 0)}
                        </span>
                        <span style={{ fontSize: 12, fontWeight: 600, color: '#64748b' }}>Candidates</span>
                    </div>

                    <div style={{ width: '1px', height: 24, background: '#e2e8f0' }} />

                    {/* Upload count */}
                    <span style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8' }}>
                        {Number(viewingRole.upload_count || 0)} upload{Number(viewingRole.upload_count || 0) === 1 ? '' : 's'}
                    </span>

                    {/* Right-side actions */}
                    <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={handleManualRefresh}
                            disabled={isRefreshing || viewingRole.candidates === null}
                            title="Reload candidates"
                            style={{ height: 32, padding: '0 12px', fontSize: 11 }}
                        >
                            <RefreshCcw size={13} className={isRefreshing ? 'animate-spin' : ''} />
                            {isRefreshing ? 'Refreshing…' : 'Refresh'}
                        </button>
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={handleSyncResponses}
                            disabled={isSyncing}
                            title="Sync replies from both platforms"
                            style={{ height: 32, padding: '0 12px', fontSize: 11 }}
                        >
                            <RefreshCcw size={13} className={isSyncing ? 'animate-spin' : ''} />
                            {isSyncing ? 'Syncing…' : 'Sync Responses'}
                        </button>
                        <button
                            className={`btn ${showFilters ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                            onClick={() => setShowFilters(!showFilters)}
                            style={{ height: 32, padding: '0 12px', fontSize: 11 }}
                            title="Toggle Filters"
                        >
                            <Filter size={13} /> Filters
                        </button>
                        <button
                            className={`btn btn-secondary btn-sm`}
                            onClick={() => setShowOutreach(!showOutreach)}
                            style={{ height: 32, padding: '0 12px', fontSize: 11, color: '#64748b' }}
                            title="Toggle outreach configuration"
                        >
                            <Settings2 size={13} /> Config {showOutreach ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </button>
                        {viewingRole.activation_status === 'active' && (
                            <button
                                className="btn btn-secondary btn-sm"
                                onClick={handleDeactivate}
                                disabled={isDeactivating}
                                title="Deactivate role (pause campaigns)"
                                style={{ height: 32, padding: '0 10px', fontSize: 11, color: '#ef4444' }}
                            >
                                <PowerOff size={13} className={isDeactivating ? 'animate-spin' : ''} />
                            </button>
                        )}
                    </div>
                </div>

                {/* ── Collapsible Outreach Config ── */}
                {showOutreach && (
                    <div style={{
                        display: 'flex', flexWrap: 'wrap', alignItems: 'stretch',
                        gap: '12px', marginBottom: '16px',
                        animation: 'fadeIn 0.2s ease-out'
                    }}>
                        <div style={{
                            display: 'flex', flexDirection: 'column', gap: '10px',
                            padding: '14px 16px', background: '#fff', borderRadius: '12px',
                            border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                            flex: '1 1 340px', minWidth: 0,
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                                <span style={{ fontSize: 11, fontWeight: 800, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Email Outreach</span>
                                {emailSetup?.campaign_id && <span style={{ fontSize: 10, color: '#64748b', background: '#f1f5f9', borderRadius: 999, padding: '3px 7px' }}>Campaign {emailSetup.campaign_id}</span>}
                            </div>
                            {emailSetupLoading ? <div style={{ display: 'flex', alignItems: 'center', gap: 7, color: '#64748b', fontSize: 12 }}><Loader2 size={13} className="animate-spin" /> Loading…</div> : emailSetup?.campaign_configured ? <div style={{ minWidth: 0 }}>
                                <div style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, color: '#334155' }}><Mail size={13} /><strong style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{emailSetup.sender_email}</strong></div>
                                <div style={{ marginTop: 4, fontSize: 11, color: '#64748b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>Shortlisted candidates push to campaign {emailSetup.campaign_id} automatically</div>
                            </div> : <div style={{ fontSize: 12, color: emailSetup?.campaign_error ? '#b91c1c' : '#64748b' }}>{emailSetup?.campaign_error || 'Not fully activated.'}</div>}
                            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                <button className="btn btn-secondary btn-sm" onClick={() => setEmailSetupModalOpen(true)} disabled={emailSetupLoading} style={{ height: 30, padding: '0 10px', fontSize: 11 }}><Settings2 size={12} /> Edit setup</button>
                                <button className={viewingRole.activation_status !== 'active' ? "btn btn-primary btn-sm" : "btn btn-secondary btn-sm"} onClick={() => handleRetryActivation(viewingRole)} style={{ height: 30, fontSize: 11 }}>{viewingRole.activation_status !== 'active' ? 'Configure & activate' : 'Update configuration'}</button>
                            </div>
                        </div>

                        <div style={{
                            display: 'flex', flexDirection: 'column', gap: '6px',
                            padding: '14px 16px', background: '#f0f9ff', borderRadius: '12px',
                            border: '1px solid #bae6fd', boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                            flex: '1 1 280px', minWidth: 0
                        }}>
                            <span style={{ fontSize: 11, fontWeight: 700, color: '#0369a1', textTransform: 'uppercase', letterSpacing: '0.05em' }}>LinkedIn Outreach (HeyReach)</span>
                            <div style={{ fontSize: 12, color: '#334155' }}>Campaign ID: <strong>{viewingRole.heyreach_campaign_id || 'Not configured'}</strong></div>
                            <div style={{ fontSize: 11, color: '#64748b' }}>Shortlisted → auto-queued for LinkedIn outreach.</div>
                            <button className={viewingRole.activation_status !== 'active' ? "btn btn-primary btn-sm" : "btn btn-secondary btn-sm"} onClick={() => handleRetryActivation(viewingRole)} style={{ height: 30, fontSize: 11, marginTop: 4, alignSelf: 'flex-start' }}>{viewingRole.activation_status !== 'active' ? 'Configure & activate' : 'Update configuration'}</button>
                        </div>
                    </div>
                )}

                {/* ── Main content area: Filters + Table ── */}
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
                    <div className="role-dashboard-shell">
                        {/* Sidebar Filters */}
                        {showFilters && (
                            <div className="sidebar-filters role-filter-sidebar" style={{ width: '258px', flexShrink: 0, padding: '18px', background: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 2px 12px rgba(15,23,42,0.04)', display: 'flex', flexDirection: 'column', position: 'sticky', top: '16px', transition: 'all 0.25s ease' }}>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, paddingBottom: 10, borderBottom: '1px solid #e2e8f0' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                                        <Filter size={14} color="#0f172a" />
                                        <span style={{ fontSize: 13, fontWeight: 800, color: '#0f172a' }}>Filters</span>
                                        {isFilteringRole && <Loader2 size={12} className="animate-spin" style={{ color: '#64748b' }} />}
                                    </div>
                                    <button onClick={() => setShowFilters(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: '#94a3b8', display: 'flex' }}><X size={14} /></button>
                                </div>
                                <TagFilterInput label="Title" values={filters?.title || []} inputValue={filters?.titleInput || ''} onInputChange={v => setFilter('titleInput', v)} onTagsChange={v => setFilter('title', v)} placeholder="e.g. Engineer" icon={Briefcase} suggestions={roleFilterMeta.titles} />
                                <TagFilterInput label="Current Company" values={filters?.company || []} inputValue={filters?.companyInput || ''} onInputChange={v => setFilter('companyInput', v)} onTagsChange={v => setFilter('company', v)} placeholder="e.g. Google" icon={Building2} suggestions={roleFilterMeta.companies} />
                                <TagFilterInput label="Product / Service" values={filters?.product_service || []} inputValue={filters?.productInput || ''} onInputChange={v => setFilter('productInput', v)} onTagsChange={v => setFilter('product_service', v)} placeholder="All Products/Services" icon={BarChart2} suggestions={roleFilterMeta.products} />
                                <TagFilterInput label="City" values={filters?.city || []} inputValue={filters?.cityInput || ''} onInputChange={v => setFilter('cityInput', v)} onTagsChange={v => setFilter('city', v)} placeholder="e.g. San Francisco" icon={MapPin} suggestions={roleFilterMeta.cities} />
                                
                                <TagFilterInput label="Status" values={filters?.status || []} inputValue={filters?.statusInput || ''} onInputChange={v => setFilter('statusInput', v)} onTagsChange={v => { setFilter('status', v); setActiveStatusTab(v.length === 1 ? v[0] : ''); }} placeholder="e.g. Shortlisted" icon={Filter} suggestions={roleStatusOptions} />
                                
                                <RangeSlider
                                  label="Total Experience"
                                  min={0}
                                  max={40}
                                  minValue={filters?.min_exp ?? 0}
                                  maxValue={filters?.max_exp ?? 40}
                                  onChange={(min, max) => {
                                    setFilter('min_exp', min);
                                    setFilter('max_exp', max);
                                  }}
                                />

                                <div style={{ marginTop: 14 }}>
                                    <div style={{ fontSize: 11, fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
                                        Added Date
                                    </div>
                                    <div style={{ display: 'flex', gap: 8 }}>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ fontSize: 10, fontWeight: 600, color: '#94a3b8', marginBottom: 3 }}>From</div>
                                            <input
                                                type="date"
                                                value={filters?.added_from || ''}
                                                max={filters?.added_to || undefined}
                                                onChange={e => setFilter('added_from', e.target.value)}
                                                style={{
                                                    width: '100%', padding: '7px 8px', borderRadius: 10,
                                                    border: '1.5px solid #e2e8f0', fontSize: 12, color: '#334155',
                                                    fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box',
                                                    background: '#fff',
                                                }}
                                            />
                                        </div>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ fontSize: 10, fontWeight: 600, color: '#94a3b8', marginBottom: 3 }}>To</div>
                                            <input
                                                type="date"
                                                value={filters?.added_to || ''}
                                                min={filters?.added_from || undefined}
                                                onChange={e => setFilter('added_to', e.target.value)}
                                                style={{
                                                    width: '100%', padding: '7px 8px', borderRadius: 10,
                                                    border: '1.5px solid #e2e8f0', fontSize: 12, color: '#334155',
                                                    fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box',
                                                    background: '#fff',
                                                }}
                                            />
                                        </div>
                                    </div>
                                    {(filters?.added_from || filters?.added_to) && (
                                        <button
                                            onClick={() => { setFilter('added_from', ''); setFilter('added_to', ''); }}
                                            style={{
                                                marginTop: 6, background: 'none', border: 'none', cursor: 'pointer',
                                                fontSize: 11, fontWeight: 600, color: '#6366f1', padding: 0,
                                            }}
                                        >
                                            Clear dates
                                        </button>
                                    )}
                                </div>

                                <div style={{ marginTop: 'auto', paddingTop: '20px' }}>
                                  <button
                                    onClick={clearFilters}
                                    style={{
                                      width: '100%', padding: '11px 12px', borderRadius: '12px',
                                      background: '#fff', border: '1px solid rgba(203, 213, 225, 0.9)',
                                      color: '#334155', fontSize: '12px', fontWeight: 700,
                                      cursor: 'pointer', transition: 'all 0.2s',
                                      boxShadow: '0 1px 2px rgba(15,23,42,0.03)'
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.background = '#f8fafc'}
                                    onMouseLeave={e => e.currentTarget.style.background = '#fff'}
                                  >
                                    <RefreshCcw size={13} style={{ display: 'inline', marginRight: 6, verticalAlign: '-2px' }} />
                                    Reset All Filters
                                  </button>
                                </div>
                            </div>
                        )}

                        {/* Main Table Area */}
                        <div style={{ flex: 1, minWidth: 0, transition: 'all 0.25s ease', display: 'flex', flexDirection: 'column' }}>
                            <div className="role-table-toolbar">
                                <div className="role-global-search">
                                    <Search size={16} />
                                    <input
                                        type="search"
                                        value={roleSearch}
                                        onChange={event => setRoleSearch(event.target.value)}
                                        placeholder="Global search across all columns..."
                                    />
                                    {isFilteringRole && <Loader2 size={15} className="animate-spin" />}
                                </div>
                                {!showFilters && (
                                    <button className="btn btn-secondary btn-sm" onClick={() => setShowFilters(true)}>
                                        <Filter size={13} /> Filters
                                    </button>
                                )}
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => setAiColumnModal({ mode: 'create' })}
                                    title={selectedIds.size > 0 ? `Run on ${selectedIds.size} selected candidates` : 'Select rows, then run a smart column'}
                                >
                                    <HayasaBrand size="compact" tone="light" showGrowton={false} />
                                    Smart Column
                                    {selectedIds.size > 0 && (
                                        <span style={{ background: '#eef2f7', color: '#334155', borderRadius: 999, padding: '1px 7px', fontSize: 11 }}>
                                            {selectedIds.size}
                                        </span>
                                    )}
                                </button>
                                <button className="btn btn-secondary btn-sm" onClick={() => setAddCandidateOpen(true)}>
                                    <UserPlus size={13} /> Add Candidate
                                </button>
                            </div>
                            {/* Status Tabs */}
                            <div style={{ padding: '14px 18px', background: 'rgba(248,250,252,0.78)', borderBottom: '1px solid #e2e8f0', display: 'flex', gap: 10, overflowX: 'auto', scrollbarWidth: 'none', borderRadius: '10px 10px 0 0', border: '1px solid #e2e8f0' }}>
                                {['', RESPONDED_TAB, ...visibleStatusTabs].map(tab => {
                                    const selectedStatuses = Array.isArray(filters.status) ? filters.status : [];
                                    const isResponded = activeStatusTab === RESPONDED_TAB;
                                    const isActive = tab === ''
                                        ? (!isResponded && selectedStatuses.length === 0)
                                        : (tab === RESPONDED_TAB ? isResponded : selectedStatuses.includes(tab));
                                    const count = tab === '' ? roleFilteredTotal : (tab === RESPONDED_TAB ? respondedCount : (statusCounts[tab] || 0));
                                    const style = tab ? (STATUS_STYLES[tab.toLowerCase()] || {}) : { bg: '#f1f5f9', color: '#475569', dot: '#94a3b8' };
                                    const label = tab === RESPONDED_TAB ? RESPONSE_TAB_LABEL : tab;

                                    return (
                                        <button key={tab || 'all'}
                                            onClick={() => {
                                                if (tab === RESPONDED_TAB) {
                                                    setFilters(prev => ({ ...prev, status: [], statusInput: '' }));
                                                    setActiveStatusTab(RESPONDED_TAB);
                                                    return;
                                                }
                                                if (!tab) {
                                                    setFilter('status', []);
                                                    setFilter('statusInput', '');
                                                    setActiveStatusTab('');
                                                    return;
                                                }
                                                const current = isResponded ? [] : selectedStatuses;
                                                const nextStatus = current.includes(tab)
                                                    ? current.filter(s => s !== tab)
                                                    : [...current, tab];
                                                setFilter('status', nextStatus);
                                                setFilter('statusInput', '');
                                            }}
                                            style={{
                                                padding: '7px 14px', borderRadius: '999px', border: isActive ? '1px solid #111827' : '1px solid rgba(203, 213, 225, 0.9)',
                                                background: isActive ? '#111827' : 'rgba(255,255,255,0.72)', cursor: 'pointer', fontSize: 12, fontWeight: 700,
                                                color: isActive ? '#fff' : '#64748b', whiteSpace: 'nowrap',
                                                display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'inherit',
                                                transition: 'all 0.15s',
                                                boxShadow: isActive ? '0 10px 18px rgba(15,23,42,0.12)' : 'none',
                                            }}
                                            onMouseEnter={e => {
                                                if (!isActive) {
                                                    e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.75)';
                                                    e.currentTarget.style.background = '#fff';
                                                }
                                            }}
                                            onMouseLeave={e => {
                                                if (!isActive) {
                                                    e.currentTarget.style.borderColor = 'rgba(203, 213, 225, 0.9)';
                                                    e.currentTarget.style.background = 'rgba(255,255,255,0.72)';
                                                }
                                            }}
                                        >
                                            {tab === '' ? (
                                                <span style={{ color: isActive ? '#fff' : '#0f172a' }}>All ({roleFilteredTotal})</span>
                                            ) : (
                                                <>
                                                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: isActive ? '#fff' : (style.dot || '#94a3b8') }} />
                                                    {label}
                                                    <span style={{
                                                        marginLeft: 4, padding: '1px 6px', borderRadius: 10, fontSize: 10,
                                                        background: isActive ? 'rgba(255,255,255,0.14)' : '#e2e8f0',
                                                        color: isActive ? '#fff' : '#64748b'
                                                    }}>
                                                        {count == null ? '…' : count}
                                                    </span>
                                                </>
                                            )}
                                        </button>
                                    );
                                })}
                                {Array.isArray(filters.status) && filters.status.length >= 2 && (
                                    <button
                                        onClick={() => {
                                            setFilter('status', []);
                                            setFilter('statusInput', '');
                                            setActiveStatusTab('');
                                        }}
                                        style={{
                                            padding: '7px 12px', borderRadius: '999px', border: '1px dashed rgba(203, 213, 225, 0.9)',
                                            background: 'transparent', cursor: 'pointer', fontSize: 12, fontWeight: 700,
                                            color: '#64748b', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: 6,
                                            fontFamily: 'inherit', transition: 'all 0.15s',
                                        }}
                                        onMouseEnter={e => { e.currentTarget.style.borderColor = '#94a3b8'; e.currentTarget.style.color = '#0f172a'; }}
                                        onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(203, 213, 225, 0.9)'; e.currentTarget.style.color = '#64748b'; }}
                                    >
                                        Clear ({filters.status.length})
                                    </button>
                                )}
                            </div>

                            <div className="table-wrapper role-candidates-table-wrapper" style={{ maxHeight: '600px', borderTop: 'none', borderRadius: '0 0 10px 10px' }}>
                                <table className="data-table role-candidates-table">
                                    <thead>
                                        <tr>
                                            <th style={{ width: 40, textAlign: 'center' }}>
                                                <input
                                                    type="checkbox"
                                                    onChange={handleSelectAll}
                                                    checked={filteredCandidates.length > 0 && selectedIds.size === filteredCandidates.length}
                                                    style={{ cursor: 'pointer', transform: 'scale(1.1)' }}
                                                />
                                            </th>
                                            <th className="role-sticky-first-name">First Name</th>
                                            <th className="role-sticky-last-name">Last Name</th>
                                            <th style={{ minWidth: 170 }}>Title</th>
                                            <th style={{ minWidth: 90 }}>LinkedIn</th>
                                            <th style={{ minWidth: 170 }}>Current Company</th>
                                            <th style={{ minWidth: 150 }}>Product/Service</th>
                                            <th style={{ minWidth: 120 }}>City</th>
                                            <th style={{ minWidth: 105 }}>Total Years</th>
                                            <th style={{ minWidth: 95 }}>Avg Years</th>
                                            <th style={{ minWidth: 190 }}>Email</th>
                                            <th style={{ minWidth: 145 }}>Phone</th>
                                            <th style={{ minWidth: 180 }}>Status</th>
                                            <th style={{ minWidth: 210 }}>Response</th>
                                            <th style={{ minWidth: 190 }}>Notes</th>
                                            <th style={{ minWidth: 64, width: 64 }}>Remove</th>
                                            {dynamicRoleAiCols.map(col => (
                                                <th key={col.key} style={{ minWidth: 220 }}>
                                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                                                            <span style={{ color: '#0f172a', fontWeight: 800, textTransform: 'none', fontSize: 12 }}>
                                                                {col.label}
                                                            </span>
                                                            {col.isPrimaryOutput && (
                                                                <span style={{ display: 'flex', gap: 4 }}>
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => rerunRoleAiColumn(col.definition)}
                                                                        style={{ width: 24, height: 24, borderRadius: 8, border: '1px solid #bbf7d0', background: '#f0fdf4', color: '#16a34a', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                                                                        title={selectedIds.size > 0 ? `Run on ${selectedIds.size} selected row(s)` : 'Select rows first, then click to run'}
                                                                    >
                                                                        <Play size={11} fill="currentColor" />
                                                                    </button>
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => setAiColumnModal({ mode: 'edit', definition: col.definition })}
                                                                        style={{ width: 24, height: 24, borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff', color: '#64748b', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                                                                        title="Edit smart column"
                                                                    >
                                                                        <Edit2 size={11} />
                                                                    </button>
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => deleteRoleAiColumn(col.definition)}
                                                                        style={{ width: 24, height: 24, borderRadius: 8, border: '1px solid #fee2e2', background: '#fff', color: '#ef4444', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                                                                        title="Delete smart column"
                                                                    >
                                                                        <Trash2 size={11} />
                                                                    </button>
                                                                </span>
                                                            )}
                                                        </div>
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                                            <div style={{ flex: 1, height: 6, borderRadius: 999, background: '#e2e8f0', overflow: 'hidden' }}>
                                                                <div
                                                                    style={{
                                                                        width: `${col.definition?.latest_run?.total ? Math.min(100, (((Number(col.definition.latest_run.completed || 0) + Number(col.definition.latest_run.failed || 0) + Number(col.definition.latest_run.skipped || 0)) / Number(col.definition.latest_run.total || 1)) * 100)) : 0}%`,
                                                                        height: '100%',
                                                                        background: col.definition?.latest_run?.status === 'completed_with_errors' ? '#f59e0b' : '#22c55e',
                                                                    }}
                                                                />
                                                            </div>
                                                            <span style={{ fontSize: 10, color: '#64748b', textTransform: 'none' }}>
                                                                {summarizeAiRun(col.definition?.latest_run)}
                                                            </span>
                                                        </div>
                                                    </div>
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredCandidates.map((candidate, idx) => {
                                            const roleOutreach = outreachStatus[candidate.id] || {}
                                            const responseState = candidateResponseSnapshot(candidate, roleOutreach, { countsReady: outreachCountsLoaded })
                                            const phone = candidate.phone || candidate.mobile_phone || ''
                                            return (
                                                <tr key={candidate.id || idx}>
                                                    <td style={{ textAlign: 'center' }}>
                                                        <input
                                                            type="checkbox"
                                                            checked={allFilteredSelected || selectedIds.has(candidate.id)}
                                                            onChange={() => toggleSelection(candidate.id)}
                                                            style={{ cursor: 'pointer', transform: 'scale(1.1)' }}
                                                        />
                                                    </td>
                                                    <td className="role-sticky-first-name">
                                                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                                                            <span
                                                                className={candidate.added_to_role_at ? 'hy-tooltip hy-tooltip-delayed hy-tooltip-left' : undefined}
                                                                data-tooltip={candidate.added_to_role_at ? addedTooltip(candidate.added_to_role_at) : undefined}
                                                            >
                                                                {candidate.first_name || candidate.name?.split(' ')[0] || '—'}
                                                            </span>
                                                            {isRecentlyAdded(candidate.added_to_role_at) && (
                                                                <span
                                                                    className="hy-tooltip"
                                                                    data-tooltip={addedTooltip(candidate.added_to_role_at)}
                                                                    style={{
                                                                        padding: '1px 6px', borderRadius: 5,
                                                                        background: '#ecfdf5', border: '1px solid #a7f3d0',
                                                                        fontSize: 10, fontWeight: 700, color: '#047857',
                                                                        letterSpacing: '0.03em', cursor: 'default',
                                                                    }}
                                                                >
                                                                    New
                                                                </span>
                                                            )}
                                                        </span>
                                                    </td>
                                                    <td className="role-sticky-last-name">{candidate.last_name || candidate.name?.split(' ').slice(1).join(' ') || '—'}</td>
                                                    <td className="role-truncate-cell" title={candidate.title || candidate.headline}>{candidate.title || candidate.headline || '—'}</td>
                                                    <td>
                                                        {candidate.linkedin ? (
                                                            <a href={candidate.linkedin} target="_blank" rel="noopener noreferrer" className="role-linkedin-button" title="Open LinkedIn profile">
                                                                <Linkedin size={15} /> Profile
                                                            </a>
                                                        ) : <span className="empty-val">Not linked</span>}
                                                    </td>
                                                    <td className="role-truncate-cell" title={candidate.company || candidate.current_company}>{candidate.company || candidate.current_company || '—'}</td>
                                                    <td className="role-truncate-cell" title={candidate.product_service}>{candidate.product_service || '—'}</td>
                                                    <td className="role-truncate-cell" title={candidate.city || candidate.location}>{candidate.city || candidate.location || '—'}</td>
                                                    <td>{Number(candidate.total_experience_years || 0) > 0 ? `${Number(candidate.total_experience_years).toFixed(1)}y` : '—'}</td>
                                                    <td>{Number(candidate.avg_tenure_years || candidate.avg_years_in_company || 0) > 0 ? `${Number(candidate.avg_tenure_years || candidate.avg_years_in_company).toFixed(1)}y` : '—'}</td>
                                                    <td>
                                                        {editingCell.candidateId === candidate.id && editingCell.field === 'email' ? (
                                                            <input
                                                                autoFocus
                                                                className="role-contact-edit-input"
                                                                type="email"
                                                                value={editValue}
                                                                onChange={e => setEditValue(e.target.value)}
                                                                onBlur={() => handleContactEdit(candidate.id, 'email', editValue)}
                                                                onKeyDown={e => {
                                                                    if (e.key === 'Enter') handleContactEdit(candidate.id, 'email', editValue)
                                                                    if (e.key === 'Escape') cancelEdit()
                                                                }}
                                                            />
                                                        ) : candidate.email ? (
                                                            <div className="role-contact-value">
                                                                <span title={candidate.email} style={{ cursor: 'text' }} onClick={() => startEdit(candidate.id, 'email', candidate.email)}>{candidate.email}</span>
                                                                <button type="button" className="icon-btn" onClick={() => handleCopy(candidate.email)} title="Copy email"><Copy size={12} /></button>
                                                            </div>
                                                        ) : (
                                                            <button type="button" className="role-contact-add" onClick={() => startEdit(candidate.id, 'email', '')}>+ Add email</button>
                                                        )}
                                                    </td>
                                                    <td>
                                                        {editingCell.candidateId === candidate.id && editingCell.field === 'mobile_phone' ? (
                                                            <input
                                                                autoFocus
                                                                className="role-contact-edit-input"
                                                                type="tel"
                                                                value={editValue}
                                                                onChange={e => setEditValue(e.target.value)}
                                                                onBlur={() => handleContactEdit(candidate.id, 'mobile_phone', editValue)}
                                                                onKeyDown={e => {
                                                                    if (e.key === 'Enter') handleContactEdit(candidate.id, 'mobile_phone', editValue)
                                                                    if (e.key === 'Escape') cancelEdit()
                                                                }}
                                                            />
                                                        ) : phone ? (
                                                            <div className="role-contact-value">
                                                                <span title={phone} style={{ cursor: 'text' }} onClick={() => startEdit(candidate.id, 'mobile_phone', phone)}>{phone}</span>
                                                                <button type="button" className="icon-btn" onClick={() => handleCopy(phone)} title="Copy phone"><Copy size={12} /></button>
                                                            </div>
                                                        ) : (
                                                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                                <button
                                                                    type="button"
                                                                    className="role-refresh-contact"
                                                                    onClick={() => handleRefreshProfile(candidate)}
                                                                    disabled={Boolean(refreshingProfileIds[candidate.id])}
                                                                >
                                                                    <RefreshCcw size={12} className={refreshingProfileIds[candidate.id] ? 'animate-spin' : ''} />
                                                                    {refreshingProfileIds[candidate.id] ? 'Fetching…' : 'Fetch contact'}
                                                                </button>
                                                                <button type="button" className="role-contact-add" onClick={() => startEdit(candidate.id, 'mobile_phone', '')}>+ Add</button>
                                                            </div>
                                                        )}
                                                    </td>
                                                    <td>
                                                        <StatusDropdown
                                                            status={candidate.status}
                                                            candidateId={candidate.id}
                                                            updateStatus={handleRoleStatusUpdate}
                                                            optimistic
                                                            onUpdate={(id, newStatus) => {
                                                                updateCandidateInRole(id, { status: newStatus });
                                                            }}
                                                        />
                                                    </td>
                                                    <td>
                                                        <button type="button" className="role-response-cell" onClick={() => {
                                                            if (responseState.hasUnread) {
                                                                useAppStore.getState().markOutreachResponseRead(viewingRole.id, candidate.id)
                                                            }
                                                            setChattingWith({
                                                                ...candidate,
                                                                response: roleOutreach.response_text || candidate.response || '',
                                                                li_response_text: roleOutreach.li_response_text || candidate.li_response_text || '',
                                                                li_status: roleOutreach.li_status || candidate.li_status || '',
                                                            })
                                                        }}>
                                                            <span>
                                                                <MessageSquare size={13} /> Open conversation
                                                                {responseState.messageCount == null ? (
                                                                    <b className="is-loading">…</b>
                                                                ) : responseState.messageCount > 0 ? (
                                                                    <b className={responseState.hasUnread ? 'is-unread' : ''}>{responseState.messageCount}</b>
                                                                ) : null}
                                                            </span>
                                                            <small title={responseState.responseText}>{responseState.responseText || 'No response yet'}</small>
                                                            {responseState.responseChannel && <em>{responseState.responseChannel}</em>}
                                                        </button>
                                                    </td>
                                                    <td><EditableCandidateNotes candidateId={candidate.id} initialNotes={candidate.notes} /></td>
                                                    <td style={{ textAlign: 'center' }}>
                                                        <button
                                                            type="button"
                                                            className="icon-btn role-remove-candidate"
                                                            onClick={event => { event.stopPropagation(); handleRemoveCandidate(candidate.id, candidate.name) }}
                                                            title="Remove from role"
                                                        >
                                                            <Trash2 size={16} />
                                                        </button>
                                                    </td>
                                                    {dynamicRoleAiCols.map(col => {
                                                        const cellsMap = col.definition?.cells_by_candidate || {}
                                                        const cell = cellsMap[candidate.id] ?? cellsMap[String(candidate.id)] ?? cellsMap[Number(candidate.id)]
                                                        const aiVal = resolveAiCellValue(cell, col.outputKey, col.isPrimaryOutput)
                                                        const st = cell?.status
                                                        const isEmpty = !aiVal && st !== 'running' && st !== 'queued'
                                                        return (
                                                            <td
                                                                key={col.key}
                                                                style={{ maxWidth: 240, verticalAlign: 'top', cursor: 'pointer' }}
                                                                onClick={() => openRoleAiCellDrawer(col.definition, candidate, col.outputKey)}
                                                            >
                                                                {st === 'running' || st === 'queued' ? (
                                                                    <div style={{
                                                                        fontSize: 11, color: '#2563eb', lineHeight: 1.55,
                                                                        background: 'linear-gradient(135deg, #eff6ff, #eef2ff)',
                                                                        border: '1px solid #bfdbfe', borderRadius: 10,
                                                                        padding: '8px 10px', fontWeight: 700,
                                                                    }}>
                                                                        {st === 'queued' ? 'Queued…' : 'Running…'}
                                                                    </div>
                                                                ) : st === 'failed' ? (
                                                                    <span style={{ color: '#b91c1c', fontSize: 11, lineHeight: 1.45, display: 'block' }}>
                                                                        {cell?.error_message || 'Run failed'}
                                                                    </span>
                                                                ) : st === 'skipped' ? (
                                                                    <span style={{ color: '#a16207', fontSize: 11, lineHeight: 1.45, display: 'block' }}>
                                                                        {cell?.error_message || 'Skipped'}
                                                                    </span>
                                                                ) : isEmpty ? (
                                                                    <span style={{ color: '#94a3b8', fontSize: 11, fontStyle: 'italic', lineHeight: 1.45 }}>
                                                                        {cell ? 'No value for this field (open for details)' : '—'}
                                                                    </span>
                                                                ) : (
                                                                    <div style={{
                                                                        fontSize: 12, color: '#1e1b4b', lineHeight: 1.55,
                                                                        whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                                                                        background: 'linear-gradient(135deg, #faf5ff, #eff6ff)',
                                                                        border: '1px solid #e0e7ff',
                                                                        borderRadius: 8, padding: '8px 10px',
                                                                        display: 'flex', flexDirection: 'column', gap: 6,
                                                                    }}>
                                                                        <div style={{ display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                                                                            {renderFriendlyAiValue(aiVal)}
                                                                        </div>
                                                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
                                                                            <span style={{ fontSize: 10, color: '#6366f1', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', whiteSpace: 'nowrap' }}>
                                                                                Click for full answer
                                                                            </span>
                                                                            {Array.isArray(cell?.details?.sources) && cell.details.sources.filter(s => s.url).length > 0 && (
                                                                                <a
                                                                                    href={cell.details.sources.find(s => s.url).url}
                                                                                    target="_blank"
                                                                                    rel="noreferrer"
                                                                                    onClick={e => e.stopPropagation()}
                                                                                    style={{ fontSize: 10, color: '#2563eb', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 3, background: '#dbeafe', padding: '2px 6px', borderRadius: 6, fontWeight: 700, whiteSpace: 'nowrap' }}
                                                                                >
                                                                                    <ExternalLink size={10} /> Source
                                                                                </a>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </td>
                                                        )
                                                    })}
                                                </tr>
                                            )
                                        })}
                                        {filteredCandidates.length === 0 && (
                                            <tr>
                                                <td colSpan={16 + dynamicRoleAiCols.length} style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                                                    No candidates match the selected filters.
                                                </td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                </div>
            </div>
                )}

                {chattingWith && (
                    <CandidateConversationModal
                        candidate={chattingWith}
                        roleId={viewingRole.id}
                        onClose={() => setChattingWith(null)}
                        updateStatus={handleRoleStatusUpdate}
                        onStatusChanged={(candidateId, status) => updateCandidateInRole(candidateId, { status })}
                    />
                )}

                {aiColumnModal && (
                    <AiColumnConfigModal
                        selectedIds={selectedIds}
                        initialDefinition={aiColumnModal?.mode === 'edit' ? aiColumnModal.definition : null}
                        viewScope={roleViewScope}
                        recruiterFilterId={null}
                        roleId={viewingRole?.id || null}
                        onClose={() => setAiColumnModal(null)}
                        onColumnDefinitionCreated={upsertOptimisticRoleAiColumn}
                        onColumnRunFailed={revertOptimisticRoleAiColumn}
                        onColumnsCreated={(runInfo) => {
                            attachRoleAiColumnRun(runInfo)
                            setAiColumnModal(null)
                            void fetchRoleAiColumns()
                        }}
                    />
                )}

                <AiColumnCellDrawer
                    open={aiCellDrawer.open}
                    loading={aiCellDrawer.loading}
                    detail={aiCellDrawer.detail}
                    title={aiCellDrawer.title}
                    onClose={() => setAiCellDrawer({ open: false, loading: false, detail: null, title: '' })}
                />


                {showAddToListModal && (
                    <AddToListModal
                        selectedCount={allFilteredSelected ? roleFilteredTotal : selectedIds.size}
                        onClose={() => setShowAddToListModal(false)}
                        onSuccess={() => {
                            setSelectedIds(new Set());
                            setAllFilteredSelected(false);
                            setShowAddToListModal(false);
                        }}
                        candidateIds={allFilteredSelected ? filteredCandidates.map(c => c.id) : Array.from(selectedIds)}
                    />
                )}

                {(selectedIds.size > 0 || allFilteredSelected) && (
                    <div style={{
                        position: 'fixed', bottom: 30, left: '50%', transform: 'translateX(-50%)',
                        background: '#0f172a', color: '#fff', padding: '12px 24px', borderRadius: '16px',
                        display: 'flex', alignItems: 'center', gap: 20, boxShadow: '0 20px 25px -5px rgba(0,0,0,0.3)',
                        zIndex: 1000, border: '1px solid rgba(255,255,255,0.1)', animation: 'slideUp 0.3s ease-out'
                    }}>
                        <span style={{ fontSize: 14, fontWeight: 600 }}>
                            {allFilteredSelected ? `${roleFilteredTotal} filtered candidates selected` : `${selectedIds.size} candidates selected`}
                        </span>
                        <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.2)' }} />
                        <button
                            onClick={() => setAllFilteredSelected(prev => !prev)}
                            style={{
                                padding: '8px 16px', background: allFilteredSelected ? '#312e81' : '#fff', color: allFilteredSelected ? '#fff' : '#0f172a', border: '1px solid rgba(255,255,255,0.18)',
                                borderRadius: '10px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
                                display: 'flex', alignItems: 'center', gap: 8, transition: 'all 0.2s'
                            }}
                        >
                            <Check size={14} /> {allFilteredSelected ? 'Using All Filtered' : `Use All Filtered (${roleFilteredTotal})`}
                        </button>
                        {(selectedIds.size > 0 || allFilteredSelected) && (
                            <button
                                onClick={() => setShowAddToListModal(true)}
                                style={{
                                    padding: '8px 16px', background: '#fff', color: '#0f172a', border: '1px solid rgba(255,255,255,0.18)',
                                    borderRadius: '10px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
                                    display: 'flex', alignItems: 'center', gap: 8, transition: 'all 0.2s'
                                }}
                                onMouseEnter={e => e.currentTarget.style.background = '#f8fafc'}
                                onMouseLeave={e => e.currentTarget.style.background = '#fff'}
                            >
                                <Phone size={14} /> Add to Call List
                            </button>
                        )}
                        <button
                            onClick={() => { setSelectedIds(new Set()); setAllFilteredSelected(false); }}
                            style={{ background: 'none', border: 'none', color: '#94a3b8', fontSize: 13, fontWeight: 600, cursor: 'pointer' }}
                        >
                            Cancel
                        </button>
                    </div>
                )}

            </div>

        )
    }

    // Role List View
    return (
        <div className="roles-page" style={{ width: '100%', position: 'relative', minHeight: '100vh', animation: 'fadeIn 0.2s ease-out' }}>
            <h2 className="screen-header">Role Management</h2>
            {uploadUi}

            <div className="result-banner">
                <div className="result-banner-title">
                    {(roles.length === 0 && !rolesLastFetchedAt) ? 'Loading...' : `${roles.length} Active Role(s)`}
                </div>
                <div className="result-banner-subtitle">Organize and manage your top talent by role</div>
            </div>

            {createModalOpen && <RoleCreateModal onClose={() => setCreateModalOpen(false)} onSubmit={handleCreateRole} />}
            {activationRole && <RoleCreateModal role={activationRole} onClose={() => setActivationRole(null)} onSubmit={handleActivateExisting} />}

            {/* Create Role */}
            <div className="quick-add-container">
                <div style={{ flex: 1, color: '#64748b', fontSize: 13 }}>Create the role and configure both outreach channels in one step.</div>
                <button
                    className="btn btn-primary"
                    onClick={() => setCreateModalOpen(true)}
                >
                    <Plus size={18} /> Add Role
                </button>
            </div>


            {/* Roles Grid */}
            <div className="roles-grid">
                {(roles.length === 0 && !rolesLastFetchedAt) ? (
                    <div className="empty-state" style={{ border: 'none', background: 'transparent' }}>
                        <Loader2 className="animate-spin" size={32} style={{ color: '#cbd5e1', marginBottom: '16px' }} />
                        <p style={{ color: '#94a3b8' }}>Loading roles...</p>
                    </div>
                ) : roles.length === 0 ? (
                    <div className="empty-state">
                        <Folder size={48} style={{ color: '#e2e8f0', marginBottom: '16px' }} />
                        <p>No recruitment roles created yet.</p>
                    </div>
                ) : (
                    roles.map(role => (
                        <div key={role.name} className="role-card">
                            <div className="role-card-main" onClick={() => openRoleView(role)}>
                                <div className="role-card-icon">
                                    <Folder size={20} />
                                </div>
                                <div className="role-card-content">
                                    <div className="role-card-name">{role.name}</div>
                                    <div className="role-card-meta">
                                        <User size={12} /> {role.candidate_count} assigned candidates
                                    </div>
                                    <div style={{ marginTop: 8, display: 'flex', gap: '12px', alignItems: 'center' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: 11, fontWeight: 600, color: role.smartlead_status === 'configured' ? '#15803d' : (role.smartlead_status === 'skipped' ? '#94a3b8' : '#b45309') }} title={`Email: ${role.smartlead_status || 'missing'}`}>
                                            <Mail size={12} /> {role.smartlead_status === 'configured' ? 'Active' : (role.smartlead_status === 'skipped' ? 'Skipped' : 'Inactive')}
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: 11, fontWeight: 600, color: role.heyreach_status === 'configured' ? '#15803d' : (role.heyreach_status === 'skipped' ? '#94a3b8' : '#b45309') }} title={`LinkedIn: ${role.heyreach_status || 'missing'}`}>
                                            <Linkedin size={12} /> {role.heyreach_status === 'configured' ? 'Active' : (role.heyreach_status === 'skipped' ? 'Skipped' : 'Inactive')}
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: 11, fontWeight: 600, color: role.has_call_list ? '#15803d' : '#94a3b8' }} title={`Call List: ${role.has_call_list ? 'Active' : 'Inactive'}`}>
                                            <Phone size={12} /> {role.has_call_list ? 'Active' : 'Inactive'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <button className="btn btn-secondary btn-sm" onClick={(event) => { event.stopPropagation(); handleRetryActivation(role) }} title={role.activation_error || 'Configure activation'} style={{ margin: '0 8px 8px' }}>{role.activation_status === 'active' ? 'Update config' : 'Configure & activate'}</button>
                            <button className="role-delete-btn" onClick={() => handleDeleteRole(role.name)} title="Remove Role">
                                <Trash2 size={16} />
                            </button>
                            <button
                                className="role-delete-btn"
                                onClick={() => openRoleUploadPicker(role)}
                                title="Upload CSV to role"
                                disabled={uploadPreviewBusy === role.name}
                                style={{ color: '#0f766e' }}
                            >
                                {uploadPreviewBusy === role.name ? <Loader2 size={16} className="animate-spin" /> : <FileUp size={16} />}
                            </button>
                        </div>
                    ))
                )}
            </div>
        </div>
    )
}

export default Roles
