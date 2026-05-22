import { create } from 'zustand'
import axios from 'axios'
import { toast } from 'sonner'

const REQUEST_TIMEOUT_MS = 15000
const CALL_REQUEST_TIMEOUT_MS = 15000
const CALL_RETRY_BACKOFF_MS = 3000

axios.defaults.timeout = REQUEST_TIMEOUT_MS

axios.interceptors.response.use(
    (response) => response,
    async (error) => {
        const config = error?.config || {}

        if (!error?.response && !config._retry && config.method?.toLowerCase() === 'get') {
            config._retry = true
            await new Promise(resolve => setTimeout(resolve, 1500))
            return axios(config)
        }

        if (error?.response?.status === 401 && localStorage.getItem('token')) {
            localStorage.removeItem('token')
            delete axios.defaults.headers.common['Authorization']
            if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
                window.location.assign('/login')
            }
        }

        return Promise.reject(error)
    }
)

export function getRequestErrorMessage(error, fallbackMessage) {
    if (error?.code === 'ECONNABORTED') {
        return 'Server is taking too long to respond'
    }

    if (!error?.response) {
        return 'Cannot reach the server'
    }

    const detail = error.response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) {
        return detail
    }
    if (Array.isArray(detail)) {
        const parts = detail.map((item) => {
            if (typeof item === 'string') return item
            if (item && typeof item.msg === 'string') return item.msg
            if (item && typeof item.message === 'string') return item.message
            return ''
        }).filter(Boolean)
        if (parts.length) {
            return parts.join('; ')
        }
    }
    if (detail && typeof detail === 'object') {
        if (typeof detail.message === 'string' && detail.message.trim()) {
            return detail.message
        }
        if (typeof detail.error === 'string' && detail.error.trim()) {
            return detail.error
        }
    }
    const status = error.response?.status
    if (status === 401 || status === 403) {
        return 'Not authorized — try signing in again'
    }
    if (status === 404) {
        return 'API endpoint not found — check API URL / proxy configuration'
    }
    if (status) {
        return `${fallbackMessage} (HTTP ${status})`
    }
    return fallbackMessage
}

function getRequestErrorDetail(error, fallbackMessage) {
    if (error?.code === 'ECONNABORTED') {
        return {
            message: 'Server is taking too long to respond',
            code: 'timeout',
            actionLabel: '',
            actionUrl: '',
            meta: null,
        }
    }

    if (!error?.response) {
        return {
            message: 'Cannot reach the server',
            code: 'network_unreachable',
            actionLabel: '',
            actionUrl: '',
            meta: null,
        }
    }

    const detail = error.response?.data?.detail
    if (detail && typeof detail === 'object') {
        return {
            message: detail.message || detail.error || fallbackMessage,
            code: detail.code || '',
            actionLabel: detail.action_label || '',
            actionUrl: detail.action_url || '',
            meta: detail.metadata || null,
        }
    }

    return {
        message: typeof detail === 'string' && detail.trim() ? detail : fallbackMessage,
        code: '',
        actionLabel: '',
        actionUrl: '',
        meta: null,
    }
}

function normalizeListName(name) {
    return (name || '').trim().toLowerCase()
}

function sortCallListsByCreatedAt(lists = []) {
    return [...lists].sort((left, right) => {
        const leftTime = new Date(left?.created_at || 0).getTime()
        const rightTime = new Date(right?.created_at || 0).getTime()
        return rightTime - leftTime
    })
}

function mergePendingCallLists(serverLists = [], currentLists = []) {
    const merged = [...(serverLists || [])]
    const pendingLists = (currentLists || []).filter(list => list?.is_pending)

    for (const pendingList of pendingLists) {
        const alreadyPresent = merged.some(serverList =>
            serverList?.id === pendingList?.id ||
            normalizeListName(serverList?.name) === normalizeListName(pendingList?.name)
        )
        if (!alreadyPresent) {
            merged.push(pendingList)
        }
    }

    return sortCallListsByCreatedAt(merged)
}

function updateCallInCollection(collection = [], updatedCall) {
    let found = false
    const next = (collection || []).map(call => {
        if (call?.id !== updatedCall?.id) {
            return call
        }
        found = true
        return { ...call, ...updatedCall }
    })

    return { next, found }
}

function patchCallAcrossCaches(callsCache = {}, updatedCall) {
    const nextCache = {}
    let found = false

    for (const [queryKey, entries] of Object.entries(callsCache || {})) {
        if (!Array.isArray(entries)) {
            nextCache[queryKey] = entries
            continue
        }

        const result = updateCallInCollection(entries, updatedCall)
        nextCache[queryKey] = result.next
        found = found || result.found
    }

    return { nextCache, found }
}

// API base URL
// On localhost/dev, always prefer same-origin `/api` to avoid stale external VITE_API_URL values
// causing cross-origin preflight failures.
let API_URL = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '')
const isLocalHost = typeof window !== 'undefined' &&
    /^(localhost|127\.0\.0\.1)(:\d+)?$/.test(window.location.host)

// Fallback to absolute production URL if env var failed to inject
if (!API_URL && !isLocalHost) {
    API_URL = 'https://growton-backend-v2-e3a3hxdmagfggcg9.centralindia-01.azurewebsites.net';
}

const useAbsoluteApi = /^https?:\/\//.test(API_URL) && !isLocalHost
export const API_BASE = useAbsoluteApi ? `${API_URL}/api` : '/api'

// Absolute base for OAuth and top-level redirects
export const BACKEND_BASE = isLocalHost ? 'http://127.0.0.1:8000' : (API_URL || window.location.origin)

// WebSocket URL
const BACKEND_HOST = useAbsoluteApi ? API_URL.replace(/^https?:\/\//, '') : window.location.host
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
const WS_URL = `${protocol}//${BACKEND_HOST}/api/ws/search`

// Global App Store
import { persist } from 'zustand/middleware'

const defaultCallStats = { due_today: 0, upcoming: 0, completed: 0, active_lists: 0 }

export const useAppStore = create(persist((set, get) => ({
    // Navigation
    currentPage: 'Screening',
    setCurrentPage: (page) => set({ currentPage: page }),

    // Auth State
    user: null,
    isAuthenticated: !!localStorage.getItem('token'),
    token: localStorage.getItem('token'),

    // Auth Actions
    register: async (name, email, phone, password) => {
        try {
            const response = await axios.post(`${API_BASE}/register`, { name, email, phone, password })
            return { success: true, message: response.data.message }
        } catch (error) {
            console.error('Registration failed:', error)
            return { success: false, error: getRequestErrorMessage(error, 'Registration failed') }
        }
    },

    verifyOtp: async (email, otp_code) => {
        try {
            const response = await axios.post(`${API_BASE}/verify-otp`, { email, otp_code })
            const { access_token, user: userData } = response.data

            localStorage.setItem('token', access_token)
            set({
                token: access_token,
                isAuthenticated: true,
                user: userData
            })

            axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
            return { success: true }
        } catch (error) {
            console.error('OTP Verification failed:', error)
            return { success: false, error: getRequestErrorMessage(error, 'Verification failed') }
        }
    },

    resendOtp: async (email) => {
        try {
            const response = await axios.post(`${API_BASE}/resend-otp`, { email })
            return { success: true, message: response.data.message }
        } catch (error) {
            console.error('Resend OTP failed:', error)
            return { success: false, error: getRequestErrorMessage(error, 'Failed to resend OTP') }
        }
    },

    login: async (email, password) => {
        try {
            const response = await axios.post(`${API_BASE}/login`, { email, password })
            const { access_token, user: userData } = response.data

            localStorage.setItem('token', access_token)
            set({
                token: access_token,
                isAuthenticated: true,
                user: userData
            })

            // Set default header
            axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`

            get().invalidateTalentPoolCaches({ clearRows: true })

            return { success: true }
        } catch (error) {
            console.error('Login failed:', error)
            return { success: false, error: getRequestErrorMessage(error, 'Login failed') }
        }
    },

    logout: () => {
        localStorage.removeItem('token')
        delete axios.defaults.headers.common['Authorization']
        set({ token: null, isAuthenticated: false, user: null })
        get().invalidateTalentPoolCaches({ clearRows: true })
    },

    // Google OAuth login
    loginWithGoogle: async (googleToken) => {
        try {
            const response = await axios.post(`${API_BASE}/auth/google`, { token: googleToken })
            const { access_token, user: userData } = response.data

            localStorage.setItem('token', access_token)
            set({
                token: access_token,
                isAuthenticated: true,
                user: userData
            })

            // Set default header
            axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
            get().invalidateTalentPoolCaches({ clearRows: true })
            return { success: true }
        } catch (error) {
            console.error('Google login failed:', error)
            return { success: false, error: getRequestErrorMessage(error, 'Google login failed') }
        }
    },

    fetchProfile: async () => {
        try {
            const res = await axios.get(`${API_BASE}/me`)
            set({ user: res.data, isAuthenticated: true })
            return { success: true }
        } catch (e) {
            console.error('Failed to fetch profile:', e)
            if (e.response?.status === 401) {
                get().logout()
            }
            return { success: false }
        }
    },

    // Admin Actions
    recruiters: [],
    fetchRecruiters: async () => {
        try {
            const res = await axios.get(`${API_BASE}/admin/recruiters`)
            set({ recruiters: res.data })
        } catch (e) {
            console.error('Failed to fetch recruiters:', e)
        }
    },
    createRecruiter: async (data) => {
        try {
            const res = await axios.post(`${API_BASE}/admin/recruiters`, data)
            set(state => ({ recruiters: [...state.recruiters, res.data] }))
            return { success: true }
        } catch (e) {
            return { success: false, error: e.response?.data?.detail }
        }
    },
    warmAll: async () => {
        try {
            // FIRE AND FORGET - Don't wait for these expensive calls to block UI mount
            axios.post(`${API_BASE}/admin/warm-all`);
            get().fetchCallStats({ force: true });
            get().fetchCalls({ due_filter: 'today', status: 'pending' }, { force: true });
            get().fetchAnalytics();
            return { success: true }
        } catch (e) {
            console.error('Failed to trigger warm-all:', e)
            return { success: false }
        }
    },
    updateRecruiterPermissions: async (id, permissions) => {
        // Optimistic Update
        const previousRecruiters = get().recruiters;
        set(state => ({
            recruiters: state.recruiters.map(r => r.id === id ? { ...r, permissions } : r)
        }))

        try {
            await axios.patch(`${API_BASE}/admin/recruiters/${id}/permissions`, permissions)
            // Removed disruptive server sync payload override for smoother UI
            return { success: true }
        } catch (e) {
            // Rollback on error
            set({ recruiters: previousRecruiters });
            return { success: false, error: e.response?.data?.detail }
        }
    },
    updateCandidateNotes: async (id, notes) => {
        try {
            await axios.patch(`${API_BASE}/candidates/${id}/notes`, { notes });

            // Update cache
            const cache = get().talentPoolCache;
            if (cache && cache.candidates) {
                const updatedCandidates = cache.candidates.map(c =>
                    c.id === id ? { ...c, notes } : c
                );
                set({ talentPoolCache: { ...cache, candidates: updatedCandidates } });
            }
            return { success: true };
        } catch (e) {
            console.error('Failed to update notes:', e);
            return { success: false, error: e.response?.data?.detail || 'Failed to update notes' };
        }
    },
    deleteRecruiter: async (id) => {
        const prevRecruiters = [...get().recruiters]
        set((state) => ({ recruiters: state.recruiters.filter(r => r.id !== id) }))

        try {
            await axios.delete(`${API_BASE}/admin/recruiters/${id}`)
            return { success: true }
        } catch (error) {
            set({ recruiters: prevRecruiters })
            console.error('Failed to delete recruiter:', error)
            return { success: false, error: 'Failed' }
        }
    },

    // Stats
    stats: {
        total_candidates: 0,
        avg_experience: 0,
        total_companies: 0,
        total_roles: 0
    },
    fetchStats: async () => {
        try {
            const res = await axios.get(`${API_BASE}/stats`)
            set({ stats: res.data })
        } catch (e) {
            console.error('Failed to fetch stats:', e)
        }
    },

    // Detailed Analytics (Pipeline + Recruiter Perf)
    analytics: null,
    analyticsLastFetchedAt: 0,
    analyticsRequest: null,
    fetchAnalytics: async (options = {}) => {
        const force = typeof options === 'boolean' ? options : options.force === true
        const state = get()
        const freshnessMs = 60 * 1000

        if (state.analytics && !force) {
            if (!state.analyticsRequest && (!state.analyticsLastFetchedAt || Date.now() - state.analyticsLastFetchedAt >= freshnessMs)) {
                setTimeout(() => { get().fetchAnalytics({ force: true }) }, 0)
            }
            return { success: true, data: state.analytics, cached: true }
        }

        if (state.analyticsRequest) {
            return state.analyticsRequest
        }

        const request = axios.get(`${API_BASE}/candidates/analytics`, { timeout: 60000 })
            .then(res => {
                set({
                    analytics: res.data,
                    analyticsLastFetchedAt: Date.now(),
                    analyticsRequest: null
                })
                return { success: true, data: res.data, cached: false }
            })
            .catch(e => {
                console.error('Failed to fetch analytics:', e)
                set({ analyticsRequest: null })
                const fallback = get().analytics
                if (fallback) return { success: true, data: fallback, cached: true }
                return { success: false, error: e.response?.data?.detail || 'Failed to fetch analytics' }
            })

        set({ analyticsRequest: request })
        return request
    },

    // HeyReach Global State
    heyreachCampaignId: '332760',
    setHeyreachCampaignId: (id) => set({ heyreachCampaignId: id }),

    lookupHeyReachCampaign: async (name) => {
        try {
            const res = await axios.get(`${API_BASE}/outreach/heyreach/find-campaign/${encodeURIComponent(name)}`)
            if (res.data.campaign_id) {
                set({ heyreachCampaignId: res.data.campaign_id.toString() })
                return { success: true, campaign_id: res.data.campaign_id }
            }
            return { success: false, error: 'No campaign found' }
        } catch (e) {
            console.error('Failed to lookup campaign:', e)
            return { success: false, error: e.response?.data?.detail || 'Lookup failed' }
        }
    },

    // Sidebar State
    sidebarWidth: 260,
    isSidebarCollapsed: false,
    setSidebarWidth: (w) => set({ sidebarWidth: Math.max(72, Math.min(420, w)), isSidebarCollapsed: w < 120 }),
    toggleSidebar: () => {
        const { isSidebarCollapsed, sidebarWidth } = get()
        if (isSidebarCollapsed) {
            set({ isSidebarCollapsed: false, sidebarWidth: 240 })
        } else {
            set({ isSidebarCollapsed: true, sidebarWidth: 72 })
        }
    },

    updateCandidateField: async (candidateId, data) => {
        try {
            const res = await axios.patch(`${API_BASE}/candidates/${candidateId}`, data)
            const normalizedPatch = { ...data }
            if (Object.prototype.hasOwnProperty.call(data, 'phone')) {
                normalizedPatch.phone = data.phone
                normalizedPatch.mobile_phone = data.phone
            }

            const patchCandidate = (candidate) => (
                candidate?.id === candidateId ? { ...candidate, ...normalizedPatch } : candidate
            )

            set(state => ({
                tpCandidates: (state.tpCandidates || []).map(patchCandidate),
                talentPoolCache: state.talentPoolCache?.data
                    ? {
                        ...state.talentPoolCache,
                        data: {
                            ...state.talentPoolCache.data,
                            candidates: (state.talentPoolCache.data.candidates || []).map(patchCandidate),
                        },
                    }
                    : state.talentPoolCache,
                talentPoolIndex: {
                    ...(state.talentPoolIndex || { rows: [], lastFetchedAt: 0 }),
                    rows: (state.talentPoolIndex?.rows || []).map(patchCandidate),
                },
            }))
            return { success: true, data: res.data }
        } catch (e) {
            console.error('Failed to update candidate field:', e)
            return { success: false, error: e.response?.data?.detail || 'Update failed' }
        }
    },

    // Search State
    searchQuery: '',
    setSearchQuery: (query) => set({ searchQuery: query }),
    isSearching: false,
    searchProgress: 0,
    searchTotal: 0,
    statusMessage: '',
    searchResults: [],
    usage: null,
    searchOutcome: 'idle',
    lastSearchError: '',
    _searchFallbackTimer: null,

    // Search via REST (synchronous)
    searchCandidates: async (query, options = {}) => {
        const initialStatus = options.initialStatus || 'Screening...'
        set({
            isSearching: true,
            searchResults: [],
            usage: null,
            statusMessage: initialStatus,
            searchProgress: 0,
            searchTotal: 0,
            searchOutcome: 'loading',
            lastSearchError: '',
        })
        try {
            const res = await axios.post(`${API_BASE}/search`, { query })
            const totalCandidates = res.data.total ?? res.data.candidates?.length ?? 0
            set({
                searchResults: res.data.candidates,
                usage: res.data.usage,
                statusMessage: totalCandidates > 0 ? `Found ${totalCandidates} candidates` : 'No close matches found yet',
                isSearching: false,
                searchProgress: 100,
                searchOutcome: totalCandidates > 0 ? 'success' : 'empty',
            })
        } catch (e) {
            const errorMessage = getRequestErrorMessage(e, 'Screening failed')
            set({
                statusMessage: 'Screening failed: ' + errorMessage,
                isSearching: false,
                searchOutcome: 'error',
                lastSearchError: errorMessage,
            })
        } finally {
            const fallbackTimer = get()._searchFallbackTimer
            if (fallbackTimer) {
                window.clearTimeout(fallbackTimer)
            }
            set({ _ws: null, _searchFallbackTimer: null })
        }
    },

    // Search via WebSocket (streaming)
    searchCandidatesStream: (query) => {
        const existingWs = get()._ws
        if (existingWs) {
            try {
                existingWs.close()
            } catch (_) { }
        }

        const existingFallbackTimer = get()._searchFallbackTimer
        if (existingFallbackTimer) {
            window.clearTimeout(existingFallbackTimer)
        }

        set({
            isSearching: true,
            searchResults: [],
            usage: null,
            statusMessage: 'Connecting...',
            searchProgress: 0,
            searchTotal: 0,
            searchOutcome: 'loading',
            lastSearchError: '',
            _searchFallbackTimer: null,
        })

        const ws = new WebSocket(WS_URL)
        let finished = false

        const clearFallbackTimer = () => {
            const timerId = get()._searchFallbackTimer
            if (timerId) {
                window.clearTimeout(timerId)
                set({ _searchFallbackTimer: null })
            }
        }

        const fallbackToRest = () => {
            if (finished || !get().isSearching) {
                return
            }
            finished = true
            clearFallbackTimer()
            try {
                ws.close()
            } catch (_) { }
            set({ _ws: null })
            get().searchCandidates(query, {
                initialStatus: 'Realtime screening unavailable. Running standard search...'
            })
        }

        const armFallbackTimer = (delayMs) => {
            clearFallbackTimer()
            const timerId = window.setTimeout(() => {
                fallbackToRest()
            }, delayMs)
            set({ _searchFallbackTimer: timerId })
        }

        armFallbackTimer(8000)

        ws.onopen = () => {
            if (finished) return
            const token = get().token
            ws.send(JSON.stringify({ query, token }))
            set({ statusMessage: 'Processing query...' })
            armFallbackTimer(12000)
        }

        ws.onmessage = (event) => {
            if (finished) return

            const data = JSON.parse(event.data)
            armFallbackTimer(20000)

            switch (data.type) {
                case 'status':
                    set({ statusMessage: data.message })
                    break

                case 'progress_start':
                    set({ searchTotal: data.total, statusMessage: `Found ${data.total} potential matches. Generating summaries...` })
                    break

                case 'candidate':
                    set(state => ({
                        searchResults: [...state.searchResults, data.data],
                        searchProgress: Math.round((data.current / data.total) * 100)
                    }))
                    break

                case 'complete':
                    finished = true
                    clearFallbackTimer()
                    const totalCandidates = data.total ?? data.candidates?.length ?? 0
                    set({
                        searchResults: data.candidates,
                        usage: data.usage,
                        statusMessage: totalCandidates > 0 ? `Found ${totalCandidates} candidates` : 'No close matches found yet',
                        isSearching: false,
                        searchProgress: 100,
                        searchOutcome: totalCandidates > 0 ? 'success' : 'empty',
                    })
                    set({ _ws: null })
                    ws.close()
                    break

                case 'error':
                    finished = true
                    clearFallbackTimer()
                    set({
                        statusMessage: 'Error: ' + data.message,
                        isSearching: false,
                        searchOutcome: 'error',
                        lastSearchError: data.message,
                    })
                    set({ _ws: null })
                    ws.close()
                    break
            }
        }

        ws.onerror = (error) => {
            console.error('WebSocket error:', error)
            fallbackToRest()
        }

        ws.onclose = () => {
            clearFallbackTimer()
            if (finished) {
                set({ _ws: null })
                return
            }
            if (get().isSearching) {
                fallbackToRest()
            }
        }

        // Store ws reference for potential cancellation
        set({ _ws: ws })
    },

    stopSearch: () => {
        const ws = get()._ws
        const fallbackTimer = get()._searchFallbackTimer
        if (ws) {
            ws.close()
        }
        if (fallbackTimer) {
            window.clearTimeout(fallbackTimer)
        }
        set({
            isSearching: false,
            statusMessage: 'Screening stopped',
            searchOutcome: 'cancelled',
            lastSearchError: '',
            _ws: null,
            _searchFallbackTimer: null,
        })
    },

    clearSearch: () => {
        const ws = get()._ws
        const fallbackTimer = get()._searchFallbackTimer
        if (ws) {
            try {
                ws.close()
            } catch (_) { }
        }
        if (fallbackTimer) {
            window.clearTimeout(fallbackTimer)
        }
        set({
            searchQuery: '',
            searchResults: [],
            statusMessage: '',
            searchProgress: 0,
            searchTotal: 0,
            usage: null,
            isSearching: false,
            searchOutcome: 'idle',
            lastSearchError: '',
            _ws: null,
            _searchFallbackTimer: null,
        })
    },

    // Selection State
    selectedCandidates: {},
    candidatePriorities: {},
    candidateFeedback: {},

    // AI Columns State
    aiColumns: [],
    setAiColumns: (columns) => set(state => ({ aiColumns: typeof columns === 'function' ? columns(state.aiColumns) : columns })),
    aiColumnsLoading: false,
    setAiColumnsLoading: (loading) => set({ aiColumnsLoading: loading }),

    toggleCandidateSelection: (id) => {
        set(state => {
            const newSelected = { ...state.selectedCandidates }
            if (newSelected[id]) {
                delete newSelected[id]
            } else {
                const candidate = state.searchResults.find(c => c.id === id)
                if (candidate) {
                    newSelected[id] = candidate
                    // Trigger Enrichment in Background
                    axios.post(`${API_BASE}/enrich/${id}`).catch(err => console.error("Enrichment trigger failed", err))
                }
            }
            return { selectedCandidates: newSelected }
        })
    },

    setCandidatePriority: (id, priority) => {
        set(state => ({
            candidatePriorities: { ...state.candidatePriorities, [id]: priority }
        }))
    },

    setCandidateFeedback: (id, feedback) => {
        set(state => ({
            candidateFeedback: { ...state.candidateFeedback, [id]: feedback }
        }))
    },

    clearSelections: () => {
        set({ selectedCandidates: {}, candidatePriorities: {}, candidateFeedback: {} })
    },

    // Roles State
    roles: [],
    rolesLastFetchedAt: 0,
    rolesRequest: null,
    viewingRole: null,
    roleDetailsCache: {},

    fetchRoles: async (options = {}) => {
        const force = typeof options === 'boolean' ? options : options.force === true
        const maxAgeMs = 60 * 1000
        const state = get()
        const isFresh = state.roles.length > 0 && state.rolesLastFetchedAt && (Date.now() - state.rolesLastFetchedAt < maxAgeMs)

        if (!force && isFresh) {
            return { success: true, data: state.roles, cached: true }
        }

        if (state.rolesRequest) {
            return state.rolesRequest
        }

        const request = axios.get(`${API_BASE}/roles`)
            .then(res => {
                set({
                    roles: res.data.roles,
                    rolesLastFetchedAt: Date.now(),
                    rolesRequest: null
                })
                return { success: true, data: res.data.roles, cached: false }
            })
            .catch(error => {
                console.error('Failed to fetch roles:', error)
                set({ rolesRequest: null })
                const fallback = get().roles
                if (fallback.length > 0) return { success: true, data: fallback, cached: true }
                return { success: false, error: error.response?.data?.detail || 'Failed to fetch roles' }
            })

        set({ rolesRequest: request })
        return request
    },

    openRole: (role) => {
        // Check cache for instant access
        const cachedRole = get().roleDetailsCache[role.name]

        if (cachedRole) {
            // Instant render with full cached data
            set({ viewingRole: cachedRole })
        } else {
            // Fallback to shell
            set({ viewingRole: { ...role, candidates: [] } })
        }

        // Fetch full details in background (always refresh)
        get()._fetchRoleDetailsBackground(role.name)
    },

    _fetchRoleDetailsBackground: async (roleName) => {
        // Simple single fetch - no aggressive polling
        try {
            const res = await axios.get(`${API_BASE}/roles/${encodeURIComponent(roleName)}`)
            const updatedRole = res.data

            // Update cache
            set(state => ({
                roleDetailsCache: {
                    ...state.roleDetailsCache,
                    [roleName]: updatedRole
                }
            }))

            // Update view if still relevant
            if (get().viewingRole?.name === roleName) {
                set({ viewingRole: updatedRole })
            }
        } catch (error) {
            // Handle deleted roles gracefully
            if (error.response?.status === 404) {
                console.warn(`Role "${roleName}" not found (may have been deleted)`)
                // Clear the deleted role from cache
                const newCache = { ...get().roleDetailsCache }
                delete newCache[roleName]
                set({ roleDetailsCache: newCache })

                // If we're currently viewing this role, clear the view
                if (get().viewingRole?.name === roleName) {
                    set({ viewingRole: null })
                }
            } else {
                console.error('Failed to fetch role details:', error)
            }
        }
    },

    // Kept for direct calls if needed, but openRole is preferred for UI
    fetchRoleDetails: async (roleName) => {
        return get()._fetchRoleDetailsBackground(roleName)
    },

    createRole: async (name) => {
        const newRole = { id: Date.now(), name, candidate_count: 0 }
        const prevRoles = [...get().roles]

        // Optimistic Update
        set({ roles: [newRole, ...prevRoles] })

        try {
            const res = await axios.post(`${API_BASE}/roles`, { name })
            // Refresh with real server data in background
            get().fetchRoles({ force: true })
            return { success: true, data: res.data }
        } catch (error) {
            // Rollback on error
            set({ roles: prevRoles })
            console.error('Failed to create role:', error)
            return { success: false, error: error.response?.data?.detail || 'Failed to create role' }
        }
    },

    deleteRole: async (roleName) => {
        const prevRoles = [...get().roles]

        // Optimistic Update
        set({ roles: prevRoles.filter(r => r.name !== roleName) })

        // Clear cache for deleted role
        const prevCache = { ...get().roleDetailsCache }
        delete prevCache[roleName]
        set({ roleDetailsCache: prevCache })

        try {
            await axios.delete(`${API_BASE}/roles/${encodeURIComponent(roleName)}`)
            return { success: true }
        } catch (error) {
            // Rollback on error
            set({ roles: prevRoles, roleDetailsCache: get().roleDetailsCache })
            console.error('Failed to delete role:', error)
            return { success: false, error: error.response?.data?.detail || 'Failed to delete role' }
        }
    },

    clearViewingRole: () => set({ viewingRole: null }),

    assignCandidatesToRole: async (roleName, assignments) => {
        // 1. Optimistically update candidate counts in roles list
        const prevRoles = [...get().roles]
        const updatedRoles = prevRoles.map(r =>
            r.name === roleName
                ? { ...r, candidate_count: r.candidate_count + assignments.length }
                : r
        )
        set({ roles: updatedRoles })

        // 2. Identify the full candidate objects from our current state (searchResults or cache)
        // We look in searchResults to find the full profile data for the assigned IDs
        const candidatesToAdd = assignments.map(assign => {
            const fullProfile = get().searchResults.find(c => c.id === assign.candidate_id) || {}
            return {
                ...fullProfile,
                ...assign, // Adds priority, feedback
                // Add placeholder/loading state for contact info if needed, or keep existing
                email: fullProfile.email || null,
                mobile_phone: fullProfile.mobile_phone || null
            }
        })

        // 3. Update the Cache immediately (User-perceived instant assignment)
        const prevCache = { ...get().roleDetailsCache }
        const existingCachedRole = prevCache[roleName] || { name: roleName, candidates: [] }

        const newCachedRole = {
            ...existingCachedRole,
            candidates: [...(existingCachedRole.candidates || []), ...candidatesToAdd]
        }

        prevCache[roleName] = newCachedRole
        set({ roleDetailsCache: prevCache })

        // 4. If currently viewing this role, update the view immediately
        if (get().viewingRole?.name === roleName) {
            set({ viewingRole: newCachedRole })
        }

        try {
            const res = await axios.post(`${API_BASE}/roles/${encodeURIComponent(roleName)}/assign`, {
                assignments
            })

            // Refresh roles immediately to get correct counts from server
            const rolesResult = await get().fetchRoles({ force: true })
            console.log('Refreshed roles after assignment:', rolesResult)

            // Background fetch to get Clay enriched data (emails/phones)
            // This will silently update the view/cache when data arrives
            get()._fetchRoleDetailsBackground(roleName)

            return { success: true, data: res.data }
        } catch (error) {
            // Rollback on fully failed request
            set({ roles: prevRoles, roleDetailsCache: get().roleDetailsCache }) // Revert cache?? Ideally revert to exact prev state but complex
            return { success: false, error: error.response?.data?.detail || 'Failed to assign candidates' }
        }
    },

    removeCandidateFromRole: async (roleName, candidateId) => {
        // Optimistic update
        const prevCache = { ...get().roleDetailsCache }
        const prevViewingRole = get().viewingRole
        const prevRoles = [...get().roles]

        // 1. Update candidate count in roles list
        const updatedRoles = prevRoles.map(r =>
            r.name === roleName
                ? { ...r, candidate_count: Math.max(0, r.candidate_count - 1) }
                : r
        )
        set({ roles: updatedRoles })

        // 2. Remove from cache
        if (prevCache[roleName] && prevCache[roleName].candidates) {
            const newCachedRole = {
                ...prevCache[roleName],
                candidates: prevCache[roleName].candidates.filter(c => c.id !== candidateId)
            }
            prevCache[roleName] = newCachedRole
            set({ roleDetailsCache: prevCache })
        }

        // 3. Remove from current view
        if (prevViewingRole?.name === roleName) {
            set({
                viewingRole: {
                    ...prevViewingRole,
                    candidates: prevViewingRole.candidates.filter(c => c.id !== candidateId)
                }
            })
        }

        try {
            const res = await axios.delete(`${API_BASE}/roles/${encodeURIComponent(roleName)}/candidates/${candidateId}`)
            return { success: true }
        } catch (error) {
            // Rollback optimistic update
            set({ roles: prevRoles, roleDetailsCache: get().roleDetailsCache, viewingRole: prevViewingRole })
            console.error('Failed to remove candidate:', error)
            return { success: false, error: error.response?.data?.detail || 'Failed to remove candidate' }
        }
    },

    // Outreach Status Cache
    outreachStatusCache: {},

    fetchOutreachStatus: async (roleId) => {
        // Return cached if available (instant render)
        const cached = get().outreachStatusCache[roleId]

        try {
            const res = await axios.get(`${API_BASE}/outreach/status/${roleId}`)
            const newData = res.data

            // Update cache
            set(state => ({
                outreachStatusCache: {
                    ...state.outreachStatusCache,
                    [roleId]: newData
                }
            }))
            return newData
        } catch (error) {
            console.error('Failed to fetch outreach status:', error)
            return cached || {} // Return cached or empty on error
        }
    },

    syncOutreachResponses: async (roleId) => {
        try {
            const res = await axios.post(`${API_BASE}/outreach/sync-responses/${roleId}`)

            // Invalidate Talent Pool cache because browse reads from backend's in-memory profile cache.
            if (roleId === 0) {
                set({
                    talentPoolCache: { data: null, lastParamsString: null }
                })
            }

            return { success: true, data: res.data }
        } catch (error) {
            console.error('Failed to sync outreach responses:', error)
            return { success: false, error: getRequestErrorMessage(error, 'Failed to sync outreach responses') }
        }
    },

    triggerHeyReachOutreach: async (payload) => {
        try {
            const res = await axios.post(`${API_BASE}/outreach/heyreach/trigger`, payload)
            return { success: true, data: res.data }
        } catch (error) {
            console.error('Failed to trigger HeyReach outreach:', error)
            return { success: false, error: error.response?.data?.detail || 'Failed to trigger HeyReach outreach' }
        }
    },

    fetchChatHistory: async (roleId, candidateId, platform = 'email', force = false) => {
        try {
            const endpoint = platform === 'linkedin' ? 'linkedin' : 'email'
            const forceParam = force ? '&force=true' : ''
            const res = await axios.get(`${API_BASE}/outreach/chat/${endpoint}/${roleId}/${candidateId}?cb=${Date.now()}${forceParam}`, {
                headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache', 'Expires': '0' }
            })
            return { success: true, messages: res.data.messages, syncing: res.data.syncing || false }

        } catch (error) {
            console.error(`Failed to fetch ${platform} chat history:`, error)
            return { success: false, error: error.response?.data?.detail || `Failed to fetch ${platform} chat history` }
        }
    },

    prewarmLinkedInCache: async (candidateIds) => {
        try {
            if (!candidateIds || candidateIds.length === 0) return { success: true }
            const res = await axios.post(`${API_BASE}/outreach/prewarm/linkedin`, { candidate_ids: candidateIds })
            return { success: true, data: res.data }
        } catch (error) {
            console.error(`Failed to prewarm LinkedIn cache:`, error)
            return { success: false, error: error.response?.data?.detail || `Failed to prewarm LinkedIn cache` }
        }
    },

    sendChatReply: async (roleId, candidateId, message, platform = 'email') => {
        try {
            const endpoint = platform === 'linkedin' ? 'linkedin' : 'email'
            const res = await axios.post(`${API_BASE}/outreach/reply/${endpoint}/${roleId}/${candidateId}`, { message })
            return { success: true, data: res.data }
        } catch (error) {
            console.error(`Failed to send ${platform} reply:`, error)
            return { success: false, error: error.response?.data?.detail || `Failed to send ${platform} reply` }
        }
    },

    shortlistAndOutreach: async (candidateId, options = {}) => {
        try {
            const res = await axios.post(`${API_BASE}/outreach/shortlist/${candidateId}`, options)
            return { success: true, data: res.data }
        } catch (error) {
            console.error('Failed to trigger shortlist outreach:', error)
            return { success: false, error: error.response?.data?.detail || 'Outreach trigger failed' }
        }
    },


    // Screen step (for screening workflow)
    screenStep: 1,
    setScreenStep: (step) => set({ screenStep: step }),

    // Talent Pool Cache & Persistent State
    tpCandidates: [],
    tpTotal: 0,
    tpTotalPages: 1,
    tpStatusCounts: {},
    tpFilters: {
        title: [], titleInput: '',
        company: [], companyInput: '',
        city: [], cityInput: '',
        product_service: [], productInput: '',
        status: '', created_by: '',
        min_exp: 0, max_exp: 40,
    },
    tpActiveStatusTab: '',
    tpSortBy: 'name',
    tpSortDir: 'asc',
    tpPage: 1,
    tpPageSize: 25,
    tpGlobalSearch: '',

    setTpFilters: (updater) => set((state) => ({
        tpFilters: typeof updater === 'function' ? updater(state.tpFilters) : updater
    })),
    setTpActiveStatusTab: (tab) => set({ tpActiveStatusTab: tab }),
    setTpPagination: (page, pageSize) => set({ tpPage: page, tpPageSize: pageSize }),
    setTpSort: (sortBy, sortDir) => set({ tpSortBy: sortBy, tpSortDir: sortDir }),
    setTpGlobalSearch: (q) => set({ tpGlobalSearch: q }),
    setTpCandidates: (candidates) => set({ tpCandidates: candidates || [] }),
    setTpStatusCounts: (counts) => set({ tpStatusCounts: counts || {} }),
    updateTpCandidate: (candidateId, data) => set(state => ({
        tpCandidates: (state.tpCandidates || []).map(c => c.id === candidateId ? { ...c, ...data } : c),
        talentPoolIndex: {
            ...state.talentPoolIndex,
            rows: (state.talentPoolIndex?.rows || []).map(c => c.id === candidateId ? { ...c, ...data } : c)
        }
    })),

    talentPoolCache: { data: null, lastParamsString: null },
    talentPoolRequest: null,
    talentPoolRequestParamsString: '',
    talentPoolRequestSeq: 0,
    talentPoolIndex: { rows: [], lastFetchedAt: 0, lastParamsString: '' },
    talentPoolIndexRequest: null,
    talentPoolIndexRequestParamsString: '',
    talentPoolIndexRequestSeq: 0,
    talentPoolViewScope: 'master',
    talentPoolRecruiterFilterId: null,
    talentPoolRoleFilterId: '',

    buildTalentPoolScopeQuery: () => {
        const u = get().user
        const parts = []
        if (u?.role === 'admin') {
            const vs = get().talentPoolViewScope || 'master'
            parts.push(`view_scope=${encodeURIComponent(vs)}`)
            if (vs === 'recruiter_pools') {
                const rid = get().talentPoolRecruiterFilterId
                if (rid) parts.push(`recruiter_filter_id=${encodeURIComponent(rid)}`)
            }
        }
        const roleId = get().talentPoolRoleFilterId
        if (roleId) parts.push(`role_id=${encodeURIComponent(roleId)}`)
        return parts.join('&')
    },

    buildTalentPoolQueryKey: (paramsString = '') => {
        const scopeQ = get().buildTalentPoolScopeQuery()
        return [paramsString, scopeQ].filter(Boolean).join('&')
    },

    setTalentPoolView: (viewScope, recruiterId = null) => {
        set((state) => ({
            talentPoolViewScope: viewScope,
            talentPoolRecruiterFilterId: recruiterId,
            talentPoolRoleFilterId: '',
            talentPoolCache: { data: null, lastParamsString: null },
            talentPoolIndex: { rows: [], lastFetchedAt: 0, lastParamsString: '' },
            talentPoolRequest: null,
            talentPoolRequestParamsString: '',
            talentPoolRequestSeq: (state.talentPoolRequestSeq || 0) + 1,
            talentPoolIndexRequest: null,
            talentPoolIndexRequestParamsString: '',
            talentPoolIndexRequestSeq: (state.talentPoolIndexRequestSeq || 0) + 1,
        }))
    },

    setTalentPoolRoleFilter: (roleId = '') => {
        set((state) => ({
            talentPoolRoleFilterId: roleId || '',
            talentPoolCache: { data: null, lastParamsString: null },
            talentPoolIndex: { rows: [], lastFetchedAt: 0, lastParamsString: '' },
            talentPoolRequest: null,
            talentPoolRequestParamsString: '',
            talentPoolRequestSeq: (state.talentPoolRequestSeq || 0) + 1,
            talentPoolIndexRequest: null,
            talentPoolIndexRequestParamsString: '',
            talentPoolIndexRequestSeq: (state.talentPoolIndexRequestSeq || 0) + 1,
        }))
    },

    invalidateTalentPoolCaches: (options = {}) => {
        const clearRows = options.clearRows === true
        set((state) => ({
            talentPoolCache: { data: null, lastParamsString: null },
            talentPoolIndex: { rows: [], lastFetchedAt: 0, lastParamsString: '' },
            ...(clearRows
                ? {
                    tpCandidates: [],
                    tpTotal: 0,
                    tpTotalPages: 1,
                    tpStatusCounts: {},
                }
                : {}),
            talentPoolRequest: null,
            talentPoolRequestParamsString: '',
            talentPoolRequestSeq: (state.talentPoolRequestSeq || 0) + 1,
            talentPoolIndexRequest: null,
            talentPoolIndexRequestParamsString: '',
            talentPoolIndexRequestSeq: (state.talentPoolIndexRequestSeq || 0) + 1,
            analytics: null,
            analyticsLastFetchedAt: 0,
            analyticsRequest: null,
        }))
    },

    fetchTalentPool: async (paramsString, options = {}) => {
        const force = options.force === true
        const state = get()
        const cache = state.talentPoolCache || { data: null, lastParamsString: null }
        const fullParams = get().buildTalentPoolQueryKey(paramsString)

        if (!force && state.talentPoolRequest && state.talentPoolRequestParamsString === fullParams) {
            return state.talentPoolRequest
        }

        // SWR Implementation: If we have cached data for these identical params, return it immediately.
        // This makes navigation back to Talent Pool feel instantaneous.
        if (!force && cache.lastParamsString === fullParams && cache.data) {
            const d = cache.data;
            const incomingHasRows = (d.candidates || []).length > 0 || Number(d.total || 0) > 0
            const currentHasRows = (state.tpCandidates || []).length > 0 || Number(state.tpTotal || 0) > 0
            if (incomingHasRows || !currentHasRows) {
                set({
                    tpCandidates: d.candidates || [],
                    tpTotal: d.total || 0,
                    tpTotalPages: d.total_pages || 1,
                    tpStatusCounts: d.status_counts || {}
                })
            }
            // If the cache is relatively fresh (less than 1 min), don't even trigger a background fetch
            // But for now, we'll let the background fetch run to ensure 100% correctness.
        }

        const requestSeq = (state.talentPoolRequestSeq || 0) + 1
        const request = axios.get(`${API_BASE}/candidates/browse?${fullParams}`)
            .then(res => {
                const d = res.data;
                const latestState = get()
                if (
                    latestState.talentPoolRequestSeq === requestSeq &&
                    latestState.talentPoolRequestParamsString === fullParams
                ) {
                    set({
                        tpCandidates: d.candidates || [],
                        tpTotal: d.total || 0,
                        tpTotalPages: d.total_pages || 1,
                        tpStatusCounts: d.status_counts || {},
                        talentPoolCache: { data: d, lastParamsString: fullParams },
                        talentPoolRequest: null,
                        talentPoolRequestParamsString: '',
                    })
                }
                return { success: true, data: d, cached: false }
            })
            .catch(error => {
                console.error('Failed to fetch talent pool:', error)
                const latestState = get()
                const isLatestRequest =
                    latestState.talentPoolRequestSeq === requestSeq &&
                    latestState.talentPoolRequestParamsString === fullParams
                if (!isLatestRequest) {
                    return { success: false, stale: true, error: 'Ignored stale talent pool request' }
                }
                if (isLatestRequest) {
                    set({ talentPoolRequest: null, talentPoolRequestParamsString: '' })
                }
                const latestCache = get().talentPoolCache
                if (latestCache.lastParamsString === fullParams && latestCache.data) {
                    const d = latestCache.data;
                    const latestStateAfterError = get()
                    const incomingHasRows = (d.candidates || []).length > 0 || Number(d.total || 0) > 0
                    const currentHasRows = (latestStateAfterError.tpCandidates || []).length > 0 || Number(latestStateAfterError.tpTotal || 0) > 0
                    if (incomingHasRows || !currentHasRows) {
                        set({
                            tpCandidates: d.candidates || [],
                            tpTotal: d.total || 0,
                            tpTotalPages: d.total_pages || 1,
                            tpStatusCounts: d.status_counts || {}
                        })
                    }
                    return { success: true, data: d, cached: true }
                }
                return { success: false, error: getRequestErrorMessage(error, 'Failed to load candidates') }
            })

        set({
            talentPoolRequest: request,
            talentPoolRequestParamsString: fullParams,
            talentPoolRequestSeq: requestSeq,
        })

        return request
    },

    fetchTalentPoolIndex: async (options = {}) => {
        const force = options.force === true
        const state = get()
        const freshnessMs = 5 * 60 * 1000
        const scopeQ = get().buildTalentPoolScopeQuery()
        const indexQs = ['page=1', 'page_size=5000', 'sort_by=name', 'sort_dir=asc', scopeQ].filter(Boolean).join('&')

        if (!force && state.talentPoolIndex?.lastParamsString === indexQs && state.talentPoolIndex?.rows?.length && state.talentPoolIndex?.lastFetchedAt && (Date.now() - state.talentPoolIndex.lastFetchedAt) < freshnessMs) {
            return { success: true, data: state.talentPoolIndex.rows, cached: true }
        }

        if (!force && state.talentPoolIndexRequest && state.talentPoolIndexRequestParamsString === indexQs) {
            return state.talentPoolIndexRequest
        }

        const requestSeq = (state.talentPoolIndexRequestSeq || 0) + 1
        const request = axios.get(`${API_BASE}/candidates/browse?${indexQs}`)
            .then(res => {
                const rows = res.data?.candidates || []
                const latestState = get()
                if (
                    latestState.talentPoolIndexRequestSeq === requestSeq &&
                    latestState.talentPoolIndexRequestParamsString === indexQs
                ) {
                    set({
                        talentPoolIndex: { rows, lastFetchedAt: Date.now(), lastParamsString: indexQs },
                        talentPoolIndexRequest: null,
                        talentPoolIndexRequestParamsString: '',
                    })
                }
                return { success: true, data: rows, cached: false }
            })
            .catch(error => {
                console.error('Failed to fetch talent pool index:', error)
                const latestState = get()
                const isLatestRequest =
                    latestState.talentPoolIndexRequestSeq === requestSeq &&
                    latestState.talentPoolIndexRequestParamsString === indexQs
                if (!isLatestRequest) {
                    return { success: false, stale: true, error: 'Ignored stale talent pool index request' }
                }
                if (isLatestRequest) {
                    set({ talentPoolIndexRequest: null, talentPoolIndexRequestParamsString: '' })
                }
                return { success: false, error: getRequestErrorMessage(error, 'Failed to load talent pool index') }
            })

        set({
            talentPoolIndexRequest: request,
            talentPoolIndexRequestParamsString: indexQs,
            talentPoolIndexRequestSeq: requestSeq,
        })
        return request
    },

    // Calls and Tasks
    callLists: [],
    calls: [],
    callStats: defaultCallStats,
    callListsLastFetchedAt: 0,
    callListsRequest: null,
    callsCache: {},
    callsCacheFetchedAt: {},
    callsLastFetchedAt: 0,
    callsLastQueryKey: '',
    callsRequest: null,
    callsRequestQueryKey: '',
    callStatsLastFetchedAt: 0,
    callStatsRequest: null,
    callListsBackoffUntil: 0,
    callStatsBackoffUntil: 0,
    callsBackoffUntilByQuery: {},
    candidateActivityCache: {},
    candidateActivityFetchedAt: {},

    fetchCallLists: async (options = {}) => {
        const force = typeof options === 'boolean' ? options : options.force === true
        const maxAgeMs = 30 * 1000
        const state = get()
        const isFresh = state.callLists.length > 0 &&
            state.callListsLastFetchedAt &&
            (Date.now() - state.callListsLastFetchedAt < maxAgeMs)

        if (!force && isFresh) {
            return { success: true, data: state.callLists, cached: true }
        }

        if (state.callListsBackoffUntil && Date.now() < state.callListsBackoffUntil) {
            if (state.callListsLastFetchedAt) {
                return { success: true, data: state.callLists, cached: true, throttled: true }
            }
            return { success: false, error: 'Retrying call lists shortly' }
        }

        if (state.callListsRequest) {
            return state.callListsRequest
        }

        const request = axios.get(`${API_BASE}/calls/lists`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            .then(res => {
                const latestState = get()
                const mergedLists = mergePendingCallLists(res.data, latestState.callLists)
                set({
                    callLists: mergedLists,
                    callListsLastFetchedAt: Date.now(),
                    callListsRequest: null,
                    callListsBackoffUntil: 0,
                })
                return { success: true, data: mergedLists, cached: false }
            })
            .catch(e => {
                console.error('Failed to fetch call lists:', e)
                set({
                    callListsRequest: null,
                    callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                })
                if (get().callLists.length) {
                    return { success: true, data: get().callLists, cached: true }
                }
                return { success: false, error: getRequestErrorMessage(e, 'Failed to fetch call lists') }
            })

        set({ callListsRequest: request })
        return request
    },

    createCallList: async (name) => {
        const trimmedName = (name || '').trim()
        if (!trimmedName) {
            return { success: false, error: 'List name is required' }
        }

        const normalizedName = normalizeListName(trimmedName)
        const state = get()
        const duplicateExists = (state.callLists || []).some(list => normalizeListName(list?.name) === normalizedName)
        if (duplicateExists) {
            return { success: false, error: 'A list with this name already exists' }
        }

        const optimisticId = -Date.now()
        const optimisticList = {
            id: optimisticId,
            name: trimmedName,
            created_at: new Date().toISOString(),
            candidate_count: 0,
            is_pending: true,
        }

        set(currentState => ({
            callLists: sortCallListsByCreatedAt([optimisticList, ...currentState.callLists]),
            callListsLastFetchedAt: Date.now(),
            callListsBackoffUntil: 0,
            callStats: {
                ...currentState.callStats,
                active_lists: Math.max((currentState.callStats?.active_lists || 0) + 1, currentState.callLists.length + 1),
            },
            callStatsLastFetchedAt: Date.now(),
            callStatsBackoffUntil: 0,
        }))

        try {
            const res = await axios.post(`${API_BASE}/calls/lists`, { name: trimmedName }, { timeout: CALL_REQUEST_TIMEOUT_MS })
            const createdList = res.data

            set(currentState => {
                const remainingLists = (currentState.callLists || []).filter(list =>
                    list.id !== optimisticId &&
                    normalizeListName(list?.name) !== normalizedName
                )

                return {
                    callLists: sortCallListsByCreatedAt([createdList, ...remainingLists]),
                    callListsLastFetchedAt: Date.now(),
                    callListsBackoffUntil: 0,
                    callStats: {
                        ...currentState.callStats,
                        active_lists: Math.max(currentState.callStats?.active_lists || 0, remainingLists.length + 1),
                    },
                    callStatsLastFetchedAt: Date.now(),
                    callStatsBackoffUntil: 0,
                }
            })

            return { success: true, data: createdList }
        } catch (e) {
            console.error('Failed to create call list:', e)
            set(currentState => ({
                callLists: currentState.callLists.filter(list => list.id !== optimisticId),
                callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                callStats: {
                    ...currentState.callStats,
                    active_lists: Math.max(0, (currentState.callStats?.active_lists || 1) - 1),
                },
                callStatsLastFetchedAt: Date.now(),
                callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
            }))
            return { success: false, error: getRequestErrorMessage(e, 'Failed to create list') }
        }
    },

    addCandidatesToCallList: async (candidateIds, listId) => {
        try {
            const uniqueCandidateIds = [...new Set((candidateIds || []).map(Number).filter(Boolean))]
            if (!uniqueCandidateIds.length) {
                return { success: true, data: { success: true, added_count: 0 } }
            }

            const targetListId = Number(listId)
            const optimisticAddedCount = uniqueCandidateIds.length
            const previousState = {
                callLists: get().callLists,
                callStats: get().callStats,
                callListsLastFetchedAt: get().callListsLastFetchedAt,
                callStatsLastFetchedAt: get().callStatsLastFetchedAt,
            }

            set(state => ({
                callLists: state.callLists.map(list =>
                    list.id === targetListId
                        ? { ...list, candidate_count: Math.max(0, (list.candidate_count || 0) + optimisticAddedCount) }
                        : list
                ),
                callListsLastFetchedAt: Date.now(),
                callStats: {
                    ...state.callStats,
                    due_today: Math.max(0, (state.callStats?.due_today || 0) + optimisticAddedCount),
                },
                callStatsLastFetchedAt: Date.now(),
            }))

            void axios.post(
                `${API_BASE}/calls/add-candidates`,
                { candidate_ids: uniqueCandidateIds, list_id: targetListId },
                { timeout: CALL_REQUEST_TIMEOUT_MS }
            )
                .then(res => {
                    const actualAddedCount = Number(res.data?.added_count || 0)
                    const delta = actualAddedCount - optimisticAddedCount
                    set(state => ({
                        callLists: state.callLists.map(list =>
                            list.id === targetListId
                                ? { ...list, candidate_count: Math.max(0, (list.candidate_count || 0) + delta) }
                                : list
                        ),
                        callStats: {
                            ...state.callStats,
                            due_today: Math.max(0, (state.callStats?.due_today || 0) + delta),
                        },
                        callListsLastFetchedAt: 0,
                        callStatsLastFetchedAt: 0,
                        callsCache: {},
                        callsCacheFetchedAt: {},
                        callsLastFetchedAt: 0,
                        callsLastQueryKey: '',
                    }))
                    get().fetchCallStats({ force: true })
                    get().fetchCallLists({ force: true })
                    get().fetchCalls({ due_filter: 'today', status: 'pending' }, { force: true, updateState: false })
                })
                .catch(e => {
                    console.error('Failed to add candidates to list:', e)
                    set({
                        callLists: previousState.callLists,
                        callStats: previousState.callStats,
                        callListsLastFetchedAt: previousState.callListsLastFetchedAt,
                        callStatsLastFetchedAt: previousState.callStatsLastFetchedAt,
                    })
                    toast.error(getRequestErrorMessage(e, 'Failed to add candidates to list'))
                })

            return { success: true, data: { success: true, added_count: optimisticAddedCount }, optimistic: true }
        } catch (e) {
            console.error('Failed to add candidates to list:', e)
            return { success: false, error: getRequestErrorMessage(e, 'Failed to add candidates to list') }
        }
    },

    fetchCalls: async (paramsArg = {}, optionsVal = {}) => {
        const force = typeof optionsVal === 'boolean' ? optionsVal : optionsVal.force === true
        const updateState = typeof optionsVal === 'object' ? optionsVal.updateState !== false : true

        // Normalize params: if it's a number/string, assume it's a listId
        let params = paramsArg;
        if (typeof paramsArg === 'number' || (typeof paramsArg === 'string' && !paramsArg.includes('=') && !isNaN(paramsArg))) {
            params = { list_id: paramsArg };
        } else if (!paramsArg) {
            params = {};
        }

        const queryParams = new URLSearchParams(params).toString()
        const queryKey = queryParams || '__all__'
        const maxAgeMs = 15 * 1000
        const state = get()
        const cachedData = state.callsCache[queryKey]
        const cachedAt = state.callsCacheFetchedAt[queryKey] || 0
        const backoffUntil = state.callsBackoffUntilByQuery?.[queryKey] || 0
        const isFresh = Array.isArray(cachedData) &&
            cachedAt &&
            (Date.now() - cachedAt < maxAgeMs)

        if (!force && isFresh) {
            if (updateState) {
                set({
                    calls: cachedData,
                    callsLastFetchedAt: cachedAt,
                    callsLastQueryKey: queryKey,
                })
            }
            return { success: true, data: cachedData, cached: true }
        }

        if (backoffUntil && Date.now() < backoffUntil) {
            if (Array.isArray(cachedData)) {
                if (updateState) {
                    set({
                        calls: cachedData,
                        callsLastFetchedAt: cachedAt,
                        callsLastQueryKey: queryKey,
                    })
                }
                return { success: true, data: cachedData, cached: true, throttled: true }
            }
            return { success: false, error: 'Retrying calls shortly' }
        }

        if (state.callsRequest && state.callsRequestQueryKey === queryKey) {
            return state.callsRequest.then(result => {
                if (updateState && result?.success) {
                    set({
                        calls: result.data || [],
                        callsLastFetchedAt: get().callsCacheFetchedAt[queryKey] || Date.now(),
                        callsLastQueryKey: queryKey,
                    })
                }
                return result
            })
        }

        const request = axios.get(`${API_BASE}/calls?${queryParams}`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            .then(res => {
                if (get().callsRequestQueryKey === queryKey) {
                    const fetchedAt = Date.now()
                    const nextState = {
                        callsCache: {
                            ...get().callsCache,
                            [queryKey]: res.data,
                        },
                        callsCacheFetchedAt: {
                            ...get().callsCacheFetchedAt,
                            [queryKey]: fetchedAt,
                        },
                        callsRequest: null,
                        callsRequestQueryKey: '',
                        callsBackoffUntilByQuery: {
                            ...get().callsBackoffUntilByQuery,
                            [queryKey]: 0,
                        },
                    }
                    if (updateState) {
                        nextState.calls = res.data
                        nextState.callsLastFetchedAt = fetchedAt
                        nextState.callsLastQueryKey = queryKey
                    }
                    set(nextState)
                }
                return { success: true, data: res.data, cached: false }
            })
            .catch(e => {
                console.error('Failed to fetch calls:', e)
                if (get().callsRequestQueryKey === queryKey) {
                    set({
                        callsRequest: null,
                        callsRequestQueryKey: '',
                        callsBackoffUntilByQuery: {
                            ...get().callsBackoffUntilByQuery,
                            [queryKey]: Date.now() + CALL_RETRY_BACKOFF_MS,
                        },
                    })
                }
                const fallbackData = get().callsCache[queryKey]
                if (Array.isArray(fallbackData)) {
                    if (updateState) {
                        set({
                            calls: fallbackData,
                            callsLastFetchedAt: get().callsCacheFetchedAt[queryKey] || 0,
                            callsLastQueryKey: queryKey,
                        })
                    }
                    return { success: true, data: fallbackData, cached: true }
                }
                return { success: false, error: e.response?.data?.detail || 'Failed to fetch calls' }
            })

        set({
            callsRequest: request,
            callsRequestQueryKey: queryKey,
        })
        return request
    },

    updateCall: async (callId, data) => {
        try {
            await axios.patch(`${API_BASE}/calls/${callId}`, data)
            set(state => ({
                calls: state.calls.filter(c => c.id !== callId),
                callsCache: {},
                callsCacheFetchedAt: {},
                callsLastFetchedAt: 0,
                callsLastQueryKey: '',
                callsRequestQueryKey: '',
            }))
            get().fetchCallStats({ force: true })
            return { success: true }
        } catch (e) {
            console.error('Failed to update call:', e)
            return { success: false, error: getRequestErrorMessage(e, 'Failed to update call') }
        }
    },

    syncCallRecording: async (callId) => {
        try {
            const res = await axios.post(`${API_BASE}/calls/${callId}/sync-recording`, {}, { timeout: CALL_REQUEST_TIMEOUT_MS })
            const updatedCall = res.data

            set(state => {
                const currentCalls = updateCallInCollection(state.calls, updatedCall)
                const cachePatch = patchCallAcrossCaches(state.callsCache, updatedCall)
                const nextActivityCache = { ...state.candidateActivityCache }
                const nextActivityFetchedAt = { ...state.candidateActivityFetchedAt }

                if (updatedCall?.candidate_id != null) {
                    delete nextActivityCache[updatedCall.candidate_id]
                    delete nextActivityFetchedAt[updatedCall.candidate_id]
                }

                return {
                    calls: currentCalls.found ? currentCalls.next : state.calls,
                    callsCache: cachePatch.nextCache,
                    candidateActivityCache: nextActivityCache,
                    candidateActivityFetchedAt: nextActivityFetchedAt,
                }
            })

            return { success: true, data: updatedCall }
        } catch (error) {
            console.error('Failed to sync call recording:', error)
            return { success: false, error: getRequestErrorMessage(error, 'Failed to sync call recording') }
        }
    },

    deleteCall: (callId) => {
        const previousState = {
            calls: get().calls,
            callLists: get().callLists,
            callStats: get().callStats,
        }
        const removedCall = (previousState.calls || []).find(c => c.id === callId)
        const listId = Number(removedCall?.list_id)

        set(state => ({
            calls: state.calls.filter(c => c.id !== callId),
            callLists: state.callLists.map(list =>
                list.id === listId
                    ? { ...list, candidate_count: Math.max(0, (list.candidate_count || 0) - 1) }
                    : list
            ),
            callStats: {
                ...state.callStats,
                due_today: Math.max(0, (state.callStats?.due_today || 0) - 1),
            },
            callListsLastFetchedAt: Date.now(),
            callStatsLastFetchedAt: Date.now(),
            callsCache: {},
            callsCacheFetchedAt: {},
            callsLastFetchedAt: 0,
            callsRequestQueryKey: '',
        }))

        void axios.delete(`${API_BASE}/calls/${callId}`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            .then(() => {
                set({
                    callListsLastFetchedAt: 0,
                    callStatsLastFetchedAt: 0,
                })
                get().fetchCallStats({ force: true })
                get().fetchCallLists({ force: true })
            })
            .catch(e => {
                console.error('Failed to delete call:', e)
                set({
                    calls: previousState.calls,
                    callLists: previousState.callLists,
                    callStats: previousState.callStats,
                })
                toast.error(getRequestErrorMessage(e, 'Failed to remove candidate from list'))
            })

        return Promise.resolve({ success: true, optimistic: true })
    },

    deleteCallList: (listId) => {
        const previousState = {
            callLists: get().callLists,
            calls: get().calls,
            callStats: get().callStats,
        }
        const removedCalls = (previousState.calls || []).filter(call => call.list_id === listId)

        set(state => ({
            callLists: state.callLists.filter(l => l.id !== listId),
            calls: state.calls.filter(call => call.list_id !== listId),
            callStats: {
                ...state.callStats,
                active_lists: Math.max(0, (state.callStats?.active_lists || 0) - 1),
                due_today: Math.max(0, (state.callStats?.due_today || 0) - removedCalls.length),
            },
            callsCache: {},
            callsCacheFetchedAt: {},
            callListsLastFetchedAt: Date.now(),
            callStatsLastFetchedAt: Date.now(),
            callsLastFetchedAt: 0,
            callsLastQueryKey: '',
            callsRequestQueryKey: '',
        }))

        void axios.delete(`${API_BASE}/calls/lists/${listId}`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            .then(() => {
                set({
                    callListsLastFetchedAt: 0,
                    callStatsLastFetchedAt: 0,
                })
                get().fetchCallStats({ force: true })
                get().fetchCallLists({ force: true })
            })
            .catch(e => {
                console.error('Failed to delete call list:', e)
                set({
                    callLists: previousState.callLists,
                    calls: previousState.calls,
                    callStats: previousState.callStats,
                })
                toast.error(getRequestErrorMessage(e, 'Failed to delete list'))
            })

        return Promise.resolve({ success: true, optimistic: true })
    },

    fetchCallStats: async (options = {}) => {
        const force = typeof options === 'boolean' ? options : options.force === true
        const maxAgeMs = 15 * 1000
        const state = get()
        const isFresh = state.callStatsLastFetchedAt &&
            (Date.now() - state.callStatsLastFetchedAt < maxAgeMs)

        if (!force && isFresh) {
            return { success: true, data: state.callStats, cached: true }
        }

        if (state.callStatsBackoffUntil && Date.now() < state.callStatsBackoffUntil) {
            if (state.callStatsLastFetchedAt) {
                return { success: true, data: state.callStats, cached: true, throttled: true }
            }
            return { success: false, error: 'Retrying call stats shortly' }
        }

        if (state.callStatsRequest) {
            return state.callStatsRequest
        }

        const request = axios.get(`${API_BASE}/calls/stats`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            .then(res => {
                set({
                    callStats: res.data,
                    callStatsLastFetchedAt: Date.now(),
                    callStatsRequest: null,
                    callStatsBackoffUntil: 0,
                })
                return { success: true, data: res.data, cached: false }
            })
            .catch(e => {
                console.error('Failed to fetch call stats:', e)
                set({
                    callStatsRequest: null,
                    callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                })
                return { success: false, error: e.response?.data?.detail || 'Failed to fetch call stats' }
            })

        set({ callStatsRequest: request })
        return request
    },

    fetchCandidateActivity: async (candidateId, options = {}) => {
        const force = typeof options === 'boolean' ? options : options.force === true
        const maxAgeMs = 15 * 1000
        const state = get()
        const cachedData = state.candidateActivityCache[candidateId]
        const cachedAt = state.candidateActivityFetchedAt[candidateId] || 0
        const isFresh = Array.isArray(cachedData) && cachedAt && (Date.now() - cachedAt < maxAgeMs)

        if (!force && isFresh) {
            return { success: true, data: cachedData, cached: true }
        }

        try {
            const res = await axios.get(`${API_BASE}/candidates/${candidateId}/activity`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            const items = res.data?.items || []
            const fetchedAt = Date.now()
            set(state => ({
                candidateActivityCache: {
                    ...state.candidateActivityCache,
                    [candidateId]: items,
                },
                candidateActivityFetchedAt: {
                    ...state.candidateActivityFetchedAt,
                    [candidateId]: fetchedAt,
                },
            }))
            return { success: true, data: items, cached: false }
        } catch (error) {
            console.error('Failed to fetch candidate activity:', error)
            if (Array.isArray(cachedData)) {
                return { success: true, data: cachedData, cached: true }
            }
            return { success: false, error: getRequestErrorMessage(error, 'Failed to fetch candidate activity') }
        }
    },

    initiateCall: async (callId) => {
        try {
            const res = await axios.post(
                `${API_BASE}/calls/initiate`,
                { call_id: callId, dial_mode: 'voip' },
                { timeout: 45000 }
            )
            return { success: true, data: res.data }
        } catch (error) {
            console.error('Failed to initiate call:', error)
            const detail = getRequestErrorDetail(error, 'Initiation failed')
            return {
                success: false,
                error: detail.message,
                errorCode: detail.code,
                actionLabel: detail.actionLabel,
                actionUrl: detail.actionUrl,
                errorMeta: detail.meta,
            }
        }
    },

    // Add explicitly to help UI clear state on tab switch
    clearCallsState: () => set({ calls: [], callsLastQueryKey: '' })
}), {
    name: 'app-storage-v2',
    partialize: (state) => ({
        user: state.user,
        heyreachCampaignId: state.heyreachCampaignId,
        isSidebarCollapsed: state.isSidebarCollapsed,
        // Do not persist analytics: stale zeros (failed fetch / cold cache) overwrite real counts after reload.
        tpCandidates: state.tpCandidates,
        tpTotal: state.tpTotal,
        tpStatusCounts: state.tpStatusCounts,
        talentPoolCache: state.talentPoolCache,
        searchResults: state.searchResults,
        searchQuery: state.searchQuery
    }),
    merge: (persistedState, currentState) => {
        const nextState = {
            ...currentState,
            ...(persistedState || {}),
        }

        return {
            ...nextState,
            callLists: currentState.callLists,
            calls: currentState.calls,
            callStats: currentState.callStats || defaultCallStats,
            callListsLastFetchedAt: 0,
            callsLastFetchedAt: 0,
            callStatsLastFetchedAt: 0,
            callListsRequest: null,
            callsRequest: null,
            callsRequestQueryKey: '',
            callStatsRequest: null,
            callListsBackoffUntil: 0,
            callStatsBackoffUntil: 0,
            callsBackoffUntilByQuery: {},
        }
    },
}))


// Initialize Axios header if token exists
const token = localStorage.getItem('token')
if (token) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
}
