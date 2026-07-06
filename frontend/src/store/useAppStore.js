import { create } from 'zustand'
import axios from 'axios'
import { toast } from 'sonner'

const REQUEST_TIMEOUT_MS = 15000
const SEARCH_REQUEST_TIMEOUT_MS = 120000
const CALL_REQUEST_TIMEOUT_MS = 15000
const CALL_RETRY_BACKOFF_MS = 3000
const RATE_LIMIT_DEFAULT_RETRY_MS = 5000
const RATE_LIMIT_MAX_RETRY_MS = 30000
const TALENT_POOL_CACHE_FRESH_MS = 30 * 1000
let rateLimitUntilMs = 0
const roleDetailsInFlight = new Map()

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))
const isDev = import.meta.env.DEV

function shouldLogApiTiming(url = '') {
    if (!isDev || !url) return false
    return [
        '/me',
        '/candidates/browse',
        '/candidates/browse/summary',
        '/candidates/browse/meta',
        '/roles/',
        '/ai-columns',
        '/calls',
    ].some(path => String(url).includes(path))
}

function logApiTiming(config = {}, status = 'ERR') {
    if (!config._timingStart || !shouldLogApiTiming(config.url)) return
    const elapsedMs = Math.round(performance.now() - config._timingStart)
    const method = String(config.method || 'GET').toUpperCase()
    console.info(`[api ${elapsedMs}ms] ${method} ${config.url} ${status}`)
}

function parseRetryAfterMs(value) {
    if (!value) return RATE_LIMIT_DEFAULT_RETRY_MS
    const seconds = Number(value)
    if (Number.isFinite(seconds)) {
        return Math.min(RATE_LIMIT_MAX_RETRY_MS, Math.max(1000, seconds * 1000))
    }
    const dateMs = new Date(value).getTime()
    if (Number.isFinite(dateMs)) {
        return Math.min(RATE_LIMIT_MAX_RETRY_MS, Math.max(1000, dateMs - Date.now()))
    }
    return RATE_LIMIT_DEFAULT_RETRY_MS
}

axios.defaults.timeout = REQUEST_TIMEOUT_MS

axios.interceptors.request.use((config) => {
    if (shouldLogApiTiming(config.url)) {
        config._timingStart = performance.now()
    }
    return config
})

axios.interceptors.response.use(
    (response) => {
        logApiTiming(response.config, response.status)
        return response
    },
    async (error) => {
        const config = error?.config || {}
        const status = error?.response?.status
        logApiTiming(config, status || error?.code || 'ERR')

        if (status === 429) {
            const retryAfterMs = parseRetryAfterMs(error?.response?.headers?.['retry-after'])
            rateLimitUntilMs = Math.max(rateLimitUntilMs, Date.now() + retryAfterMs)
            toast.warning(`Server is rate limiting requests. Retrying shortly.`, { id: 'api-rate-limit' })
            if (!config._rateLimitRetry && config.method?.toLowerCase() === 'get') {
                config._rateLimitRetry = true
                await sleep(retryAfterMs)
                return axios(config)
            }
        }

        if (status !== 429 && rateLimitUntilMs > Date.now() && !config._rateLimitWaited && config.method?.toLowerCase() === 'get') {
            config._rateLimitWaited = true
            await sleep(Math.min(RATE_LIMIT_MAX_RETRY_MS, rateLimitUntilMs - Date.now()))
            return axios(config)
        }

        if (!error?.response && !config._retry && config.method?.toLowerCase() === 'get') {
            config._retry = true
            await sleep(1500)
            return axios(config)
        }

        if (status === 401 && localStorage.getItem('token')) {
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
    if (status === 429) {
        const retryAfterMs = parseRetryAfterMs(error.response?.headers?.['retry-after'])
        return `Server is busy. Please retry in about ${Math.ceil(retryAfterMs / 1000)} seconds`
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

function searchCandidateKey(candidate, index = 0) {
    return String(candidate?.id ?? candidate?.linkedin ?? candidate?.name ?? `candidate:${index}`)
}

function mergeSearchCandidates(existing = [], incoming = [], preferIncomingOrder = false) {
    const merged = new Map()
    const order = []

    const add = (candidate, index) => {
        if (!candidate) return
        const key = searchCandidateKey(candidate, index)
        if (!merged.has(key)) order.push(key)
        merged.set(key, { ...(merged.get(key) || {}), ...candidate })
    }

    if (preferIncomingOrder) {
        incoming.forEach(add)
        existing.forEach(add)
    } else {
        existing.forEach(add)
        incoming.forEach(add)
    }

    return order.map(key => merged.get(key)).filter(Boolean)
}

function countVerifiedSearchCandidates(candidates = []) {
    return (candidates || []).filter(candidate => (
        candidate?.is_verified_match ||
        candidate?.shortlist_status === 'verified_match' ||
        candidate?.shortlist_status === 'shortlisted'
    )).length
}

function searchFoundMessage(total, verifiedCount = null) {
    const safeTotal = Number(total) || 0
    const safeVerified = Number.isFinite(Number(verifiedCount)) ? Number(verifiedCount) : 0
    if (safeTotal <= 0) return 'No qualified matches found'
    if (safeVerified > 0) {
        return `${safeVerified} qualified match${safeVerified === 1 ? '' : 'es'}`
    }
    return `${safeTotal} qualified match${safeTotal === 1 ? '' : 'es'}`
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

function removeCallAcrossCaches(callsCache = {}, callId) {
    const targetId = Number(callId)
    const nextCache = {}

    for (const [queryKey, entries] of Object.entries(callsCache || {})) {
        if (!Array.isArray(entries)) {
            nextCache[queryKey] = entries
            continue
        }
        nextCache[queryKey] = entries.filter(call => Number(call?.id) !== targetId)
    }

    return nextCache
}

function removeCallsByListAcrossCaches(callsCache = {}, listId) {
    const targetListId = Number(listId)
    const nextCache = {}

    for (const [queryKey, entries] of Object.entries(callsCache || {})) {
        if (!Array.isArray(entries)) {
            nextCache[queryKey] = entries
            continue
        }
        nextCache[queryKey] = entries.filter(call => Number(call?.list_id) !== targetListId)
    }

    return nextCache
}

function isDueTodayCall(call) {
    if (!call || call.status === 'completed' || !call.due_date) return false
    const dueDate = new Date(call.due_date)
    if (Number.isNaN(dueDate.getTime())) return false
    const today = new Date()
    dueDate.setHours(0, 0, 0, 0)
    today.setHours(0, 0, 0, 0)
    return dueDate <= today
}

// Canonical query key for the calls cache: drops empty values, stringifies,
// and sorts params so `{status, list_id}` and `{list_id, status}` (or a number
// vs string list_id) always map to the SAME cache entry. Without this, two
// callers requesting the same view could read/write different cache keys and
// briefly show each other's stale data.
export function canonicalCallsQuery(params = {}) {
    const entries = Object.entries(params || {})
        .filter(([, value]) => value !== undefined && value !== null && value !== '')
        .map(([key, value]) => [key, String(value)])
        .sort(([left], [right]) => left.localeCompare(right))
    return new URLSearchParams(entries).toString()
}

function clearCallRequestState() {
    return {
        callListsRequest: null,
        callListsRequestSeq: -1,
        callsRequest: null,
        callsRequestQueryKey: '',
        callsRequestSeq: -1,
        callStatsRequest: null,
        callStatsRequestSeq: -1,
    }
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

const roleCacheKey = (role) => {
    if (role && typeof role === 'object') return String(role.id ?? role.name)
    return String(role || '')
}

const roleNameFrom = (role) => {
    if (role && typeof role === 'object') return role.name
    return role
}

const roleUrl = (role, suffix = '') => {
    const name = roleNameFrom(role)
    const id = role && typeof role === 'object' ? role.id : null
    const params = id ? `?role_id=${encodeURIComponent(id)}` : ''
    return `${API_BASE}/roles/${encodeURIComponent(name)}${suffix}${params}`
}

// Absolute base for OAuth and top-level redirects
export const BACKEND_BASE = isLocalHost ? 'http://127.0.0.1:8000' : (API_URL || window.location.origin)

// WebSocket URL
const BACKEND_HOST = useAbsoluteApi ? API_URL.replace(/^https?:\/\//, '') : window.location.host
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
const WS_URL = `${protocol}//${BACKEND_HOST}/api/ws/search`

// Global App Store
import { persist } from 'zustand/middleware'

const defaultCallStats = { due_today: 0, upcoming: 0, completed: 0, active_lists: 0 }

function getAnalyticsUserKey(user) {
    if (!user?.id) return ''
    return `${user.id}:${String(user.role || '').trim().toLowerCase()}`
}

function emptyAuthScopedCaches() {
    return {
        analytics: null,
        analyticsLastFetchedAt: 0,
        analyticsRequest: null,
        analyticsUserKey: '',
        analyticsRequestUserKey: '',
        tpCandidates: [],
        tpTotal: 0,
        tpTotalPages: 1,
        tpStatusCounts: {},
        tpScopeTotal: null,
        tpScopeStatusCounts: {},
        tpScopeSummaryIsRefreshing: false,
        tpScopeSummaryRequest: null,
        tpScopeSummaryRequestParamsString: '',
        tpScopeSummaryLastFetchedAt: 0,
        tpScopeSummaryLastParamsString: '',
        talentPoolCache: { data: null, lastParamsString: null, lastFetchedAt: 0 },
        talentPoolIndex: { rows: [], lastFetchedAt: 0, lastParamsString: '' },
        talentPoolRequest: null,
        talentPoolRequestParamsString: '',
        talentPoolIndexRequest: null,
        talentPoolIndexRequestParamsString: '',
    }
}

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
                user: userData,
                ...emptyAuthScopedCaches()
            })

            axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
            get().invalidateTalentPoolCaches({ clearRows: true })
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
                user: userData,
                ...emptyAuthScopedCaches()
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
        set({ token: null, isAuthenticated: false, user: null, ...emptyAuthScopedCaches() })
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
                user: userData,
                ...emptyAuthScopedCaches()
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
            const res = await axios.get(`${API_BASE}/me`, { timeout: 8000 })
            const previousUserKey = getAnalyticsUserKey(get().user)
            const nextUserKey = getAnalyticsUserKey(res.data)
            set({
                user: res.data,
                isAuthenticated: true,
                ...(previousUserKey && previousUserKey !== nextUserKey ? emptyAuthScopedCaches() : {}),
            })
            if (previousUserKey && previousUserKey !== nextUserKey) {
                get().invalidateTalentPoolCaches({ clearRows: true })
            }
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
    updateRecruiter: async (id, data) => {
        const previousRecruiters = get().recruiters
        try {
            const res = await axios.patch(`${API_BASE}/admin/recruiters/${id}`, data)
            set(state => ({
                recruiters: state.recruiters.map(r => r.id === id ? res.data : r)
            }))
            return { success: true, data: res.data }
        } catch (e) {
            set({ recruiters: previousRecruiters })
            return { success: false, error: e.response?.data?.detail || 'Failed to update recruiter' }
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
        const previousTalentPoolCandidate = get().talentPoolCache?.candidates?.find(candidate => Number(candidate.id) === Number(id));
        const previousRoleCandidate = get().viewingRole?.candidates?.find(candidate => Number(candidate.id) === Number(id));
        const previousRoleName = get().viewingRole?.name;

        set(state => {
            const next = {};
            const cache = state.talentPoolCache;
            if (cache?.candidates) {
                next.talentPoolCache = {
                    ...cache,
                    candidates: cache.candidates.map(candidate =>
                        Number(candidate.id) === Number(id) ? { ...candidate, notes } : candidate
                    ),
                };
            }

            if (state.viewingRole?.candidates) {
                const updatedRole = {
                    ...state.viewingRole,
                    candidates: state.viewingRole.candidates.map(candidate =>
                        Number(candidate.id) === Number(id) ? { ...candidate, notes } : candidate
                    ),
                };
                next.viewingRole = updatedRole;
                next.roleDetailsCache = {
                    ...state.roleDetailsCache,
                    [updatedRole.name]: updatedRole,
                };
            }
            return next;
        });

        try {
            await axios.patch(`${API_BASE}/candidates/${id}/notes`, { notes });
            return { success: true };
        } catch (e) {
            set(state => {
                const next = {};
                if (previousTalentPoolCandidate && state.talentPoolCache?.candidates) {
                    next.talentPoolCache = {
                        ...state.talentPoolCache,
                        candidates: state.talentPoolCache.candidates.map(candidate =>
                            Number(candidate.id) === Number(id) ? { ...candidate, notes: previousTalentPoolCandidate.notes } : candidate
                        ),
                    };
                }
                if (previousRoleCandidate && state.viewingRole?.candidates) {
                    const updatedRole = {
                        ...state.viewingRole,
                        candidates: state.viewingRole.candidates.map(candidate =>
                            Number(candidate.id) === Number(id) ? { ...candidate, notes: previousRoleCandidate.notes } : candidate
                        ),
                    };
                    next.viewingRole = updatedRole;
                    next.roleDetailsCache = {
                        ...state.roleDetailsCache,
                        [previousRoleName || updatedRole.name]: updatedRole,
                    };
                }
                return next;
            });
            console.error('Failed to update notes:', e);
            const error = e.response?.data?.detail || 'Failed to update notes';
            toast.error(error);
            return { success: false, error };
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
    analyticsUserKey: '',
    analyticsRequestUserKey: '',
    fetchAnalytics: async (options = {}) => {
        const force = typeof options === 'boolean' ? options : options.force === true
        const state = get()
        const freshnessMs = 60 * 1000
        const userKey = getAnalyticsUserKey(state.user)

        if (!userKey) {
            set({
                analytics: null,
                analyticsLastFetchedAt: 0,
                analyticsRequest: null,
                analyticsUserKey: '',
                analyticsRequestUserKey: '',
            })
            return { success: false, error: 'No authenticated user' }
        }

        if (state.analytics && state.analyticsUserKey === userKey && !force) {
            if (!state.analyticsRequest && (!state.analyticsLastFetchedAt || Date.now() - state.analyticsLastFetchedAt >= freshnessMs)) {
                setTimeout(() => { get().fetchAnalytics({ force: true }) }, 0)
            }
            return { success: true, data: state.analytics, cached: true }
        }

        if (state.analyticsRequest && state.analyticsRequestUserKey === userKey) {
            return state.analyticsRequest
        }

        const request = axios.get(`${API_BASE}/candidates/analytics`, { timeout: 60000 })
            .then(res => {
                if (getAnalyticsUserKey(get().user) !== userKey) {
                    return { success: false, stale: true, error: 'Ignored stale analytics request' }
                }
                // If total_sourced is 0 the DB may have been temporarily unavailable.
                // Mark lastFetchedAt as 0 so a background re-fetch is triggered on
                // the next render instead of persisting stale zeros to localStorage.
                const isEmpty = !res.data?.summary
                    ? (res.data?.total_sourced === 0 || res.data?.total_sourced == null)
                    : (res.data.summary?.total_sourced === 0 || res.data.summary?.total_sourced == null)
                set({
                    analytics: res.data,
                    analyticsLastFetchedAt: isEmpty ? 0 : Date.now(),
                    analyticsRequest: null,
                    analyticsUserKey: userKey,
                    analyticsRequestUserKey: '',
                })
                return { success: true, data: res.data, cached: false }
            })
            .catch(e => {
                console.error('Failed to fetch analytics:', e)
                if (getAnalyticsUserKey(get().user) === userKey) {
                    set({ analyticsRequest: null, analyticsRequestUserKey: '' })
                }
                const fallback = get().analytics
                if (fallback && get().analyticsUserKey === userKey) return { success: true, data: fallback, cached: true }
                return { success: false, error: e.response?.data?.detail || 'Failed to fetch analytics' }
            })

        set({ analyticsRequest: request, analyticsRequestUserKey: userKey })
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
    searchDebug: null,
    usage: null,
    searchOutcome: 'idle',
    lastSearchError: '',
    _searchFallbackTimer: null,

    // Search via REST (synchronous)
    searchCandidates: async (query, options = {}) => {
        const initialStatus = options.initialStatus || 'Screening...'
        const sourceType = options.sourceType || options.source_type || 'master'
        const sourceRoleId = options.sourceRoleId || options.source_role_id || null
        const useWebSearch = Boolean(options.useWebSearch ?? options.use_web_search)
        // Preserve any results already streamed via WS — only clear if starting fresh
        const alreadyHaveResults = get().searchResults?.length > 0
        set({
            isSearching: true,
            // Don't wipe live-streamed results when falling back to REST
            ...(alreadyHaveResults ? {} : { searchResults: [] }),
            usage: null,
            statusMessage: initialStatus,
            searchProgress: alreadyHaveResults ? get().searchProgress : 0,
            searchTotal: alreadyHaveResults ? get().searchTotal : 0,
            searchDebug: alreadyHaveResults ? get().searchDebug : null,
            searchOutcome: 'loading',
            lastSearchError: '',
        })
        try {
            const res = await axios.post(`${API_BASE}/search`, {
                query,
                source_type: sourceType,
                source_role_id: sourceRoleId,
                use_web_search: useWebSearch,
            }, { timeout: SEARCH_REQUEST_TIMEOUT_MS })
            const totalCandidates = res.data.total ?? res.data.candidates?.length ?? 0
            const currentResults = get().searchResults || []
            const mergedResults = mergeSearchCandidates(currentResults, res.data.candidates || [], true)
            const visibleTotal = Math.max(totalCandidates, mergedResults.length)
            const verifiedCount = res.data.verified_count ?? countVerifiedSearchCandidates(mergedResults)
            set({
                searchResults: mergedResults,
                searchDebug: res.data.filter_debug || null,
                usage: res.data.usage,
                statusMessage: searchFoundMessage(visibleTotal, verifiedCount),
                isSearching: false,
                searchProgress: 100,
                searchOutcome: visibleTotal > 0 ? 'success' : 'empty',
            })
        } catch (e) {
            const errorMessage = getRequestErrorMessage(e, 'Screening failed')
            const currentResults = get().searchResults || []
            set({
                statusMessage: currentResults.length > 0
                    ? `Found ${currentResults.length} candidates (connection dropped)`
                    : 'Screening failed: ' + errorMessage,
                isSearching: false,
                searchOutcome: currentResults.length > 0 ? 'success' : 'error',
                lastSearchError: currentResults.length > 0 ? '' : errorMessage,
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
    searchCandidatesStream: (query, options = {}) => {
        const sourceType = options.sourceType || options.source_type || 'master'
        const sourceRoleId = options.sourceRoleId || options.source_role_id || null
        const useWebSearch = Boolean(options.useWebSearch ?? options.use_web_search)
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
            searchDebug: null,
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
                initialStatus: 'Realtime screening unavailable. Running standard search...',
                sourceType,
                sourceRoleId,
                useWebSearch,
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
            ws.send(JSON.stringify({
                query,
                token,
                source_type: sourceType,
                source_role_id: sourceRoleId,
                use_web_search: useWebSearch,
            }))
            set({ statusMessage: useWebSearch ? 'Researching company facts...' : 'Processing query...' })
            // 2 minutes — backend may need to warm the candidate cache on cold start
            armFallbackTimer(120000)
        }

        ws.onmessage = (event) => {
            if (finished) return

            const data = JSON.parse(event.data)

            // A transport heartbeat is not search progress. Do not let pings keep
            // a hung model stage alive forever; real status/progress events reset it.
            if (data.type === 'ping') {
                return
            }

            // Database is cold-loading — give it 3 minutes before falling back
            if (data.type === 'status' && typeof data.message === 'string' && data.message.toLowerCase().includes('loading')) {
                set({ statusMessage: data.message })
                armFallbackTimer(180000)
                return
            }

            // Real progress signals (candidate, progress, progress_start, complete) get
            // the full 120s window. Plain status messages (e.g. "Expanding geographies")
            // only get 45s — enough for LLM calls, not enough to mask indefinite CPU blocking.
            const isProgressSignal = ['candidate', 'complete', 'progress', 'progress_start'].includes(data.type)
            armFallbackTimer(isProgressSignal ? 120000 : 45000)

            switch (data.type) {
                case 'status':
                    set({ statusMessage: data.message })
                    break

                case 'progress_start':
                    set({ searchTotal: data.total, statusMessage: `Generating reasoning for ${data.total} qualified match${data.total === 1 ? '' : 'es'}...` })
                    break

                case 'candidate':
                    set(state => {
                        const nextResults = mergeSearchCandidates(state.searchResults, [data.data])
                        const reviewed = Number(data.reviewed ?? data.current ?? 0)
                        const total = Number(data.total || state.searchTotal || 0)
                        const verified = Number(data.verified ?? countVerifiedSearchCandidates(nextResults))
                        const pendingCount = nextResults.filter(c => c?.shortlist_status === 'pending_reasoning').length
                        const verifiedCount = nextResults.filter(c => c?.shortlist_status !== 'pending_reasoning').length

                        let statusMsg
                        if (data.phase === 'scoring') {
                            // Early candidate from scoring phase — no reasoning yet
                            statusMsg = `${nextResults.length} candidate${nextResults.length === 1 ? '' : 's'} passed filter — generating reasoning...`
                        } else if (total > 0 && reviewed > 0) {
                            statusMsg = verifiedCount > 0
                                ? `${verifiedCount} verified · ${pendingCount} pending reasoning (${reviewed}/${total})`
                                : `Generated reasoning for ${reviewed} of ${total}. ${searchFoundMessage(nextResults.length, verified)}`
                        } else {
                            statusMsg = `Scoring ${total || nextResults.length} profile${(total || nextResults.length) === 1 ? '' : 's'}...`
                        }

                        return {
                            searchResults: nextResults,
                            searchProgress: total > 0 && reviewed > 0 ? Math.round((reviewed / total) * 100) : state.searchProgress,
                            statusMessage: statusMsg,
                        }
                    })
                    break

                case 'progress': {
                    const pCurrent = Number(data.current) || 0
                    const pTotal = Number(data.total) || 1
                    const pPct = Math.round((pCurrent / pTotal) * 100)
                    const pMsg = data.phase === 'filtering'
                        ? (data.message || `Evaluating candidate pool ${pCurrent}/${pTotal}...`)
                        : `Scored ${pCurrent} of ${pTotal} profiles...`
                    set(state => ({
                        searchProgress: Math.max(state.searchProgress, pPct),
                        statusMessage: pMsg,
                    }))
                    break
                }

                case 'complete':
                    finished = true
                    clearFallbackTimer()
                    set(state => {
                        const mergedResults = mergeSearchCandidates(state.searchResults, data.candidates || [], true)
                        const totalCandidates = Math.max(data.total ?? 0, mergedResults.length)
                        const verifiedCount = data.verified_count ?? countVerifiedSearchCandidates(mergedResults)
                        return {
                            searchResults: mergedResults,
                            searchDebug: data.filter_debug || state.searchDebug || null,
                            usage: data.usage,
                            statusMessage: searchFoundMessage(totalCandidates, verifiedCount),
                            isSearching: false,
                            searchProgress: 100,
                            searchOutcome: totalCandidates > 0 ? 'success' : 'empty',
                        }
                    })
                    set({ _ws: null })
                    ws.close()
                    break

                case 'candidate_batch': {
                    // Batch of pre-reasoning candidates from the scoring phase.
                    // We no longer display these in the table immediately to prevent "profile without reasoning" UI.
                    // We just update the status message to show we are working on them.
                    const batchItems = Array.isArray(data.data) ? data.data : []
                    if (batchItems.length) {
                        set({ statusMessage: `${batchItems.length} candidate${batchItems.length === 1 ? '' : 's'} passed filter — generating reasoning...` })
                    }
                    break
                }

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
            isSearchPaused: false,
            statusMessage: 'Screening stopped',
            searchOutcome: 'cancelled',
            lastSearchError: '',
            _ws: null,
            _searchFallbackTimer: null,
        })
    },

    pauseSearch: () => {
        const ws = get()._ws
        if (ws) {
            ws.send(JSON.stringify({ action: 'pause' }))
            set({ isSearchPaused: true })
        }
    },

    resumeSearch: () => {
        const ws = get()._ws
        if (ws) {
            ws.send(JSON.stringify({ action: 'resume' }))
            set({ isSearchPaused: false })
        }
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
            searchDebug: null,
            usage: null,
            isSearching: false,
            isSearchPaused: false,
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
                const fetchedRoles = res.data.roles || []
                set({
                    roles: fetchedRoles,
                    rolesLastFetchedAt: Date.now(),
                    rolesRequest: null
                })

                // Warm small/medium role details after the list paints. Opening
                // those roles then uses the same instant cache-first experience
                // as Talent Pool without preloading very large role payloads.
                setTimeout(() => {
                    fetchedRoles
                        .filter(role => Number(role.candidate_count || 0) <= 250)
                        .slice(0, 5)
                        .forEach(role => {
                            if (!get().roleDetailsCache[role.name]) {
                                void get()._fetchRoleDetailsBackground(role.name)
                            }
                        })
                }, 0)

                return { success: true, data: fetchedRoles, cached: false }
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
            // Preserve the list metadata while clearly marking candidate details as loading.
            set({ viewingRole: { ...role, candidates: null } })
        }

        // Fetch full details in background (always refresh)
        get()._fetchRoleDetailsBackground(role.name)
    },

    _fetchRoleDetailsBackground: async (roleName, options = {}) => {
        const force = options.force === true
        if (!force && roleDetailsInFlight.has(roleName)) {
            return roleDetailsInFlight.get(roleName)
        }

        const request = (async () => {
          try {
            const query = force ? `?refresh=true&cb=${Date.now()}` : ''
            const res = await axios.get(`${API_BASE}/roles/${encodeURIComponent(roleName)}${query}`)
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
            return { success: true, data: updatedRole }
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
            return { success: false, error: error.response?.data?.detail || 'Failed to fetch role details' }
          }
        })()

        if (!force) roleDetailsInFlight.set(roleName, request)
        try {
            return await request
        } finally {
            if (!force && roleDetailsInFlight.get(roleName) === request) {
                roleDetailsInFlight.delete(roleName)
            }
        }
    },

    // Kept for direct calls if needed, but openRole is preferred for UI
    fetchRoleDetails: async (roleName, options = {}) => {
        return get()._fetchRoleDetailsBackground(roleName, options)
    },

    createRole: async (setup) => {
        const name = setup.name
        const newRole = { id: Date.now(), name, candidate_count: 0, activation_status: 'activating' }
        const prevRoles = [...get().roles]

        // Optimistic Update
        set({ roles: [newRole, ...prevRoles] })

        try {
            const res = await axios.post(`${API_BASE}/roles`, setup, { timeout: 180000 })
            // Refresh with real server data in background
            get().fetchRoles({ force: true })
            return { success: true, data: res.data }
        } catch (error) {
            const detail = error.response?.data?.detail || ''
            const mayHaveCompleted = !error.response
                || error.code === 'ECONNABORTED'
                || String(detail).toLowerCase().includes('already exists')

            // External provisioning can complete after the browser connection
            // times out. Recover the durable role instead of inviting a duplicate.
            if (mayHaveCompleted) {
                try {
                    const verification = await axios.get(`${API_BASE}/roles`, { timeout: 60000 })
                    const fetchedRoles = verification.data?.roles || []
                    const created = fetchedRoles.find(role =>
                        String(role.name || '').trim().toLowerCase() === String(name || '').trim().toLowerCase()
                    )
                    if (created) {
                        set({ roles: fetchedRoles, rolesLastFetchedAt: Date.now() })
                        return { success: true, data: created, recovered: true }
                    }
                } catch (verificationError) {
                    console.error('Failed to verify timed-out role creation:', verificationError)
                }
            }

            set({ roles: prevRoles })
            console.error('Failed to create role:', error)
            return { success: false, error: detail || error.message || 'Failed to create role' }
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
        const prevRoles = [...get().roles]
        const previousCache = { ...get().roleDetailsCache }
        const previousViewingRole = get().viewingRole

        const candidatesToAdd = assignments.map(assign => {
            const fullProfile = get().searchResults.find(c => c.id === assign.candidate_id) || {}
            return {
                ...fullProfile,
                ...assign,
                email: fullProfile.email || null,
                mobile_phone: fullProfile.mobile_phone || null
            }
        })

        const nextCache = { ...previousCache }
        const existingCachedRole = nextCache[roleName] || { name: roleName, candidates: [] }
        const existingIds = new Set((existingCachedRole.candidates || []).map(candidate => Number(candidate.id)))
        const optimisticAdds = candidatesToAdd.filter(candidate => !existingIds.has(Number(candidate.id)))

        const newCachedRole = {
            ...existingCachedRole,
            candidates: [...(existingCachedRole.candidates || []), ...optimisticAdds]
        }

        nextCache[roleName] = newCachedRole
        set({ roleDetailsCache: nextCache })

        if (get().viewingRole?.name === roleName) {
            set({ viewingRole: newCachedRole })
        }

        try {
            const res = await axios.post(`${API_BASE}/roles/${encodeURIComponent(roleName)}/assign`, {
                assignments
            })

            await get().fetchRoles({ force: true })
            await get()._fetchRoleDetailsBackground(roleName)
            get().invalidateTalentPoolCaches?.()
            get().fetchTalentPoolSummary?.({ force: true })
            get().fetchAnalytics?.({ force: true })

            return { success: true, data: res.data }
        } catch (error) {
            set({ roles: prevRoles, roleDetailsCache: previousCache, viewingRole: previousViewingRole })
            return { success: false, error: error.response?.data?.detail || 'Failed to assign candidates' }
        }
    },

    shortlistEnrichAndAddToRole: async (roleId, assignments) => {
        const role = get().roles.find(item => Number(item.id) === Number(roleId))
        if (!role) return { success: false, error: 'Destination role not found' }
        const previousRoles = [...get().roles]
        const previousCache = { ...get().roleDetailsCache }
        const previousSearchResults = [...get().searchResults]
        const selectedProfiles = assignments.map(assignment => ({
            ...(get().searchResults.find(candidate => Number(candidate.id) === Number(assignment.candidate_id)) || {}),
            ...assignment,
            status: 'Shortlisted',
        }))
        const cachedRole = previousCache[role.name] || { ...role, candidates: [] }
        const existingIds = new Set((cachedRole.candidates || []).map(candidate => Number(candidate.id)))
        const optimisticAdds = selectedProfiles.filter(candidate => !existingIds.has(Number(candidate.id)))
        const optimisticRole = {
            ...cachedRole,
            candidates: [...(cachedRole.candidates || []), ...optimisticAdds],
        }
        set(state => ({
            roles: state.roles.map(item => Number(item.id) === Number(roleId)
                ? { ...item, candidate_count: Number(item.candidate_count || 0) + optimisticAdds.length }
                : item),
            roleDetailsCache: { ...state.roleDetailsCache, [role.name]: optimisticRole },
            searchResults: state.searchResults.map(candidate =>
                assignments.some(item => Number(item.candidate_id) === Number(candidate.id))
                    ? { ...candidate, status: 'Shortlisted' }
                    : candidate),
        }))
        try {
            const res = await axios.post(
                `${API_BASE}/outreach/roles/${roleId}/shortlist-selected`,
                { assignments },
                { timeout: 120000 },
            )
            void get().fetchRoles({ force: true })
            void get()._fetchRoleDetailsBackground(role.name, { force: true })
            get().invalidateTalentPoolCaches?.()
            void get().fetchTalentPoolSummary?.({ force: true, freshnessMs: 0 })
            void get().fetchAnalytics?.({ force: true })
            return { success: true, data: res.data }
        } catch (error) {
            set({ roles: previousRoles, roleDetailsCache: previousCache, searchResults: previousSearchResults })
            return { success: false, error: error.response?.data?.detail || error.message || 'Failed to shortlist selected profiles' }
        }
    },

    // One-click add a single candidate to a role — skips enrichment if email+phone already present
    quickAddCandidateToRole: async (candidateId, roleId) => {
        const roles = get().roles
        const role = roles.find(r => Number(r.id) === Number(roleId))
        if (!role) return { success: false, error: 'Role not found' }
        try {
            const res = await axios.post(
                `${API_BASE}/outreach/roles/${roleId}/shortlist-selected`,
                { assignments: [{ candidate_id: candidateId, priority: '--', feedback: '' }] },
                { timeout: 30000 },
            )
            // Update the candidate status in search results
            set(state => ({
                searchResults: state.searchResults.map(c =>
                    Number(c.id) === Number(candidateId) ? { ...c, status: 'Shortlisted', _addedToRole: role.name } : c
                ),
            }))
            void get().fetchRoles({ force: true })
            return { success: true, data: res.data }
        } catch (error) {
            return { success: false, error: error.response?.data?.detail || error.message || 'Failed to add to role' }
        }
    },

    deactivateRole: async (roleId) => {
        try {
            const res = await axios.post(`${API_BASE}/roles/id/${roleId}/deactivate`)
            // Optimistically update the active role status
            set(state => {
                if (state.viewingRole && state.viewingRole.id === roleId) {
                    return { viewingRole: { ...state.viewingRole, activation_status: 'inactive' } }
                }
                return state
            })
            void get().fetchRoles({ force: true })
            return { success: true, data: res.data }
        } catch (error) {
            return { success: false, error: error.response?.data?.detail || error.message || 'Failed to deactivate role' }
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
            const res = await axios.get(`${API_BASE}/outreach/status/${roleId}?cb=${Date.now()}`)
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
                    talentPoolCache: { data: null, lastParamsString: null, lastFetchedAt: 0 }
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
    tpScopeTotal: null,
    tpScopeStatusCounts: {},
    tpScopeSummaryIsRefreshing: false,
    tpScopeSummaryRequest: null,
    tpScopeSummaryRequestParamsString: '',
    tpScopeSummaryRequestSeq: 0,
    tpScopeSummaryLastFetchedAt: 0,
    tpScopeSummaryLastParamsString: '',
    tpFilters: {
        title: [], titleInput: '',
        company: [], companyInput: '',
        city: [], cityInput: '',
        product_service: [], productInput: '',
        status: [], statusInput: '', created_by: '',
        min_exp: 0, max_exp: 40,
    },
    tpActiveStatusTab: '',
    tpSortBy: 'name',
    tpSortDir: 'asc',
    tpPage: 1,
    tpPageSize: 25,
    tpGlobalSearch: '',
    tpAiRunFocus: null,

    setTpFilters: (updater) => set((state) => ({
        tpFilters: typeof updater === 'function' ? updater(state.tpFilters) : updater
    })),
    setTpActiveStatusTab: (tab) => set({ tpActiveStatusTab: tab }),
    setTpPagination: (page, pageSize) => set({ tpPage: page, tpPageSize: pageSize }),
    setTpSort: (sortBy, sortDir) => set({ tpSortBy: sortBy, tpSortDir: sortDir }),
    setTpGlobalSearch: (q) => set({ tpGlobalSearch: q }),
    setTpCandidates: (candidates) => set({ tpCandidates: candidates || [] }),
    setTpStatusCounts: (counts) => set({ tpStatusCounts: counts || {} }),
    setTpScopeSummary: (summary = {}) => set({
        tpScopeTotal: summary.total ?? null,
        tpScopeStatusCounts: summary.status_counts || {},
    }),
    updateTpCandidate: (candidateId, data) => set(state => ({
        tpCandidates: (state.tpCandidates || []).map(c => c.id === candidateId ? { ...c, ...data } : c),
        talentPoolIndex: {
            ...state.talentPoolIndex,
            rows: (state.talentPoolIndex?.rows || []).map(c => c.id === candidateId ? { ...c, ...data } : c)
        }
    })),

    talentPoolCache: { data: null, lastParamsString: null, lastFetchedAt: 0 },
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

    isTalentPoolScopeReady: () => {
        const u = get().user
        if (u?.role !== 'admin') return true
        const vs = get().talentPoolViewScope || 'master'
        return vs !== 'recruiter_pools' || Boolean(get().talentPoolRecruiterFilterId)
    },

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
            tpAiRunFocus: null,
            tpCandidates: [],
            tpTotal: 0,
            tpTotalPages: 1,
            tpStatusCounts: {},
            tpScopeTotal: null,
            tpScopeStatusCounts: {},
            tpScopeSummaryIsRefreshing: false,
            tpScopeSummaryRequest: null,
            tpScopeSummaryRequestParamsString: '',
            tpScopeSummaryRequestSeq: (state.tpScopeSummaryRequestSeq || 0) + 1,
            tpScopeSummaryLastFetchedAt: 0,
            tpScopeSummaryLastParamsString: '',
            talentPoolCache: { data: null, lastParamsString: null, lastFetchedAt: 0 },
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
            tpAiRunFocus: null,
            tpCandidates: [],
            tpTotal: 0,
            tpTotalPages: 1,
            tpStatusCounts: {},
            tpScopeTotal: null,
            tpScopeStatusCounts: {},
            tpScopeSummaryIsRefreshing: false,
            tpScopeSummaryRequest: null,
            tpScopeSummaryRequestParamsString: '',
            tpScopeSummaryRequestSeq: (state.tpScopeSummaryRequestSeq || 0) + 1,
            tpScopeSummaryLastFetchedAt: 0,
            tpScopeSummaryLastParamsString: '',
            talentPoolCache: { data: null, lastParamsString: null, lastFetchedAt: 0 },
            talentPoolIndex: { rows: [], lastFetchedAt: 0, lastParamsString: '' },
            talentPoolRequest: null,
            talentPoolRequestParamsString: '',
            talentPoolRequestSeq: (state.talentPoolRequestSeq || 0) + 1,
            talentPoolIndexRequest: null,
            talentPoolIndexRequestParamsString: '',
            talentPoolIndexRequestSeq: (state.talentPoolIndexRequestSeq || 0) + 1,
        }))
    },

    startTpAiRunFocus: (payload = {}) => {
        console.log("[DEBUG STORE] startTpAiRunFocus payload:", payload)
        const state = get()
        const candidateIds = Array.isArray(payload.candidateIds)
            ? [...new Set(payload.candidateIds.map(Number).filter(Number.isFinite))]
            : []
        if (!candidateIds.length) return
        set({
            tpAiRunFocus: {
                runId: payload.runId || null,
                columnDefinitionId: payload.columnDefinitionId || null,
                columnName: payload.columnName || 'Smart Column',
                candidateIds,
                startedAt: new Date().toISOString(),
                previousView: {
                    filters: state.tpFilters,
                    activeStatusTab: state.tpActiveStatusTab,
                    sortBy: state.tpSortBy,
                    sortDir: state.tpSortDir,
                    page: state.tpPage,
                    pageSize: state.tpPageSize,
                    globalSearch: state.tpGlobalSearch,
                    viewScope: state.talentPoolViewScope,
                    recruiterFilterId: state.talentPoolRecruiterFilterId,
                    roleFilterId: state.talentPoolRoleFilterId,
                },
            },
            tpPage: 1,
            tpCandidates: [],
            tpTotal: candidateIds.length,
            tpTotalPages: 1,
            tpStatusCounts: {},
            talentPoolCache: { data: null, lastParamsString: null, lastFetchedAt: 0 },
            talentPoolRequest: null,
            talentPoolRequestParamsString: '',
            talentPoolRequestSeq: (state.talentPoolRequestSeq || 0) + 1,
        })
    },

    exitTpAiRunFocus: () => {
        const focus = get().tpAiRunFocus
        const previousView = focus?.previousView || {}
        set({
            tpAiRunFocus: null,
            tpFilters: previousView.filters || get().tpFilters,
            tpActiveStatusTab: previousView.activeStatusTab || '',
            tpSortBy: previousView.sortBy || get().tpSortBy,
            tpSortDir: previousView.sortDir || get().tpSortDir,
            tpPage: previousView.page || 1,
            tpPageSize: previousView.pageSize || get().tpPageSize,
            tpGlobalSearch: previousView.globalSearch || '',
            talentPoolViewScope: previousView.viewScope || get().talentPoolViewScope,
            talentPoolRecruiterFilterId: previousView.recruiterFilterId || null,
            talentPoolRoleFilterId: previousView.roleFilterId || '',
            talentPoolCache: { data: null, lastParamsString: null, lastFetchedAt: 0 },
            talentPoolRequest: null,
            talentPoolRequestParamsString: '',
        })
    },

    invalidateTalentPoolCaches: (options = {}) => {
        const clearRows = options.clearRows === true
        set((state) => ({
            talentPoolCache: { data: null, lastParamsString: null, lastFetchedAt: 0 },
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
            tpScopeTotal: null,
            tpScopeStatusCounts: {},
            tpScopeSummaryIsRefreshing: false,
            tpScopeSummaryRequest: null,
            tpScopeSummaryRequestParamsString: '',
            tpScopeSummaryRequestSeq: (state.tpScopeSummaryRequestSeq || 0) + 1,
            tpScopeSummaryLastFetchedAt: 0,
            tpScopeSummaryLastParamsString: '',
            analyticsRequest: null,
        }))
    },

    fetchTalentPoolSummary: async (options = {}) => {
        const freshnessMs = options.freshnessMs ?? 30 * 1000
        if (!get().isTalentPoolScopeReady()) {
            set({
                tpScopeSummaryRequest: null,
                tpScopeSummaryRequestParamsString: '',
                tpScopeSummaryIsRefreshing: false,
                tpScopeTotal: null,
                tpScopeStatusCounts: {},
                tpScopeSummaryLastFetchedAt: 0,
                tpScopeSummaryLastParamsString: '',
            })
            return { success: false, blocked: true, error: 'Select a recruiter to load this pool' }
        }
        const scopeQ = get().buildTalentPoolScopeQuery()
        const paramsString = scopeQ || ''
        const state = get()
        const hasCachedSummary = state.tpScopeTotal != null && state.tpScopeSummaryLastParamsString === paramsString

        if (!options.force && hasCachedSummary && state.tpScopeSummaryLastFetchedAt && Date.now() - state.tpScopeSummaryLastFetchedAt < freshnessMs) {
            return {
                success: true,
                cached: true,
                data: {
                    total: state.tpScopeTotal,
                    status_counts: state.tpScopeStatusCounts || {},
                },
            }
        }

        if (!options.force && state.tpScopeSummaryRequest && state.tpScopeSummaryRequestParamsString === paramsString) {
            return state.tpScopeSummaryRequest
        }

        const requestSeq = (state.tpScopeSummaryRequestSeq || 0) + 1
        const url = paramsString
            ? `${API_BASE}/candidates/browse/summary?${paramsString}`
            : `${API_BASE}/candidates/browse/summary`
            
        const requestPromise = axios.get(url, { timeout: 60000 })
            .then(res => {
                const d = res.data || {}
                const statusCounts = d.status_counts || {}
                const latestState = get()
                const isLatestRequest =
                    latestState.tpScopeSummaryRequestSeq === requestSeq &&
                    latestState.tpScopeSummaryRequestParamsString === paramsString
                if (!isLatestRequest) {
                    return { success: false, stale: true, error: 'Ignored stale talent pool summary request' }
                }
                const looksLikeColdEmpty =
                    Number(d.total || 0) === 0 &&
                    Object.keys(statusCounts).length === 0 &&
                    latestState.tpScopeSummaryLastParamsString === paramsString &&
                    Number(latestState.tpScopeTotal || 0) > 0
                if (looksLikeColdEmpty) {
                    set({
                        tpScopeSummaryRequest: null,
                        tpScopeSummaryRequestParamsString: '',
                        tpScopeSummaryIsRefreshing: false,
                    })
                    return {
                        success: true,
                        data: {
                            total: latestState.tpScopeTotal,
                            status_counts: latestState.tpScopeStatusCounts || {},
                        },
                        cached: true,
                        staleEmptyIgnored: true,
                    }
                }
                set({
                    tpScopeTotal: d.total ?? 0,
                    tpScopeStatusCounts: statusCounts,
                    tpScopeSummaryRequest: null,
                    tpScopeSummaryRequestParamsString: '',
                    tpScopeSummaryIsRefreshing: false,
                    tpScopeSummaryLastFetchedAt: Date.now(),
                    tpScopeSummaryLastParamsString: paramsString,
                })
                return { success: true, data: d, cached: false }
            })
            .catch(error => {
                console.error('Failed to fetch talent pool summary:', error)
                const latestState = get()
                if (latestState.tpScopeSummaryRequestSeq === requestSeq) {
                    set({
                        tpScopeSummaryRequest: null,
                        tpScopeSummaryRequestParamsString: '',
                        tpScopeSummaryIsRefreshing: false,
                    })
                }
                return { success: false }
            })

        set({
            tpScopeSummaryRequest: requestPromise,
            tpScopeSummaryRequestParamsString: paramsString,
            tpScopeSummaryRequestSeq: requestSeq,
            tpScopeSummaryIsRefreshing: true,
        })
        return requestPromise
    },

    fetchTalentPool: async (paramsString, options = {}) => {
        const force = options.force === true
        if (!get().isTalentPoolScopeReady()) {
            set({
                tpCandidates: [],
                tpTotal: 0,
                tpTotalPages: 1,
                tpStatusCounts: {},
                talentPoolRequest: null,
                talentPoolRequestParamsString: '',
            })
            return { success: false, blocked: true, error: 'Select a recruiter to load this pool' }
        }
        const state = get()
        const cache = state.talentPoolCache || { data: null, lastParamsString: null, lastFetchedAt: 0 }
        const fullParams = get().buildTalentPoolQueryKey(paramsString)

        console.log("[DEBUG STORE] fetchTalentPool start:", { paramsString, fullParams, force })

        if (!force && state.talentPoolRequest && state.talentPoolRequestParamsString === fullParams) {
            console.log("[DEBUG STORE] fetchTalentPool returning ongoing request for", fullParams)
            return state.talentPoolRequest
        }

        // SWR Implementation: If we have cached data for these identical params, return it immediately.
        // This makes navigation back to Talent Pool feel instantaneous.
        if (!force && cache.lastParamsString === fullParams && cache.data) {
            console.log("[DEBUG STORE] fetchTalentPool SWR cache HIT for", fullParams)
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
            if (cache.lastFetchedAt && Date.now() - cache.lastFetchedAt < TALENT_POOL_CACHE_FRESH_MS) {
                return { success: true, data: d, cached: true }
            }
        }

        const requestSeq = (state.talentPoolRequestSeq || 0) + 1
        console.log("[DEBUG STORE] fetchTalentPool dispatching requestSeq:", requestSeq, "for fullParams:", fullParams)
        const request = axios.get(`${API_BASE}/candidates/browse?${fullParams}&cb=${Date.now()}`)
            .then(res => {
                const d = res.data;
                const latestState = get()
                const matches = latestState.talentPoolRequestSeq === requestSeq &&
                    latestState.talentPoolRequestParamsString === fullParams;
                console.log("[DEBUG STORE] fetchTalentPool promise resolved for requestSeq:", requestSeq, {
                    matches,
                    latestRequestSeq: latestState.talentPoolRequestSeq,
                    latestRequestParamsString: latestState.talentPoolRequestParamsString,
                    candidatesCount: (d.candidates || []).length,
                    total: d.total
                })
                if (matches) {
                    set({
                        tpCandidates: d.candidates || [],
                        tpTotal: d.total || 0,
                        tpTotalPages: d.total_pages || 1,
                        tpStatusCounts: d.status_counts || {},
                        talentPoolCache: { data: d, lastParamsString: fullParams, lastFetchedAt: Date.now() },
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
                console.log("[DEBUG STORE] fetchTalentPool promise rejected for requestSeq:", requestSeq, {
                    isLatestRequest,
                    latestRequestSeq: latestState.talentPoolRequestSeq,
                    latestRequestParamsString: latestState.talentPoolRequestParamsString
                })
                if (!isLatestRequest) {
                    return { success: false, stale: true, error: 'Ignored stale talent pool request' }
                }
                if (isLatestRequest) {
                    set({ talentPoolRequest: null, talentPoolRequestParamsString: '' })
                }
                const latestCache = get().talentPoolCache || { data: null, lastParamsString: null, lastFetchedAt: 0 }
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
        if (!get().isTalentPoolScopeReady()) {
            set({
                talentPoolIndex: { rows: [], lastFetchedAt: 0, lastParamsString: '' },
                talentPoolIndexRequest: null,
                talentPoolIndexRequestParamsString: '',
            })
            return { success: false, blocked: true, error: 'Select a recruiter to load this pool' }
        }
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
        const request = axios.get(`${API_BASE}/candidates/browse?${indexQs}&cb=${Date.now()}`)
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
    callListsRequestSeq: 0,
    callsCache: {},
    callsCacheFetchedAt: {},
    callsLastFetchedAt: 0,
    callsLastQueryKey: '',
    callsRequest: null,
    callsRequestQueryKey: '',
    callsRequestSeq: 0,
    callsMutationSeq: 0,
    callStatsLastFetchedAt: 0,
    callStatsRequest: null,
    callStatsRequestSeq: 0,
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

        if (state.callListsRequest && (!force || state.callListsRequestSeq === state.callsMutationSeq)) {
            return state.callListsRequest
        }

        const requestSeq = state.callsMutationSeq
        const request = axios.get(`${API_BASE}/calls/lists`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            .then(res =>
            {
                const latestState = get()
                const mergedLists = mergePendingCallLists(res.data, latestState.callLists)
                // Only commit if this is still the latest list request (dedup guard).
                // We do NOT guard by callsMutationSeq here — that guard incorrectly
                // discards fresh server data after any mutation bumped the seq.
                if (latestState.callListsRequestSeq === requestSeq) {
                    set({
                        callLists: mergedLists,
                        callListsLastFetchedAt: Date.now(),
                        callListsRequest: null,
                        callListsRequestSeq: 0,
                        callListsBackoffUntil: 0,
                    })
                }
                return { success: true, data: mergedLists, cached: false }
            })
            .catch(e => {
                console.error('Failed to fetch call lists:', e)
                if (get().callListsRequestSeq === requestSeq) {
                    set({
                        callListsRequest: null,
                        callListsRequestSeq: 0,
                        callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    })
                }
                if (get().callLists.length) {
                    return { success: true, data: get().callLists, cached: true }
                }
                return { success: false, error: getRequestErrorMessage(e, 'Failed to fetch call lists') }
            })

        set({ callListsRequest: request, callListsRequestSeq: requestSeq })
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
        const mutationSeq = get().callsMutationSeq + 1

        set(currentState => ({
            ...clearCallRequestState(),
            callsMutationSeq: mutationSeq,
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
                const nextLists = sortCallListsByCreatedAt([createdList, ...remainingLists])

                return {
                    callsMutationSeq: Math.max(currentState.callsMutationSeq || 0, mutationSeq),
                    callLists: nextLists,
                    callListsLastFetchedAt: Date.now(),
                    callListsBackoffUntil: 0,
                    callStats: {
                        ...currentState.callStats,
                        active_lists: nextLists.length,
                    },
                    callStatsLastFetchedAt: Date.now(),
                    callStatsBackoffUntil: 0,
                }
            })

            return { success: true, data: createdList }
        } catch (e) {
            console.error('Failed to create call list:', e)
            set(currentState => {
                if (currentState.callsMutationSeq !== mutationSeq) {
                    return {
                        callLists: currentState.callLists.filter(list => list.id !== optimisticId),
                        callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                        callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    }
                }

                return {
                    callLists: currentState.callLists.filter(list => list.id !== optimisticId),
                    callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    callStats: {
                        ...currentState.callStats,
                        active_lists: Math.max(0, (currentState.callStats?.active_lists || 1) - 1),
                    },
                    callStatsLastFetchedAt: Date.now(),
                    callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                }
            })
            return { success: false, error: getRequestErrorMessage(e, 'Failed to create list') }
        }
    },

    addCandidatesToCallList: async (candidateIds, listId) => {
        let mutationSeq = 0
        let previousState = null
        try {
            const uniqueCandidateIds = [...new Set((candidateIds || []).map(Number).filter(Boolean))]
            if (!uniqueCandidateIds.length) {
                return { success: true, data: { success: true, added_count: 0 } }
            }

            const targetListId = Number(listId)
            const optimisticAddedCount = uniqueCandidateIds.length
            mutationSeq = get().callsMutationSeq + 1
            previousState = {
                callLists: get().callLists,
                callStats: get().callStats,
                callListsLastFetchedAt: get().callListsLastFetchedAt,
                callStatsLastFetchedAt: get().callStatsLastFetchedAt,
                callsCache: get().callsCache,
                callsCacheFetchedAt: get().callsCacheFetchedAt,
                callsLastFetchedAt: get().callsLastFetchedAt,
                callsLastQueryKey: get().callsLastQueryKey,
            }

            set(state => ({
                ...clearCallRequestState(),
                callsMutationSeq: mutationSeq,
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

            const res = await axios.post(
                `${API_BASE}/calls/add-candidates`,
                { candidate_ids: uniqueCandidateIds, list_id: targetListId },
                { timeout: CALL_REQUEST_TIMEOUT_MS }
            )
            const actualAddedCount = Number(res.data?.added_count || 0)
            const delta = actualAddedCount - optimisticAddedCount

            set(state => ({
                callsMutationSeq: Math.max(state.callsMutationSeq || 0, mutationSeq),
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

            // Fire-and-forget: counts were already reconciled from added_count
            // above, so the modal shouldn't wait ~2-3s for these revalidations.
            void Promise.allSettled([
                get().fetchCallStats({ force: true }),
                get().fetchCallLists({ force: true }),
                get().fetchCalls({ due_filter: 'today', status: 'pending' }, { force: true, updateState: false }),
                get().fetchCalls({ list_id: targetListId, status: 'pending' }, { force: true, updateState: false }),
            ])

            return { success: true, data: res.data }
        } catch (e) {
            console.error('Failed to add candidates to list:', e)
            set(state => {
                if (!previousState || state.callsMutationSeq !== mutationSeq) {
                    return {
                        callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                        callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    }
                }

                return {
                    callLists: previousState.callLists,
                    callStats: previousState.callStats,
                    callListsLastFetchedAt: previousState.callListsLastFetchedAt,
                    callStatsLastFetchedAt: previousState.callStatsLastFetchedAt,
                    callsCache: previousState.callsCache,
                    callsCacheFetchedAt: previousState.callsCacheFetchedAt,
                    callsLastFetchedAt: previousState.callsLastFetchedAt,
                    callsLastQueryKey: previousState.callsLastQueryKey,
                    callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                }
            })
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

        const queryParams = canonicalCallsQuery(params)
        const queryKey = queryParams || '__all__'

        // Background mode: plain fetch that ONLY refreshes the cache entry.
        // It never touches callsRequest/callsRequestQueryKey/callsRequestSeq,
        // so a poller (e.g. the post-call review modal) can run concurrently
        // without clobbering the page's in-flight fetch and its commit guard.
        if (typeof optionsVal === 'object' && optionsVal.background === true) {
            try {
                const res = await axios.get(`${API_BASE}/calls?${queryParams}`, { timeout: CALL_REQUEST_TIMEOUT_MS })
                set(state => ({
                    callsCache: { ...state.callsCache, [queryKey]: res.data },
                    callsCacheFetchedAt: { ...state.callsCacheFetchedAt, [queryKey]: Date.now() },
                }))
                return { success: true, data: res.data, cached: false }
            } catch (e) {
                console.error('Background calls fetch failed:', e)
                return { success: false, error: getRequestErrorMessage(e, 'Failed to fetch calls') }
            }
        }

        const maxAgeMs = 15 * 1000
        const state = get()
        const cachedData = state.callsCache[queryKey]
        const cachedAt = state.callsCacheFetchedAt[queryKey] || 0
        const backoffUntil = state.callsBackoffUntilByQuery?.[queryKey] || 0
        const hasCachedData = Array.isArray(cachedData)
        const isFresh = Array.isArray(cachedData) &&
            cachedAt &&
            (Date.now() - cachedAt < maxAgeMs)

        if (!force && hasCachedData) {
            if (updateState) {
                set({
                    calls: cachedData,
                    callsLastFetchedAt: cachedAt,
                    callsLastQueryKey: queryKey,
                })
            }
            if (isFresh) {
                return { success: true, data: cachedData, cached: true }
            }
        }

        if (backoffUntil && Date.now() < backoffUntil) {
            if (hasCachedData) {
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

        if (state.callsRequest && state.callsRequestQueryKey === queryKey && (!force || state.callsRequestSeq === state.callsMutationSeq)) {
            const existingRequestSeq = state.callsRequestSeq
            return state.callsRequest.then(result =>
            {
                const latestState = get()
                const requestStillCurrent = latestState.callsRequestQueryKey === queryKey &&
                    latestState.callsRequestSeq === existingRequestSeq
                const requestSettledWithoutMutation = latestState.callsRequestQueryKey === '' &&
                    latestState.callsRequestSeq === 0 &&
                    latestState.callsMutationSeq === existingRequestSeq
                // When deduplicating, only update state if this request was not invalidated by a mutation.
                if (updateState && result?.success && (requestStillCurrent || requestSettledWithoutMutation)) {
                    set({
                        calls: result.data || [],
                        callsLastFetchedAt: get().callsCacheFetchedAt[queryKey] || Date.now(),
                        callsLastQueryKey: queryKey,
                    })
                }
                return result
            })
        }

        const requestSeq = state.callsMutationSeq
        const request = axios.get(`${API_BASE}/calls?${queryParams}`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            .then(res => {
                const latestState = get()
                // Only commit if this is still the latest request for this query key.
                // Do NOT guard by callsMutationSeq — that discards fresh server data
                // after any mutation (add/delete/etc.) bumps the seq.
                if (latestState.callsRequestQueryKey === queryKey && latestState.callsRequestSeq === requestSeq) {
                    const fetchedAt = Date.now()
                    const nextState = {
                        callsCache: {
                            ...latestState.callsCache,
                            [queryKey]: res.data,
                        },
                        callsCacheFetchedAt: {
                            ...latestState.callsCacheFetchedAt,
                            [queryKey]: fetchedAt,
                        },
                        callsRequest: null,
                        callsRequestQueryKey: '',
                        callsRequestSeq: 0,
                        callsBackoffUntilByQuery: {
                            ...latestState.callsBackoffUntilByQuery,
                            [queryKey]: 0,
                        },
                    }
                    // Always update UI state when data arrives for the right query
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
                if (get().callsRequestQueryKey === queryKey && get().callsRequestSeq === requestSeq) {
                    set({
                        callsRequest: null,
                        callsRequestQueryKey: '',
                        callsRequestSeq: 0,
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
            callsRequestSeq: requestSeq,
        })
        return request
    },

    updateCall: async (callId, data) => {
        const mutationSeq = get().callsMutationSeq + 1
        set(state => ({
            ...clearCallRequestState(),
            callsMutationSeq: mutationSeq,
        }))

        try {
            const res = await axios.patch(`${API_BASE}/calls/${callId}`, data)
            set(state => ({
                callsMutationSeq: Math.max(state.callsMutationSeq || 0, mutationSeq),
                calls: state.calls.filter(c => c.id !== callId),
                callsCache: {},
                callsCacheFetchedAt: {},
                callsLastFetchedAt: 0,
                callsLastQueryKey: '',
                callsRequestQueryKey: '',
                callsRequestSeq: 0,
            }))
            get().fetchCallStats({ force: true })
            return { success: true, data: res.data }
        } catch (e) {
            console.error('Failed to update call:', e)
            return { success: false, error: getRequestErrorMessage(e, 'Failed to update call') }
        }
    },

    syncCallRecording: async (callId) => {
        // NOTE: deliberately does NOT clearCallRequestState() — the review
        // modal polls this every few seconds and wiping the request
        // bookkeeping made the page's own in-flight refetch discard its
        // results (stale list until manual refresh). The success handler
        // patches the affected call across caches directly instead.
        const mutationSeq = get().callsMutationSeq + 1
        set({ callsMutationSeq: mutationSeq })

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
                    callsMutationSeq: Math.max(state.callsMutationSeq || 0, mutationSeq),
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

    deleteCall: async (callId) => {
        const previousState = {
            calls: get().calls,
            callLists: get().callLists,
            callStats: get().callStats,
            callsCache: get().callsCache,
            callsCacheFetchedAt: get().callsCacheFetchedAt,
            callsLastFetchedAt: get().callsLastFetchedAt,
            callsLastQueryKey: get().callsLastQueryKey,
            callListsLastFetchedAt: get().callListsLastFetchedAt,
            callStatsLastFetchedAt: get().callStatsLastFetchedAt,
        }
        const removedCall = (previousState.calls || []).find(c => c.id === callId)
        const listId = Number(removedCall?.list_id)
        const dueTodayDelta = isDueTodayCall(removedCall) ? 1 : 0
        const mutationSeq = get().callsMutationSeq + 1

        set(state => ({
            ...clearCallRequestState(),
            callsMutationSeq: mutationSeq,
            calls: state.calls.filter(c => c.id !== callId),
            callsCache: removeCallAcrossCaches(state.callsCache, callId),
            callLists: state.callLists.map(list =>
                list.id === listId
                    ? { ...list, candidate_count: Math.max(0, (list.candidate_count || 0) - 1) }
                    : list
            ),
            callStats: {
                ...state.callStats,
                due_today: Math.max(0, (state.callStats?.due_today || 0) - dueTodayDelta),
            },
            callListsLastFetchedAt: Date.now(),
            callStatsLastFetchedAt: Date.now(),
        }))

        try {
            const res = await axios.delete(`${API_BASE}/calls/${callId}`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            set(state => ({
                callsMutationSeq: Math.max(state.callsMutationSeq || 0, mutationSeq),
                callListsLastFetchedAt: 0,
                callStatsLastFetchedAt: 0,
            }))
            // Fire-and-forget: the optimistic update already removed the row(s);
            // don't hold the delete spinner on revalidation round trips.
            void Promise.allSettled([
                get().fetchCallStats({ force: true }),
                get().fetchCallLists({ force: true }),
            ])
            return { success: true, data: res.data }
        } catch (e) {
            console.error('Failed to delete call:', e)
            set(state => {
                if (state.callsMutationSeq !== mutationSeq) {
                    return {
                        callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                        callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    }
                }

                return {
                    calls: previousState.calls,
                    callLists: previousState.callLists,
                    callStats: previousState.callStats,
                    callsCache: previousState.callsCache,
                    callsCacheFetchedAt: previousState.callsCacheFetchedAt,
                    callsLastFetchedAt: previousState.callsLastFetchedAt,
                    callsLastQueryKey: previousState.callsLastQueryKey,
                    callListsLastFetchedAt: previousState.callListsLastFetchedAt,
                    callStatsLastFetchedAt: previousState.callStatsLastFetchedAt,
                    callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                }
            })
            return { success: false, error: getRequestErrorMessage(e, 'Failed to remove candidate from list') }
        }
    },

    deleteCallList: async (listId) => {
        const previousState = {
            callLists: get().callLists,
            calls: get().calls,
            callStats: get().callStats,
            callsCache: get().callsCache,
            callsCacheFetchedAt: get().callsCacheFetchedAt,
            callsLastFetchedAt: get().callsLastFetchedAt,
            callsLastQueryKey: get().callsLastQueryKey,
            callListsLastFetchedAt: get().callListsLastFetchedAt,
            callStatsLastFetchedAt: get().callStatsLastFetchedAt,
        }
        const targetListId = Number(listId)
        const removedCalls = (previousState.calls || []).filter(call => Number(call.list_id) === targetListId)
        const removedDueTodayCount = removedCalls.filter(isDueTodayCall).length
        const mutationSeq = get().callsMutationSeq + 1

        set(state => ({
            ...clearCallRequestState(),
            callsMutationSeq: mutationSeq,
            callLists: state.callLists.filter(l => Number(l.id) !== targetListId),
            calls: state.calls.filter(call => Number(call.list_id) !== targetListId),
            callStats: {
                ...state.callStats,
                active_lists: Math.max(0, (state.callStats?.active_lists || 0) - 1),
                due_today: Math.max(0, (state.callStats?.due_today || 0) - removedDueTodayCount),
            },
            callsCache: removeCallsByListAcrossCaches(state.callsCache, targetListId),
            callListsLastFetchedAt: Date.now(),
            callStatsLastFetchedAt: Date.now(),
        }))

        try {
            const res = await axios.delete(`${API_BASE}/calls/lists/${targetListId}`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            set(state => ({
                callsMutationSeq: Math.max(state.callsMutationSeq || 0, mutationSeq),
                callListsLastFetchedAt: 0,
                callStatsLastFetchedAt: 0,
            }))
            // Fire-and-forget: the optimistic update already removed the row(s);
            // don't hold the delete spinner on revalidation round trips.
            void Promise.allSettled([
                get().fetchCallStats({ force: true }),
                get().fetchCallLists({ force: true }),
            ])
            return { success: true, data: res.data }
        } catch (e) {
            console.error('Failed to delete call list:', e)
            set(state => {
                if (state.callsMutationSeq !== mutationSeq) {
                    return {
                        callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                        callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    }
                }

                return {
                    callLists: previousState.callLists,
                    calls: previousState.calls,
                    callStats: previousState.callStats,
                    callsCache: previousState.callsCache,
                    callsCacheFetchedAt: previousState.callsCacheFetchedAt,
                    callsLastFetchedAt: previousState.callsLastFetchedAt,
                    callsLastQueryKey: previousState.callsLastQueryKey,
                    callListsLastFetchedAt: previousState.callListsLastFetchedAt,
                    callStatsLastFetchedAt: previousState.callStatsLastFetchedAt,
                    callListsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                }
            })
            return { success: false, error: getRequestErrorMessage(e, 'Failed to delete list') }
        }
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

        if (state.callStatsRequest && (!force || state.callStatsRequestSeq === state.callsMutationSeq)) {
            return state.callStatsRequest
        }

        const requestSeq = state.callsMutationSeq
        const request = axios.get(`${API_BASE}/calls/stats`, { timeout: CALL_REQUEST_TIMEOUT_MS })
            .then(res => {
                const latestState = get()
                // Only commit if this is still the latest stats request.
                // Do NOT guard by callsMutationSeq — that discards real stats after mutations.
                if (latestState.callStatsRequestSeq === requestSeq) {
                    set({
                        callStats: res.data,
                        callStatsLastFetchedAt: Date.now(),
                        callStatsRequest: null,
                        callStatsRequestSeq: 0,
                        callStatsBackoffUntil: 0,
                    })
                }
                return { success: true, data: res.data, cached: false }
            })
            .catch(e => {
                console.error('Failed to fetch call stats:', e)
                if (get().callStatsRequestSeq === requestSeq) {
                    set({
                        callStatsRequest: null,
                        callStatsRequestSeq: 0,
                        callStatsBackoffUntil: Date.now() + CALL_RETRY_BACKOFF_MS,
                    })
                }
                return { success: false, error: e.response?.data?.detail || 'Failed to fetch call stats' }
            })

        set({ callStatsRequest: request, callStatsRequestSeq: requestSeq })
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

    initiateCall: async (callId, options = {}) => {
        try {
            const res = await axios.post(
                `${API_BASE}/calls/initiate`,
                {
                    call_id: callId,
                    dial_mode: 'voip',
                    plivo_username: options.plivoUsername || options.plivo_username || '',
                },
                { timeout: 45000 }
            )
            return { success: true, data: res.data }
        } catch (error) {
            console.error('Failed to initiate call:', error)
            const detail = getRequestErrorDetail(error, 'Initiation failed')
            const status = error?.response?.status
            const isMissingCallTask = status === 404 &&
                String(detail.message || '').toLowerCase().includes('call task not found')

            if (isMissingCallTask) {
                const mutationSeq = get().callsMutationSeq + 1
                set(state => ({
                    ...clearCallRequestState(),
                    callsMutationSeq: mutationSeq,
                    calls: (state.calls || []).filter(call => Number(call?.id) !== Number(callId)),
                    callsCache: removeCallAcrossCaches(state.callsCache, callId),
                    callsCacheFetchedAt: {},
                    callsLastFetchedAt: 0,
                }))
            }

            return {
                success: false,
                error: isMissingCallTask
                    ? 'This call task no longer exists. Refreshing the Calls workspace.'
                    : detail.message,
                errorCode: isMissingCallTask ? 'call_task_not_found' : detail.code,
                actionLabel: detail.actionLabel,
                actionUrl: detail.actionUrl,
                errorMeta: detail.meta,
                httpStatus: status,
            }
        }
    },

    // Add explicitly to help UI clear state on tab/list switch
    clearCallsState: () => set({
        ...clearCallRequestState(),
        calls: [],
        callsLastFetchedAt: 0,
        callsLastQueryKey: '',
    })
}), {
    name: 'app-storage-v2',
    partialize: (state) => ({
        user: state.user,
        heyreachCampaignId: state.heyreachCampaignId,
        isSidebarCollapsed: state.isSidebarCollapsed,
        // Do not persist analytics: stale zeros (failed fetch / cold cache) overwrite real counts after reload.
        tpAiRunFocus: state.tpAiRunFocus,
        searchQuery: state.searchQuery,
        tpScopeTotal: state.tpScopeTotal,
        tpScopeStatusCounts: state.tpScopeStatusCounts,
        tpTotal: state.tpTotal,
        tpStatusCounts: state.tpStatusCounts,
        // Talent Pool view state — filters/search/sort/page/scope survive a
        // refresh so the recruiter lands back on the exact same view.
        tpFilters: state.tpFilters,
        tpActiveStatusTab: state.tpActiveStatusTab,
        tpSortBy: state.tpSortBy,
        tpSortDir: state.tpSortDir,
        tpPage: state.tpPage,
        tpPageSize: state.tpPageSize,
        tpGlobalSearch: state.tpGlobalSearch,
        talentPoolViewScope: state.talentPoolViewScope,
        talentPoolRecruiterFilterId: state.talentPoolRecruiterFilterId,
        talentPoolRoleFilterId: state.talentPoolRoleFilterId,
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
            callListsRequestSeq: 0,
            callsRequest: null,
            callsRequestQueryKey: '',
            callsRequestSeq: 0,
            callsMutationSeq: 0,
            callStatsRequest: null,
            callStatsRequestSeq: 0,
            tpScopeTotal: nextState.tpScopeTotal,
            tpScopeStatusCounts: nextState.tpScopeStatusCounts || {},
            tpScopeSummaryRequest: null,
            tpScopeSummaryRequestParamsString: '',
            analyticsUserKey: '',
            analyticsRequestUserKey: '',
            tpCandidates: currentState.tpCandidates,
            tpTotal: nextState.tpTotal,
            tpStatusCounts: nextState.tpStatusCounts || {},
            talentPoolCache: currentState.talentPoolCache,
            searchResults: currentState.searchResults,
            talentPoolRequest: null,
            talentPoolRequestParamsString: '',
            talentPoolIndexRequest: null,
            talentPoolIndexRequestParamsString: '',
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
