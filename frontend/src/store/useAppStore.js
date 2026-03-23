import { create } from 'zustand'
import axios from 'axios'

// API base URL
// API base URL
// const API_BASE = '/api'
const API_BASE = '/api'

// WebSocket URL
// WebSocket URL
// const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
// const WS_URL = `${protocol}//${window.location.host}/api/ws/search`
const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/search`

// Global App Store
import { persist } from 'zustand/middleware'

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
            return { success: false, error: error.response?.data?.detail || 'Registration failed' }
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
            return { success: false, error: error.response?.data?.detail || 'Verification failed' }
        }
    },

    resendOtp: async (email) => {
        try {
            const response = await axios.post(`${API_BASE}/resend-otp`, { email })
            return { success: true, message: response.data.message }
        } catch (error) {
            console.error('Resend OTP failed:', error)
            return { success: false, error: error.response?.data?.detail || 'Failed to resend OTP' }
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
            return { success: true }
        } catch (error) {
            console.error('Login failed:', error)
            return { success: false, error: error.response?.data?.detail || 'Login failed' }
        }
    },

    logout: () => {
        localStorage.removeItem('token')
        delete axios.defaults.headers.common['Authorization']
        set({ token: null, isAuthenticated: false, user: null })
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
            return { success: true }
        } catch (error) {
            console.error('Google login failed:', error)
            return { success: false, error: error.response?.data?.detail || 'Google login failed' }
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
        const maxAgeMs = 60 * 1000
        const state = get()
        const isFresh = state.analytics && state.analyticsLastFetchedAt && (Date.now() - state.analyticsLastFetchedAt < maxAgeMs)

        if (!force && isFresh) {
            return { success: true, data: state.analytics, cached: true }
        }

        if (state.analyticsRequest) {
            return state.analyticsRequest
        }

        const request = axios.get(`${API_BASE}/candidates/analytics`)
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

    // Search State
    searchQuery: '',
    setSearchQuery: (query) => set({ searchQuery: query }),
    isSearching: false,
    searchProgress: 0,
    searchTotal: 0,
    statusMessage: '',
    searchResults: [],
    usage: null,

    // Search via REST (synchronous)
    searchCandidates: async (query) => {
        set({ isSearching: true, searchResults: [], statusMessage: 'Screening...', searchProgress: 0 })
        try {
            const res = await axios.post(`${API_BASE}/search`, { query })
            set({
                searchResults: res.data.candidates,
                usage: res.data.usage,
                statusMessage: `Found ${res.data.total} candidates`,
                isSearching: false,
                searchProgress: 100
            })
        } catch (e) {
            set({ statusMessage: 'Screening failed: ' + e.message, isSearching: false })
        }
    },

    // Search via WebSocket (streaming)
    searchCandidatesStream: (query) => {
        set({ isSearching: true, searchResults: [], statusMessage: 'Connecting...', searchProgress: 0, searchTotal: 0 })

        const ws = new WebSocket(WS_URL)

        ws.onopen = () => {
            ws.send(JSON.stringify({ query }))
            set({ statusMessage: 'Processing query...' })
        }

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)

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
                    set({
                        searchResults: data.candidates,
                        usage: data.usage,
                        statusMessage: `Found ${data.total} candidates`,
                        isSearching: false,
                        searchProgress: 100
                    })
                    ws.close()
                    break

                case 'error':
                    set({ statusMessage: 'Error: ' + data.message, isSearching: false })
                    ws.close()
                    break
            }
        }

        ws.onerror = (error) => {
            console.error('WebSocket error:', error)
            set({ statusMessage: 'Connection error', isSearching: false })
        }

        ws.onclose = () => {
            if (get().isSearching) {
                set({ isSearching: false })
            }
        }

        // Store ws reference for potential cancellation
        set({ _ws: ws })
    },

    stopSearch: () => {
        const ws = get()._ws
        if (ws) {
            ws.close()
        }
        set({ isSearching: false, statusMessage: 'Screening stopped' })
    },

    clearSearch: () => {
        set({ searchQuery: '', searchResults: [], statusMessage: '', searchProgress: 0, usage: null })
    },

    // Selection State
    selectedCandidates: {},
    candidatePriorities: {},
    candidateFeedback: {},

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
            get().fetchRoles()
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
            const rolesResult = await get().fetchRoles()
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
            set({ viewingRole: {
                ...prevViewingRole,
                candidates: prevViewingRole.candidates.filter(c => c.id !== candidateId)
            }})
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

    triggerHeyReachOutreach: async (payload) => {
        try {
            const res = await axios.post(`${API_BASE}/outreach/heyreach/trigger`, payload)
            return { success: true, data: res.data }
        } catch (error) {
            console.error('Failed to trigger HeyReach outreach:', error)
            return { success: false, error: error.response?.data?.detail || 'Failed to trigger HeyReach outreach' }
        }
    },

    fetchChatHistory: async (roleId, candidateId, platform = 'email') => {
        try {
            const endpoint = platform === 'linkedin' ? 'linkedin' : 'email'
            const res = await axios.get(`${API_BASE}/outreach/chat/${endpoint}/${roleId}/${candidateId}`)
            return { success: true, messages: res.data.messages }
        } catch (error) {
            console.error(`Failed to fetch ${platform} chat history:`, error)
            return { success: false, error: error.response?.data?.detail || `Failed to fetch ${platform} chat history` }
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

    // Talent Pool Cache
    talentPoolCache: { data: null, lastParamsString: null },

    fetchTalentPool: async (paramsString) => {
        const cache = get().talentPoolCache
        const isCached = cache.lastParamsString === paramsString && cache.data
        
        if (isCached) {
            // Silently background fetch to check for updates
            axios.get(`${API_BASE}/candidates/browse?${paramsString}`)
                .then(res => {
                    set({ talentPoolCache: { data: res.data, lastParamsString: paramsString } })
                })
                .catch(err => console.error("Background talent pool fetch failed", err))
                
            return { success: true, data: cache.data, cached: true }
        }

        try {
            const res = await axios.get(`${API_BASE}/candidates/browse?${paramsString}`)
            set({ talentPoolCache: { data: res.data, lastParamsString: paramsString } })
            return { success: true, data: res.data, cached: false }
        } catch (error) {
            console.error('Failed to fetch talent pool:', error)
            if (cache.data) return { success: true, data: cache.data, cached: true }
            return { success: false, error: 'Failed' }
        }
    }
}), {
    name: 'app-storage',
    partialize: (state) => ({
        // Persist roles for instant display on refresh (stale-while-revalidate)
        roles: state.roles,
        rolesLastFetchedAt: state.rolesLastFetchedAt,
        roleDetailsCache: state.roleDetailsCache, // Added for instant role load
        outreachStatusCache: state.outreachStatusCache, // Added for instant status load
        searchResults: state.searchResults,
        selectedCandidates: state.selectedCandidates,
        candidatePriorities: state.candidatePriorities,
        candidateFeedback: state.candidateFeedback,
        searchQuery: state.searchQuery,
        stats: state.stats,
        analytics: state.analytics,
        analyticsLastFetchedAt: state.analyticsLastFetchedAt,
        user: state.user,
        talentPoolCache: state.talentPoolCache,
        // Persist recruiters so list shows instantly on page load (stale-while-revalidate)
        recruiters: state.recruiters,
    }),
}))


// Initialize Axios header if token exists
const token = localStorage.getItem('token')
if (token) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
}
