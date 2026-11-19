// Loads Google Identity Services once and hands back `window.google`.
const GIS_SRC = 'https://accounts.google.com/gsi/client'
let loading = null

export function loadGoogleIdentity() {
    if (window.google?.accounts?.oauth2) return Promise.resolve(window.google)
    if (loading) return loading
    loading = new Promise((resolve, reject) => {
        const done = () => (window.google?.accounts?.oauth2
            ? resolve(window.google)
            : reject(new Error('Google sign-in did not load')))
        const existing = document.querySelector(`script[src="${GIS_SRC}"]`)
        if (existing) {
            // Another screen (login) already added the tag; wait for it.
            existing.addEventListener('load', done, { once: true })
            existing.addEventListener('error', () => reject(new Error('Google sign-in did not load')), { once: true })
            window.setTimeout(() => { if (window.google?.accounts?.oauth2) resolve(window.google) }, 1500)
            return
        }
        const script = document.createElement('script')
        script.src = GIS_SRC
        script.async = true
        script.defer = true
        script.onload = done
        script.onerror = () => reject(new Error('Google sign-in did not load'))
        document.head.appendChild(script)
    }).catch((err) => {
        loading = null
        throw err
    })
    return loading
}
