import { useEffect, useRef, useState } from 'react'

/**
 * Returns `value` only after it has stopped changing for `delayMs`.
 *
 * Filter inputs feed effects that call /candidates/browse. Passing the raw
 * input state straight into an effect's dependency array meant every keystroke
 * queued a request — typing a 6-character search fired 19 of them, because the
 * old 120ms timer was shorter than a normal inter-keystroke gap. Debouncing the
 * *value* (rather than only delaying inside the effect) collapses that to one
 * request once typing settles.
 */
export function useDebouncedValue(value, delayMs = 300) {
    const [debounced, setDebounced] = useState(value)

    useEffect(() => {
        if (Object.is(debounced, value)) return undefined
        const timer = window.setTimeout(() => setDebounced(value), delayMs)
        return () => window.clearTimeout(timer)
    }, [value, delayMs, debounced])

    return debounced
}

/**
 * Keeps one AbortController per caller, aborting the previous request before
 * starting the next. A sequence-number guard alone (the previous approach) only
 * discards a late *response* — the request still reaches the server and runs the
 * full query, so superseded keystrokes kept competing for the small DB pool.
 *
 * Returns a getter for a fresh signal; abort-on-unmount is handled for you.
 */
export function useAbortableRequest() {
    const controllerRef = useRef(null)

    const nextSignal = () => {
        if (controllerRef.current) controllerRef.current.abort()
        controllerRef.current = new AbortController()
        return controllerRef.current.signal
    }

    useEffect(() => () => {
        if (controllerRef.current) controllerRef.current.abort()
    }, [])

    return nextSignal
}

/** True when an error came from an aborted request rather than a real failure. */
export function isAbortError(error) {
    return error?.name === 'CanceledError'
        || error?.code === 'ERR_CANCELED'
        || error?.name === 'AbortError'
}
