import { ChevronLeft, ChevronRight } from 'lucide-react'

// Client-side pager shared by long lists (shortlist results, calls tables).
// Renders "Showing a–b of N", a per-page selector, and prev / numbered / next
// buttons with an ellipsis either side of the current page. Nothing is drawn
// for an empty list; a list shorter than the smallest page size still shows
// the count and the selector but no page buttons.
export function pageWindow(total, page, pageSize) {
    const totalPages = Math.max(1, Math.ceil(total / pageSize))
    const current = Math.min(Math.max(1, page), totalPages)
    const start = total === 0 ? 0 : (current - 1) * pageSize + 1
    const end = Math.min(current * pageSize, total)
    return { totalPages, current, start, end }
}

export default function Pagination({
    page, pageSize, total,
    onPageChange, onPageSizeChange,
    pageSizeOptions = [10, 25, 50, 100],
    noun = 'items',
    range = 2,
}) {
    if (!total) return null
    const { totalPages, current, start, end } = pageWindow(total, page, pageSize)
    const showPages = totalPages > 1
    const buttonStyle = (active, disabled) => ({
        width: 34, height: 34, borderRadius: 10, fontSize: 13, fontWeight: active ? 700 : 600,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: active ? '#f97316' : '#fff', color: active ? '#fff' : '#64748b',
        border: active ? 'none' : '1px solid rgba(203, 213, 225, 0.9)',
        cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.4 : 1,
    })

    const pages = []
    if (showPages) {
        for (let i = 1; i <= totalPages; i++) {
            if (i === 1 || i === totalPages || (i >= current - range && i <= current + range)) {
                pages.push(
                    <button key={i} type="button" onClick={() => onPageChange(i)} style={buttonStyle(i === current, false)} aria-current={i === current ? 'page' : undefined}>
                        {i}
                    </button>
                )
            } else if (i === current - range - 1 || i === current + range + 1) {
                pages.push(<span key={`gap-${i}`} style={{ color: '#94a3b8', margin: '0 4px' }}>…</span>)
            }
        }
    }

    return (
        <div style={{
            padding: '14px 18px', background: 'rgba(248,250,252,0.78)', borderTop: '1px solid #eef2f7',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12,
            borderBottomLeftRadius: 12, borderBottomRightRadius: 12,
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 13, color: '#64748b' }}>
                    Showing {start.toLocaleString()}–{end.toLocaleString()} of <strong style={{ color: '#0f172a' }}>{total.toLocaleString()}</strong> {noun}
                </span>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#64748b' }}>
                    Per page
                    <select
                        value={pageSize}
                        onChange={e => onPageSizeChange(Number(e.target.value))}
                        style={{
                            padding: '4px 8px', borderRadius: 8, border: '1px solid rgba(203, 213, 225, 0.9)',
                            fontSize: 12, fontWeight: 600, color: '#0f172a', outline: 'none', background: '#fff', cursor: 'pointer',
                        }}
                    >
                        {pageSizeOptions.map(size => <option key={size} value={size}>{size}</option>)}
                    </select>
                </label>
            </div>
            {showPages && (
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <button type="button" onClick={() => onPageChange(Math.max(1, current - 1))} disabled={current === 1} style={buttonStyle(false, current === 1)} aria-label="Previous page">
                        <ChevronLeft size={14} color="#64748b" />
                    </button>
                    {pages}
                    <button type="button" onClick={() => onPageChange(Math.min(totalPages, current + 1))} disabled={current === totalPages} style={buttonStyle(false, current === totalPages)} aria-label="Next page">
                        <ChevronRight size={14} color="#64748b" />
                    </button>
                </div>
            )}
        </div>
    )
}
