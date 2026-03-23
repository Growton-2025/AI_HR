function ComingSoon({ title, description }) {
    return (
        <div style={{ width: '100%', position: 'relative', minHeight: '100vh' }}>
            <h2 className="screen-header">{title}</h2>

            <div className="result-banner coming-soon">
                <div className="result-banner-title">Coming Soon</div>
                <div className="result-banner-subtitle">{description}</div>
            </div>
        </div>
    )
}

export default ComingSoon
