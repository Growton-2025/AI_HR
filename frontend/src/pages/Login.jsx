import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../store/useAppStore'
import { Lock, Mail, ArrowRight, CheckCircle, Loader2, User, Phone, Eye, EyeOff, TrendingUp, Users, BarChart3 } from 'lucide-react'
import { toast } from 'sonner'
import { useShallow } from 'zustand/react/shallow'
import HayasaBrand from '../components/HayasaBrand'

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || ''

const styles = {
  page: {
    display: 'flex',
    height: '100vh',
    width: '100vw',
    overflow: 'hidden',
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
  },
  leftPanel: {
    width: '45%',
    background: 'linear-gradient(155deg, #111827 0%, #1f2937 50%, #7c5a2f 100%)',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    padding: '48px',
    color: 'white',
    position: 'relative',
    overflow: 'hidden',
  },
  rightPanel: {
    flex: 1,
    background: 'linear-gradient(180deg, #f3f5f7 0%, #ebeff3 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '32px',
    overflowY: 'auto',
  },
  formCard: {
    width: '100%',
    maxWidth: '420px',
    background: 'rgba(255,255,255,0.92)',
    border: '1px solid rgba(226,232,240,0.92)',
    borderRadius: '28px',
    padding: '40px',
    boxShadow: '0 22px 50px rgba(15,23,42,0.10)',
    backdropFilter: 'blur(16px)',
  },
  label: {
    display: 'block',
    color: '#64748b',
    fontSize: '12px',
    fontWeight: 600,
    marginBottom: '8px',
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
  },
  inputWrap: {
    position: 'relative',
    marginBottom: '20px',
  },
  input: {
    width: '100%',
    padding: '14px 16px 14px 46px',
    background: '#ffffff',
    border: '1px solid rgba(203,213,225,0.92)',
    borderRadius: '12px',
    color: '#0f172a',
    fontSize: '15px',
    outline: 'none',
    transition: 'border-color 0.2s, box-shadow 0.2s',
    fontFamily: 'inherit',
  },
  inputIcon: {
    position: 'absolute',
    left: '14px',
    top: '50%',
    transform: 'translateY(-50%)',
    color: '#94a3b8',
    display: 'flex',
    alignItems: 'center',
  },
  inputEyeBtn: {
    position: 'absolute',
    right: '14px',
    top: '50%',
    transform: 'translateY(-50%)',
    background: 'none',
    border: 'none',
    color: '#94a3b8',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    padding: 0,
  },
  primaryBtn: {
    width: '100%',
    padding: '15px',
    background: '#111827',
    color: 'white',
    border: '1px solid #111827',
    borderRadius: '12px',
    fontSize: '15px',
    fontWeight: 700,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    transition: 'all 0.2s',
    letterSpacing: '0.01em',
    boxShadow: '0 14px 28px rgba(15, 23, 42, 0.14)',
  },
  divider: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    margin: '24px 0',
  },
  dividerLine: {
    flex: 1,
    height: '1px',
    background: '#e2e8f0',
  },
  dividerText: {
    color: '#94a3b8',
    fontSize: '12px',
    fontWeight: 600,
    letterSpacing: '0.05em',
    whiteSpace: 'nowrap',
  },
  switchText: {
    textAlign: 'center',
    color: '#64748b',
    fontSize: '14px',
    marginTop: '28px',
    lineHeight: 1.6,
  },
  switchLink: {
    background: 'none',
    border: 'none',
    color: '#8b6b44',
    cursor: 'pointer',
    fontWeight: 700,
    fontSize: '14px',
    padding: 0,
  },
  otpInput: {
    width: '100%',
    padding: '18px',
    background: '#ffffff',
    border: '1px solid rgba(203,213,225,0.92)',
    borderRadius: '12px',
    color: '#0f172a',
    fontSize: '28px',
    letterSpacing: '10px',
    textAlign: 'center',
    outline: 'none',
    fontWeight: 800,
    fontFamily: 'monospace',
    transition: 'border-color 0.2s',
  },
}

function InputField({ icon: Icon, label, type = 'text', value, onChange, placeholder, required, hasEye }) {
  const [showPwd, setShowPwd] = useState(false)
  const [focused, setFocused] = useState(false)
  const actualType = hasEye ? (showPwd ? 'text' : 'password') : type

  return (
    <div style={{ marginBottom: '20px' }}>
      {label && <label style={styles.label}>{label}</label>}
      <div style={{ position: 'relative' }}>
        <span style={styles.inputIcon}>
          <Icon size={17} />
        </span>
        <input
          type={actualType}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          required={required}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          style={{
            ...styles.input,
            paddingRight: hasEye ? '46px' : '16px',
            borderColor: focused ? 'rgba(194,124,63,0.45)' : 'rgba(203,213,225,0.92)',
            boxShadow: focused ? '0 0 0 3px rgba(194,124,63,0.10)' : 'none',
          }}
        />
        {hasEye && (
          <button type="button" onClick={() => setShowPwd(p => !p)} style={styles.inputEyeBtn}>
            {showPwd ? <EyeOff size={17} /> : <Eye size={17} />}
          </button>
        )}
      </div>
    </div>
  )
}

function Login() {
  const navigate = useNavigate()
  const { login, loginWithGoogle, register, verifyOtp, resendOtp, isAuthenticated } = useAppStore(useShallow((state) => ({
    login: state.login,
    loginWithGoogle: state.loginWithGoogle,
    register: state.register,
    verifyOtp: state.verifyOtp,
    resendOtp: state.resendOtp,
    isAuthenticated: state.isAuthenticated,
  })))

  const [mode, setMode] = useState('login')
  const [isLoading, setIsLoading] = useState(false)
  const [formData, setFormData] = useState({ name: '', email: '', password: '', phone: '' })
  const [otp, setOtp] = useState('')
  const [otpFocused, setOtpFocused] = useState(false)
  const googleButtonRef = useRef(null)
  const currentYear = new Date().getFullYear()

  const renderGoogleButton = useCallback(() => {
    if (!window.google?.accounts?.id || !googleButtonRef.current) return

    const buttonWidth = Math.max(280, Math.floor(googleButtonRef.current.getBoundingClientRect().width))
    googleButtonRef.current.innerHTML = ''

    window.google.accounts.id.renderButton(googleButtonRef.current, {
      theme: 'filled_black',
      size: 'large',
      width: buttonWidth,
      shape: 'rectangular',
      text: 'continue_with',
    })
  }, [])

  useEffect(() => {
    if (isAuthenticated) navigate('/')
  }, [isAuthenticated, navigate])

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID || mode !== 'login') return
    let resizeObserver
    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.defer = true
    script.onload = () => {
      window.google?.accounts.id.initialize({ client_id: GOOGLE_CLIENT_ID, callback: handleGoogleCallback })
      renderGoogleButton()

      if (googleButtonRef.current && 'ResizeObserver' in window) {
        resizeObserver = new window.ResizeObserver(() => {
          renderGoogleButton()
        })
        resizeObserver.observe(googleButtonRef.current)
      }
    }
    document.body.appendChild(script)
    return () => {
      resizeObserver?.disconnect()
      if (googleButtonRef.current) {
        googleButtonRef.current.innerHTML = ''
      }
      try { document.body.removeChild(script) } catch (e) { }
    }
  }, [mode, renderGoogleButton])

  const handleGoogleCallback = async (response) => {
    setIsLoading(true)
    const id = toast.loading('Signing in with Google...')
    try {
      const res = await loginWithGoogle(response.credential)
      if (res.success) { toast.success('Welcome back!', { id }); setTimeout(() => navigate('/'), 500) }
      else toast.error(res.error || 'Google login failed.', { id })
    } catch { toast.error('An error occurred.', { id }) }
    finally { setIsLoading(false) }
  }

  const handleLogin = async (e) => {
    e.preventDefault()
    if (!formData.email || !formData.password) { toast.error('Please fill in all fields.'); return }
    setIsLoading(true)
    try {
      const res = await login(formData.email, formData.password)
      if (res.success) { toast.success('Welcome back!'); navigate('/') }
      else toast.error(res.error || 'Invalid credentials.')
    } catch { toast.error('Login failed. Please try again.') }
    finally { setIsLoading(false) }
  }

  const handleRegister = async (e) => {
    e.preventDefault()
    if (!formData.email || !formData.name || !formData.password) { toast.error('Please fill in all required fields.'); return }
    setIsLoading(true)
    try {
      const res = await register(formData.name, formData.email, formData.phone, formData.password)
      if (res.success) { toast.success('Account created! Please verify your email.'); setMode('verify') }
      else toast.error(res.error || 'Registration failed.')
    } catch { toast.error('Registration failed.') }
    finally { setIsLoading(false) }
  }

  const handleVerify = async (e) => {
    e.preventDefault()
    if (!otp || otp.length < 6) { toast.error('Please enter the 6-digit code.'); return }
    setIsLoading(true)
    try {
      const res = await verifyOtp(formData.email, otp)
      if (res.success) { toast.success('Email verified! Logging you in...'); setTimeout(() => navigate('/'), 1000) }
      else toast.error(res.error || 'Invalid code.')
    } catch { toast.error('Verification failed.') }
    finally { setIsLoading(false) }
  }

  const handleResendOtp = async () => {
    setIsLoading(true)
    try {
      const res = await resendOtp(formData.email)
      if (res.success) toast.success('Code resent!')
      else toast.error(res.error || 'Failed to resend code.')
    } finally { setIsLoading(false) }
  }

  const features = [
    { icon: TrendingUp, text: 'AI-Powered Candidate Matching' },
    { icon: Users, text: 'Automated Screening Calls' },
    { icon: BarChart3, text: 'Smart Role Management' },
  ]

  return (
    <div style={styles.page}>
      {/* ── Left Branding Panel ── */}
      <div style={styles.leftPanel}>
        {/* Background glow */}
        <div style={{ position: 'absolute', top: '-20%', right: '-20%', width: '500px', height: '500px', borderRadius: '50%', background: 'rgba(255,255,255,0.05)', filter: 'blur(80px)', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', bottom: '-10%', left: '-10%', width: '350px', height: '350px', borderRadius: '50%', background: 'rgba(124,92,47,0.28)', filter: 'blur(80px)', pointerEvents: 'none' }} />

        {/* Logo */}
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ maxWidth: '100%', padding: '12px 14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(255,255,255,0.08)', boxShadow: '0 14px 32px rgba(0,0,0,0.16)' }}>
            <HayasaBrand size="hero" tone="dark" />
          </div>
        </div>

        {/* Hero Text */}
        <div style={{ position: 'relative' }}>
          <p style={{ fontSize: '12px', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', opacity: 0.7, marginBottom: '16px' }}>Talent Intelligence Platform</p>
          <h1 style={{ fontSize: '44px', fontWeight: 900, lineHeight: 1.1, marginBottom: '24px', letterSpacing: '-1.5px' }}>
            Hire the best,<br />10× faster.
          </h1>
          <p style={{ fontSize: '16px', opacity: 0.75, lineHeight: 1.7, maxWidth: '380px', marginBottom: '40px' }}>
            AI-driven sourcing, screening and outreach—all in one elegant platform for modern recruiting teams.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {features.map(({ icon: Icon, text }, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <div style={{ width: '36px', height: '36px', background: 'rgba(255,255,255,0.15)', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Icon size={18} color="white" />
                </div>
                <span style={{ fontSize: '15px', fontWeight: 500, opacity: 0.9 }}>{text}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div style={{ position: 'relative', opacity: 0.45, fontSize: '12px' }}>
          © {currentYear} Hayasa.ai · a growton.co product
        </div>
      </div>

      {/* ── Right Form Panel ── */}
      <div style={styles.rightPanel}>
        <div style={styles.formCard}>

          {/* Header */}
          <div style={{ marginBottom: '36px' }}>
            <h2 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a', marginBottom: '8px', letterSpacing: '-0.5px' }}>
              {mode === 'login' && 'Welcome back'}
              {mode === 'register' && 'Create your account'}
              {mode === 'verify' && 'Check your email'}
            </h2>
            <p style={{ color: '#64748b', fontSize: '15px', lineHeight: 1.5 }}>
              {mode === 'login' && 'Sign in to your Hayasa.ai workspace'}
              {mode === 'register' && 'Join recruiters using Hayasa.ai'}
              {mode === 'verify' && `We sent a code to ${formData.email}`}
            </p>
          </div>

          {/* LOGIN FORM */}
          {mode === 'login' && (
            <form onSubmit={handleLogin}>
              <InputField icon={Mail} label="Work Email" type="email" value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} placeholder="you@company.com" required />
              <InputField icon={Lock} label="Password" value={formData.password} onChange={e => setFormData({ ...formData, password: e.target.value })} placeholder="Your password" required hasEye />

              <button type="submit" disabled={isLoading} style={{ ...styles.primaryBtn, marginTop: '8px', opacity: isLoading ? 0.7 : 1 }}>
                {isLoading ? <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} /> : <><span>Sign In</span><ArrowRight size={18} /></>}
              </button>

              <div style={styles.divider}>
                <div style={styles.dividerLine} />
                <span style={styles.dividerText}>OR CONTINUE WITH</span>
                <div style={styles.dividerLine} />
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', minHeight: '44px' }}>
                {GOOGLE_CLIENT_ID
                  ? <div ref={googleButtonRef} style={{ width: '100%' }} />
                  : <p style={{ color: '#52525b', fontSize: '13px', textAlign: 'center' }}>Google Sign-In unavailable (Client ID missing)</p>
                }
              </div>

              <p style={styles.switchText}>
                New to Hayasa.ai?{' '}
                <button type="button" onClick={() => setMode('register')} style={styles.switchLink}>Create account</button>
              </p>
            </form>
          )}

          {/* REGISTER FORM */}
          {mode === 'register' && (
            <form onSubmit={handleRegister}>
              <InputField icon={User} label="Full Name" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} placeholder="John Smith" required />
              <InputField icon={Mail} label="Work Email" type="email" value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} placeholder="you@company.com" required />
              <InputField icon={Lock} label="Password" value={formData.password} onChange={e => setFormData({ ...formData, password: e.target.value })} placeholder="Create a strong password" required hasEye />
              <InputField icon={Phone} label="Phone (optional)" type="tel" value={formData.phone} onChange={e => setFormData({ ...formData, phone: e.target.value })} placeholder="+1 555 000 0000" />

              <button type="submit" disabled={isLoading} style={{ ...styles.primaryBtn, marginTop: '8px', opacity: isLoading ? 0.7 : 1 }}>
                {isLoading ? <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} /> : 'Create Account'}
              </button>

              <p style={styles.switchText}>
                Already have an account?{' '}
                <button type="button" onClick={() => setMode('login')} style={styles.switchLink}>Sign in</button>
              </p>
            </form>
          )}

          {/* VERIFY FORM */}
          {mode === 'verify' && (
            <form onSubmit={handleVerify}>
              <div style={{ textAlign: 'center', marginBottom: '28px' }}>
                <div style={{ width: '72px', height: '72px', background: '#f8fafc', border: '1px solid rgba(203,213,225,0.92)', borderRadius: '20px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginBottom: '20px' }}>
                  <Mail size={32} color="#8b6b44" />
                </div>
              </div>

              <div style={{ marginBottom: '28px' }}>
                <label style={styles.label}>6-Digit Code</label>
                <input
                  type="text"
                  value={otp}
                  onChange={e => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                  placeholder="000000"
                  maxLength={6}
                  onFocus={() => setOtpFocused(true)}
                  onBlur={() => setOtpFocused(false)}
                  style={{
                    ...styles.otpInput,
                    borderColor: otpFocused ? 'rgba(194,124,63,0.45)' : 'rgba(203,213,225,0.92)',
                    boxShadow: otpFocused ? '0 0 0 3px rgba(194,124,63,0.10)' : 'none',
                  }}
                />
              </div>

              <button type="submit" disabled={isLoading} style={{ ...styles.primaryBtn, opacity: isLoading ? 0.7 : 1 }}>
                {isLoading ? <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} /> : <><span>Verify Email</span><CheckCircle size={18} /></>}
              </button>

              <p style={{ ...styles.switchText, marginTop: '20px' }}>
                Didn't receive it?{' '}
                <button type="button" onClick={handleResendOtp} disabled={isLoading} style={styles.switchLink}>Resend code</button>
              </p>
              <p style={styles.switchText}>
                <button type="button" onClick={() => setMode('login')} style={{ ...styles.switchLink, color: '#71717a', fontWeight: 500 }}>← Back to sign in</button>
              </p>
            </form>
          )}
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

export default Login
