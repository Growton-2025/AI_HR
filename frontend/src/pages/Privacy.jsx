import React from 'react'

const CONTACT_EMAIL = 'nethranandpsaiml@gmail.com'
const LAST_UPDATED = '19 November 2026'

const h2 = { fontSize: 18, fontWeight: 800, color: '#0f172a', margin: '28px 0 8px' }
const p = { fontSize: 14, lineHeight: 1.7, color: '#334155', margin: '0 0 10px' }
const li = { fontSize: 14, lineHeight: 1.7, color: '#334155', marginBottom: 4 }

/** Public privacy policy (no sign-in required); linked from the Google consent screen. */
export default function Privacy() {
    return (
        <div style={{ minHeight: '100vh', background: '#f8fafc', padding: '40px 20px' }}>
            <div style={{ maxWidth: 760, margin: '0 auto', background: '#fff', border: '1px solid #e2e8f0', borderRadius: 16, padding: '32px 36px' }}>
                <div style={{ fontSize: 13, fontWeight: 800, color: '#9a3412', letterSpacing: 0.4 }}>HAYASA.AI · A GROWTON PRODUCT</div>
                <h1 style={{ fontSize: 28, fontWeight: 800, color: '#0f172a', margin: '8px 0 4px' }}>Privacy Policy</h1>
                <div style={{ fontSize: 12, color: '#64748b' }}>Last updated: {LAST_UPDATED}</div>

                <h2 style={h2}>What Hayasa is</h2>
                <p style={p}>
                    Hayasa is a recruiting tool used by recruiters to source, screen and contact candidates. This policy
                    explains what information the app handles and how it is used.
                </p>

                <h2 style={h2}>Information we handle</h2>
                <ul style={{ paddingLeft: 20, margin: 0 }}>
                    <li style={li}><b>Account details:</b> the name and email address of each recruiter who signs in.</li>
                    <li style={li}><b>Candidate information:</b> names, contact details, work history and notes that recruiters upload, import or enter, including publicly available professional profile information.</li>
                    <li style={li}><b>Activity records:</b> call logs, messages and status changes made in the app.</li>
                </ul>

                <h2 style={h2}>Google account data</h2>
                <p style={p}>
                    If you choose to connect your Google account, Hayasa asks for read-only access to your Google Sheets and
                    your Google account email address. This access is used only to:
                </p>
                <ul style={{ paddingLeft: 20, margin: 0 }}>
                    <li style={li}>read the spreadsheet you select, so its rows can be imported as candidates into your Hayasa workspace;</li>
                    <li style={li}>read that same spreadsheet again when you press “Sync now”;</li>
                    <li style={li}>show you which Google account is connected.</li>
                </ul>
                <p style={{ ...p, marginTop: 10 }}>
                    Hayasa does not edit or delete your spreadsheets and does not read any spreadsheet you have not chosen
                    to import. The connection token is stored encrypted. You can disconnect at any time from the import
                    window, or from your Google Account under third-party access, after which Hayasa can no longer read
                    your sheets.
                </p>
                <p style={p}>
                    Hayasa’s use and transfer of information received from Google APIs adheres to the Google API Services
                    User Data Policy, including the Limited Use requirements. Google user data is not sold, is not used
                    for advertising, and is not used to train general-purpose AI models. It is not transferred to others
                    except as needed to provide the import feature, to comply with the law, or as part of a merger or
                    acquisition with notice to users.
                </p>

                <h2 style={h2}>How information is used</h2>
                <p style={p}>
                    Information is used to run the recruiting features of the app: storing and searching candidate
                    profiles, enriching profiles with professional background details, and supporting outreach by
                    recruiters. Service providers that host the app, store its data, place calls, send messages or
                    process profile information handle data only to provide those services.
                </p>

                <h2 style={h2}>Storage and security</h2>
                <p style={p}>
                    Data is stored on cloud infrastructure with access limited to authorised users. Access to the app
                    requires sign-in, and connections to it are encrypted.
                </p>

                <h2 style={h2}>Retention and deletion</h2>
                <p style={p}>
                    Data is kept while the workspace is in use. To ask for your account, your Google connection, or
                    information about you to be removed, contact us at the address below.
                </p>

                <h2 style={h2}>Contact</h2>
                <p style={p}>
                    Questions or requests about this policy: <a href={`mailto:${CONTACT_EMAIL}`} style={{ color: '#c2410c', fontWeight: 700 }}>{CONTACT_EMAIL}</a>
                </p>

                <div style={{ marginTop: 28, paddingTop: 16, borderTop: '1px solid #e2e8f0' }}>
                    <a href="/login" style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', textDecoration: 'none' }}>← Back to Hayasa</a>
                </div>
            </div>
        </div>
    )
}
