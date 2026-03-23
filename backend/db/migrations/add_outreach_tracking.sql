-- Migration: Add Candidate Outreach Tracking
-- Purpose: Track Smartlead AI campaigns per candidate per role

CREATE TABLE IF NOT EXISTS candidate_outreach (
    id SERIAL PRIMARY KEY,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    recruitment_role_id INTEGER REFERENCES recruitment_roles(id) ON DELETE CASCADE,
    campaign_id VARCHAR(255),
    campaign_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    message_sent_count INTEGER DEFAULT 0,
    last_message_sent_at TIMESTAMP,
    response_received_at TIMESTAMP,
    response_text TEXT,
    li_status VARCHAR(50), -- connection_sent, connection_accepted, message_sent, replied
    li_last_action_at TIMESTAMP,
    li_response_text TEXT,
    heyreach_campaign_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(candidate_id, recruitment_role_id)
);

CREATE INDEX idx_candidate_outreach_candidate ON candidate_outreach(candidate_id);
CREATE INDEX idx_candidate_outreach_role ON candidate_outreach(recruitment_role_id);
CREATE INDEX idx_candidate_outreach_status ON candidate_outreach(status);

COMMENT ON TABLE candidate_outreach IS 'Tracks Smartlead AI outreach campaigns per candidate per role';
COMMENT ON COLUMN candidate_outreach.status IS 'Smartlead status Values: pending, sent, replied, bounced, unsubscribed';
COMMENT ON COLUMN candidate_outreach.li_status IS 'HeyReach status Values: connection_sent, connection_accepted, message_sent, replied';
