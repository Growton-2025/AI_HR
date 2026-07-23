// Slicer (date-range / outcome) options for the Calls workspace.
// Values mirror the backend contract in backend/api/routes/calls.py
// (RANGE_PRESETS / OUTCOME_GROUPS) — keep the two in sync.

export const RANGE_OPTIONS = [
  { value: 'all', label: 'All Dates' },
  { value: 'today', label: 'Today' },
  { value: 'yesterday', label: 'Yesterday' },
  { value: 'last7', label: 'Last 7 Days' },
  { value: 'last30', label: 'Last 30 Days' },
];

export const RANGE_DROPDOWN_OPTIONS = [
  ...RANGE_OPTIONS,
  { value: 'custom', label: 'Custom Date Range' },
];

export const OUTCOME_GROUP_OPTIONS = [
  { value: '', label: 'All Outcomes' },
  { value: 'connected', label: 'Connected' },
  { value: 'followup', label: 'Follow-up' },
  { value: 'not_connected', label: 'Not Connected / Voicemail' },
];

const SCOPE_LABELS = {
  all: 'All Time',
  today: 'Today',
  yesterday: 'Yesterday',
  last7: 'Last 7 Days',
  last30: 'Last 30 Days',
  custom: 'Custom Range',
};

export function rangeScopeLabel(range) {
  return SCOPE_LABELS[range] || SCOPE_LABELS.all;
}

// Query params for /api/calls and /api/calls/stats. A custom range is only
// sent once BOTH dates are picked, so a half-filled picker never fires a 400.
export function buildSlicerParams({ range, customFrom, customTo, outcomeGroup }) {
  const params = {};
  if (range && range !== 'all') {
    if (range === 'custom') {
      if (customFrom && customTo) {
        params.range = 'custom';
        params.date_from = customFrom;
        params.date_to = customTo;
      }
    } else {
      params.range = range;
    }
  }
  if (outcomeGroup) params.outcome_group = outcomeGroup;
  return params;
}

export function isSlicerDefault({ range, outcomeGroup }) {
  return (!range || range === 'all') && !outcomeGroup;
}

const HEADER_DATE_FORMAT = new Intl.DateTimeFormat('en-US', {
  weekday: 'long',
  year: 'numeric',
  month: 'long',
  day: 'numeric',
});

export function formatHeaderDate(d) {
  return HEADER_DATE_FORMAT.format(d);
}
