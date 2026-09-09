// The backend sends naive timestamps (no 'Z'/offset — Postgres columns are
// `timestamp without time zone`, and the DB session stores UTC wall-clock
// values into them). Without a timezone suffix, `new Date(...)` parses the
// string as browser-local time instead of UTC, so the raw UTC numbers get
// displayed unchanged and mislabeled (e.g. a 07:38 UTC value shows as
// "7:38 AM" instead of the correct 1:08 PM IST). Treat as UTC before use,
// and always format in IST so the displayed time doesn't depend on whatever
// timezone the viewer's machine happens to be set to.
export const DISPLAY_TIME_ZONE = 'Asia/Kolkata';

export function parseUtcTimestamp(isoString) {
  if (!isoString) return null;
  const hasTimezone = /Z$|[+-]\d{2}:?\d{2}$/.test(isoString);
  const date = new Date(hasTimezone ? isoString : `${isoString}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatIstDateTime(isoString, options = {}) {
  const date = parseUtcTimestamp(isoString);
  if (!date) return '';
  return date.toLocaleString(options.locale ?? 'en-GB', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    ...options,
    timeZone: DISPLAY_TIME_ZONE,
  });
}

export function formatIstDate(isoString, options = {}) {
  const date = parseUtcTimestamp(isoString);
  if (!date) return '';
  return date.toLocaleDateString(options.locale ?? 'en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    ...options,
    timeZone: DISPLAY_TIME_ZONE,
  });
}

export function formatIstTime(isoString, options = {}) {
  const date = parseUtcTimestamp(isoString);
  if (!date) return '';
  return date.toLocaleTimeString(options.locale ?? [], {
    hour: '2-digit',
    minute: '2-digit',
    ...options,
    timeZone: DISPLAY_TIME_ZONE,
  });
}
