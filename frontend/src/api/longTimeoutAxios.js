import axios from 'axios';

/** Matches slow AI routes (generate / test / field-catalog / delete). Global axios.defaults.timeout is 15s. */
const LONG_OPERATION_TIMEOUT_MS = 600000;
const isDev = import.meta.env.DEV;

function shouldLogLongTiming(url = '') {
  return isDev && String(url || '').includes('/ai-columns');
}

export const longOperationAxios = axios.create({
  timeout: LONG_OPERATION_TIMEOUT_MS,
});

longOperationAxios.interceptors.request.use((config) => {
  if (shouldLogLongTiming(config.url)) {
    config._timingStart = performance.now();
  }
  const auth = axios.defaults.headers.common?.Authorization;
  if (auth) {
    config.headers = config.headers || {};
    config.headers.Authorization = auth;
  }
  return config;
});

longOperationAxios.interceptors.response.use(
  (response) => {
    if (response.config?._timingStart && shouldLogLongTiming(response.config.url)) {
      const elapsedMs = Math.round(performance.now() - response.config._timingStart);
      console.info(`[api ${elapsedMs}ms] ${String(response.config.method || 'GET').toUpperCase()} ${response.config.url} ${response.status}`);
    }
    return response;
  },
  (error) => {
    const config = error?.config || {};
    if (config._timingStart && shouldLogLongTiming(config.url)) {
      const elapsedMs = Math.round(performance.now() - config._timingStart);
      console.info(`[api ${elapsedMs}ms] ${String(config.method || 'GET').toUpperCase()} ${config.url} ${error?.response?.status || error?.code || 'ERR'}`);
    }
    return Promise.reject(error);
  }
);
