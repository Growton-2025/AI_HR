import axios from 'axios';

/** Matches slow AI routes (generate / test / field-catalog / delete). Global axios.defaults.timeout is 15s. */
const LONG_OPERATION_TIMEOUT_MS = 600000;

export const longOperationAxios = axios.create({
  timeout: LONG_OPERATION_TIMEOUT_MS,
});

longOperationAxios.interceptors.request.use((config) => {
  const auth = axios.defaults.headers.common?.Authorization;
  if (auth) {
    config.headers = config.headers || {};
    config.headers.Authorization = auth;
  }
  return config;
});
