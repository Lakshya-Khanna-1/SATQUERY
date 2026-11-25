const gateway = (import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '/api')).replace(/\/$/, '');
const proxy = (import.meta.env.VITE_N8N_URL ?? (import.meta.env.DEV ? 'http://127.0.0.1:5678' : '')).replace(/\/$/, '');

export const hasWebhook = Boolean(proxy);

export function apiUrl(tab, path) {
  if (tab === 'n8n' && proxy) {
    return proxy + (path === '/query' ? '/satquery' : path);
  }
  return gateway + path;
}

export function fetchSession(url, options = {}) {
  if (import.meta.env.DEV && !import.meta.env.VITE_API_URL && !import.meta.env.VITE_N8N_URL) return fetch(url, options);
  let token = localStorage.getItem('satquery_history_key');
  if (!token) {
    token = crypto.randomUUID();
    localStorage.setItem('satquery_history_key', token);
  }
  return fetch(url, { ...options, headers: { ...options.headers, 'X-Satquery-History': token } });
}
