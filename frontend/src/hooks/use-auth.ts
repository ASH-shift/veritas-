"use client";

import { useCallback, useSyncExternalStore } from "react";

// The JWT lives in localStorage. Access is wrapped in try/catch because
// storage can be blocked (private mode, disabled site data).
const KEY = "veritas_token";
const listeners = new Set<() => void>();

function read(): string | null {
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

function write(value: string | null) {
  try {
    if (value) window.localStorage.setItem(KEY, value);
    else window.localStorage.removeItem(KEY);
  } catch {
    // storage unavailable: token only lives for this page view
  }
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  window.addEventListener("storage", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

/**
 * Returns { token, ready, setToken, logout }.
 * `ready` is false during server render / first hydration pass, so pages
 * can avoid redirecting before localStorage has been read.
 */
export function useAuth() {
  const token = useSyncExternalStore(subscribe, read, () => null);
  const ready = useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
  const setToken = useCallback((t: string | null) => write(t), []);
  const logout = useCallback(() => write(null), []);
  return { token, ready, setToken, logout };
}
