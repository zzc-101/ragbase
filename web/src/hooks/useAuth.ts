import { useSyncExternalStore } from 'react'
import { getAuthState, subscribeAuth } from '@/lib/auth'

export function useAuth() {
  return useSyncExternalStore(subscribeAuth, getAuthState, getAuthState)
}
