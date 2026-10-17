/** sessionStorage / localStorage 的安全包装：隐私模式或被禁用时静默失败。 */
export const safeStorage = {
  get(kind: 'session' | 'local', key: string): string | null {
    try {
      return (kind === 'session' ? sessionStorage : localStorage).getItem(key)
    } catch {
      return null
    }
  },
  set(kind: 'session' | 'local', key: string, value: string | null) {
    try {
      const store = kind === 'session' ? sessionStorage : localStorage
      if (value === null) store.removeItem(key)
      else store.setItem(key, value)
    } catch {
      // 忽略
    }
  },
}
