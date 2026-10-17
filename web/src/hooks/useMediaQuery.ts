import { useSyncExternalStore } from 'react'

/** 订阅一条 CSS 媒体查询（用于侧栏 / 依据面板在窄屏下改成抽屉）。 */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (notify) => {
      const list = window.matchMedia(query)
      list.addEventListener('change', notify)
      return () => list.removeEventListener('change', notify)
    },
    () => window.matchMedia(query).matches,
    () => false,
  )
}
