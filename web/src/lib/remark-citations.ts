/** remark 插件：把正文里的 [n]、[1][2]、[1, 2] 变成 <citation-chip n="…"> 节点（代码块 / 行内代码里的不处理）。
 * 不依赖 unist 类型包，只声明用到的最小节点形状。 */

interface MdNode {
  type: string
  value?: string
  children?: MdNode[]
  data?: Record<string, unknown>
}

// 一组方括号：[3]、[1, 2]、[1，2]、[1、2]
const GROUP = /\[(\d{1,3}(?:\s*[,，、]\s*\d{1,3})*)\]/g

function chip(n: string): MdNode {
  return { type: 'citationRef', data: { hName: 'citation-chip', hProperties: { n } } }
}

function splitText(value: string): MdNode[] {
  const out: MdNode[] = []
  let last = 0
  for (const match of value.matchAll(GROUP)) {
    if (match.index > last) out.push({ type: 'text', value: value.slice(last, match.index) })
    for (const n of match[1].split(/[,，、]/)) out.push(chip(String(Number(n.trim()))))
    last = match.index + match[0].length
  }
  if (last === 0) return [{ type: 'text', value }]
  if (last < value.length) out.push({ type: 'text', value: value.slice(last) })
  return out
}

function transform(node: MdNode) {
  if (!node.children) return
  const next: MdNode[] = []
  for (const child of node.children) {
    if (child.type === 'text' && child.value) {
      next.push(...splitText(child.value))
    } else {
      if (child.type !== 'code' && child.type !== 'inlineCode') transform(child)
      next.push(child)
    }
  }
  node.children = next
}

export function remarkCitations() {
  return (tree: MdNode) => transform(tree)
}
