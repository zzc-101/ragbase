/** 书名展示：原始文件名里常带扩展名和下载站后缀，去掉后更好读；完整文件名放在 tooltip 里。 */

const EXTENSION = /\.(pdf|docx|md|markdown|txt)$/i
const TRAILING_GROUP = /\s*[(（][^()（）]*(z-?lib|1lib|z-library)[^()（）]*[)）]\s*$/i

export function displayBookName(filename: string): string {
  let name = filename.replace(EXTENSION, '').trim()
  // 可能有多个尾部括号组，逐个去掉
  for (let previous = ''; previous !== name; ) {
    previous = name
    name = name.replace(TRAILING_GROUP, '').trim()
  }
  return name || filename
}

/** 引用来源的一行描述：书名 › 章节路径 › 第N页 */
export function citationLabel(source: string, headingPath: string | null, page: number | null): string[] {
  const parts = [displayBookName(source)]
  if (headingPath) parts.push(headingPath.replaceAll(' > ', ' › '))
  if (page != null) parts.push(`第 ${page} 页`)
  return parts
}
