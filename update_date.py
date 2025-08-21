import os
import re
from datetime import datetime

POSTS_DIR = "_posts"

# 匹配 update 和 date 字段
update_pattern = re.compile(
    r'^(update:\s*)(\d{4}-\d{1,2}-\d{1,2}(?: \d{1,2}:\d{2}(?::\d{2})?)?)',
    re.MULTILINE
)
date_pattern = re.compile(
    r'^(date:\s*)(\d{4}-\d{1,2}-\d{1,2}(?: \d{1,2}:\d{2}(?::\d{2})?)?)',
    re.MULTILINE
)

def normalize_datetime(dt_str):
    """把不规范的日期字符串转成 YYYY-MM-DD HH:MM:SS"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    # 最后兜底：fromisoformat 支持 2025-1-7 这种
    try:
        return datetime.fromisoformat(dt_str).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return dt_str

for root, _, files in os.walk(POSTS_DIR):
    for filename in files:
        if filename.endswith((".md", ".markdown")):
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            changed = False

            # 先处理已有的 update 字段
            def replacer(match):
                prefix, dt_str = match.groups()
                return prefix + normalize_datetime(dt_str)

            new_content = update_pattern.sub(replacer, content)
            if new_content != content:
                changed = True
                content = new_content

            # 如果没有 update 字段，就尝试补充
            if not update_pattern.search(content):
                date_match = date_pattern.search(content)
                if date_match:
                    _, date_val = date_match.groups()
                    norm_date = normalize_datetime(date_val)
                    # 插在 date 后面一行
                    insert_pos = date_match.end()
                    content = (
                        content[:insert_pos] + f"\nupdate: {norm_date}" + content[insert_pos:]
                    )
                    changed = True

            if changed:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ 已更新: {filepath}")
