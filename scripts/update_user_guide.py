#!/usr/bin/env python3
"""
Convert docs/USER_GUIDE.md to static/user_guide.html

Usage:
    python scripts/update_user_guide.py
"""

import re
from pathlib import Path


def markdown_to_html(md_content: str) -> str:
    """Convert markdown to HTML with basic formatting."""
    html = md_content

    # Escape HTML entities first (except for our conversions)
    # Skip this since we want to allow some HTML-like content

    # Convert code blocks (``` ... ```)
    html = re.sub(
        r'```(\w*)\n(.*?)```',
        lambda m: f'<pre><code class="language-{m.group(1)}">{m.group(2).strip()}</code></pre>',
        html,
        flags=re.DOTALL
    )

    # Convert inline code (`code`)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # Convert headers
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Convert bold and italic
    html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Convert blockquotes
    html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)

    # Convert tables
    def convert_table(match):
        lines = match.group(0).strip().split('\n')
        if len(lines) < 2:
            return match.group(0)

        # Parse header
        header_cells = [cell.strip() for cell in lines[0].split('|')[1:-1]]

        # Skip separator line (line with dashes)

        # Parse body rows
        body_rows = []
        for line in lines[2:]:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                body_rows.append(cells)

        # Build HTML table
        html_table = '<table>\n<thead>\n<tr>\n'
        for cell in header_cells:
            html_table += f'<th>{cell}</th>\n'
        html_table += '</tr>\n</thead>\n<tbody>\n'

        for row in body_rows:
            html_table += '<tr>\n'
            for cell in row:
                html_table += f'<td>{cell}</td>\n'
            html_table += '</tr>\n'

        html_table += '</tbody>\n</table>'
        return html_table

    # Match tables (lines starting with |)
    html = re.sub(
        r'(\|.+\|\n)+',
        convert_table,
        html
    )

    # Convert unordered lists
    def convert_list(match):
        items = match.group(0).strip().split('\n')
        html_list = '<ul>\n'
        for item in items:
            item_text = re.sub(r'^[\-\*]\s+', '', item.strip())
            if item_text:
                html_list += f'<li>{item_text}</li>\n'
        html_list += '</ul>'
        return html_list

    html = re.sub(r'(^[\-\*] .+\n?)+', convert_list, html, flags=re.MULTILINE)

    # Convert numbered lists
    def convert_numbered_list(match):
        items = match.group(0).strip().split('\n')
        html_list = '<ol>\n'
        for item in items:
            item_text = re.sub(r'^\d+\.\s+', '', item.strip())
            if item_text:
                html_list += f'<li>{item_text}</li>\n'
        html_list += '</ol>'
        return html_list

    html = re.sub(r'(^\d+\. .+\n?)+', convert_numbered_list, html, flags=re.MULTILINE)

    # Convert links [text](url)
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

    # Convert horizontal rules
    html = re.sub(r'^---+$', '<hr>', html, flags=re.MULTILINE)

    # Wrap paragraphs (lines not already in tags)
    lines = html.split('\n')
    result = []
    in_paragraph = False

    for line in lines:
        stripped = line.strip()

        # Check if line starts with an HTML tag
        is_tag = (stripped.startswith('<') and not stripped.startswith('<a ') and
                  not stripped.startswith('<strong') and not stripped.startswith('<em') and
                  not stripped.startswith('<code'))
        is_empty = not stripped

        if is_empty:
            if in_paragraph:
                result.append('</p>')
                in_paragraph = False
            result.append('')
        elif is_tag:
            if in_paragraph:
                result.append('</p>')
                in_paragraph = False
            result.append(line)
        else:
            if not in_paragraph:
                result.append('<p>')
                in_paragraph = True
            result.append(line)

    if in_paragraph:
        result.append('</p>')

    return '\n'.join(result)


def generate_html(content: str) -> str:
    """Generate full HTML document with styling."""

    html_content = markdown_to_html(content)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ham Radio Olympics User Guide</title>
    <style>
        :root {{
            --bg-primary: #f5f5f5;
            --bg-secondary: white;
            --text-primary: #333;
            --text-secondary: #666;
            --accent-color: #2c5282;
            --accent-hover: #1a365d;
            --border-color: #e2e8f0;
            --code-bg: #f7fafc;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--bg-primary);
            margin: 0;
            padding: 0;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg-secondary);
            min-height: 100vh;
        }}

        h1 {{
            color: var(--accent-color);
            border-bottom: 3px solid var(--accent-color);
            padding-bottom: 0.5rem;
            margin-top: 0;
        }}

        h2 {{
            color: var(--accent-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.3rem;
            margin-top: 2rem;
        }}

        h3 {{
            color: var(--text-primary);
            margin-top: 1.5rem;
        }}

        h4 {{
            color: var(--text-secondary);
            margin-top: 1rem;
        }}

        a {{
            color: var(--accent-color);
            text-decoration: none;
        }}

        a:hover {{
            color: var(--accent-hover);
            text-decoration: underline;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}

        th, td {{
            border: 1px solid var(--border-color);
            padding: 0.5rem 0.75rem;
            text-align: left;
        }}

        th {{
            background: var(--bg-primary);
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background: var(--code-bg);
        }}

        code {{
            background: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.9em;
        }}

        pre {{
            background: var(--code-bg);
            padding: 1rem;
            border-radius: 5px;
            overflow-x: auto;
            border: 1px solid var(--border-color);
        }}

        pre code {{
            padding: 0;
            background: none;
        }}

        blockquote {{
            border-left: 4px solid var(--accent-color);
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            background: var(--code-bg);
            color: var(--text-secondary);
        }}

        ul, ol {{
            padding-left: 1.5rem;
        }}

        li {{
            margin: 0.25rem 0;
        }}

        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 2rem 0;
        }}

        .warning {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}

            table {{
                font-size: 0.9rem;
            }}

            th, td {{
                padding: 0.4rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
{html_content}
    </div>
</body>
</html>
'''


def main():
    # Get paths
    script_dir = Path(__file__).parent
    app_dir = script_dir.parent

    md_path = app_dir / 'docs' / 'USER_GUIDE.md'
    html_path = app_dir / 'static' / 'user_guide.html'

    # Read markdown
    print(f"Reading {md_path}...")
    md_content = md_path.read_text()

    # Convert to HTML
    print("Converting to HTML...")
    html_content = generate_html(md_content)

    # Write HTML
    print(f"Writing {html_path}...")
    html_path.write_text(html_content)

    print("Done!")


if __name__ == '__main__':
    main()
