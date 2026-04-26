import pathlib

pathlib.Path(
    '/Users/dansmart/gitsky/but_with_subs/src/but_with_subs/llm.py'
).write_text(
    content := open('/dev/stdin').read() if False else 'x'
)
print('OK')
