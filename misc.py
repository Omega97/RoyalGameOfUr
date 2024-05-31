
def bar(p, color=92, length=20):
    p = min(1, max(0, p))
    chars = '▏▎▍▌▋▊▉██'
    n = int(p * length)
    residue = p * length - n
    out = f"{'█'*n}"
    if residue:
        i = int(residue * 8)
        out += f'{chars[i]}'
    out = out.ljust(length, ' ')
    out = f"\033[{color}m{out}\033[0m"
    return f"|{out}| "  # + "{p:6.2%}"
