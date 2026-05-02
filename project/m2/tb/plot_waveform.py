"""Parse waveform.vcd and render an annotated waveform PNG."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VCD_FILE = "waveform.vcd"
OUT_FILE = "../sim/waveform.png"
T_END    = 12500.0   # ns — covers write + read transaction

SIGNALS = ["clk", "cs_n", "sck", "mosi", "miso", "spike_in", "start", "weight_wr_en"]
BUS_SIGNALS = {"spike_in"}


def parse_vcd(path):
    with open(path) as f:
        text = f.read()

    # timescale
    m = re.search(r'\$timescale\s+(.*?)\s*\$end', text, re.DOTALL)
    scale = 1.0
    if m:
        ts = m.group(1).strip()
        if "1ps"   in ts: scale = 0.001
        elif "10ps" in ts: scale = 0.01
        elif "100ps" in ts: scale = 0.1
        elif "1ns"  in ts: scale = 1.0

    # var id -> name
    var_map = {}
    for m in re.finditer(r'\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+).*?\$end', text):
        width, vid, name = int(m.group(1)), m.group(2), m.group(3)
        var_map[vid] = (name, width)

    signals = {}
    time = 0.0
    tokens = text.split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith('#'):
            try:
                time = int(tok[1:]) * scale
            except ValueError:
                pass
            i += 1
        elif tok.lower().startswith('b'):
            vec = tok[1:]
            i += 1
            if i < len(tokens):
                vid = tokens[i]
                if vid in var_map:
                    name, _ = var_map[vid]
                    try:
                        val = str(int(vec, 2))
                    except ValueError:
                        val = '0'
                    signals.setdefault(name, []).append((time, val))
            i += 1
        elif len(tok) >= 2 and tok[0] in '01xzXZ' and not tok.startswith('$'):
            val, vid = tok[0], tok[1:]
            if vid in var_map:
                name, _ = var_map[vid]
                signals.setdefault(name, []).append((time, val))
            i += 1
        else:
            i += 1

    # sort and deduplicate (keep last value at each time)
    for name in signals:
        seen = {}
        for t, v in signals[name]:
            seen[t] = v
        signals[name] = sorted(seen.items())

    return signals


def bit_steps(data, t_end):
    xs, ys = [], []
    for i, (t, v) in enumerate(data):
        val = 1 if v == '1' else 0
        if not xs or xs[-1] != t:
            xs.append(t); ys.append(val)
        else:
            ys[-1] = val
    if xs:
        xs.append(t_end); ys.append(ys[-1])
    return xs, ys


def draw_bit(ax, data, t_end, label, color='steelblue'):
    data = [(t, v) for t, v in data if t <= t_end]
    if not data:
        return
    xs, ys = bit_steps(data, t_end)
    ax.fill_between(xs, ys, step='post', alpha=0.25, color=color)
    ax.step(xs, ys, where='post', color=color, lw=1.3)
    ax.set_ylim(-0.3, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'], fontsize=7)
    ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8.5, labelpad=46)


def draw_bus(ax, data, t_end, label, color='darkorange'):
    data = [(t, v) for t, v in data if t <= t_end]
    ax.set_ylim(-0.3, 1.5)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8.5, labelpad=46)
    for i, (t, v) in enumerate(data):
        nt = data[i+1][0] if i + 1 < len(data) else t_end
        try:
            iv = int(v)
        except ValueError:
            iv = 0
        ax.fill_between([t, nt], [0.15, 0.15], [0.85, 0.85], alpha=0.25, color=color)
        for x in [t, nt]:
            ax.plot([x, x], [0.15, 0.85], color=color, lw=1.2)
        ax.plot([t, nt], [0.85, 0.85], color=color, lw=1.2)
        ax.plot([t, nt], [0.15, 0.15], color=color, lw=1.2)
        if nt - t > 300:
            ax.text((t + nt) / 2, 0.5, f'0x{iv:02X}',
                    ha='center', va='center', fontsize=7.5, color='saddlebrown')


signals = parse_vcd(VCD_FILE)
rows = [s for s in SIGNALS if s in signals]

fig, axes = plt.subplots(len(rows), 1, figsize=(13, len(rows) * 0.85 + 1.2),
                         sharex=True)
if len(rows) == 1:
    axes = [axes]

fig.suptitle("spi_interface — SPI WRITE (spike_in←0xC5) then READ (status reg)",
             fontsize=10.5, y=0.99)

for ax, name in zip(axes, rows):
    if name in BUS_SIGNALS:
        draw_bus(ax, signals[name], T_END, name)
    else:
        draw_bit(ax, signals[name], T_END, name)
    ax.set_xlim(0, T_END)
    ax.tick_params(axis='x', labelsize=7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

axes[-1].set_xlabel("Time (ns)", fontsize=8.5)

# Dashed dividers between transactions
for ax in axes:
    ax.axvline(x=60,   color='gray', lw=0.7, linestyle='--', alpha=0.6)
    ax.axvline(x=6060, color='gray', lw=0.7, linestyle='--', alpha=0.6)

# Annotations on the clk row
axes[0].annotate('WRITE 0xA000 ← 0xC5',
                 xy=(200, 1.2), fontsize=8, color='navy',
                 fontstyle='italic')
axes[0].annotate('READ status 0xA01A',
                 xy=(6200, 1.2), fontsize=8, color='darkred',
                 fontstyle='italic')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"Saved {OUT_FILE}")
