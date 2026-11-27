"""Plot measured checkpoint changes with aligned-element denominators."""


def plot_modules(path, modules, identities):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    names = [key for key in modules if not key.startswith("transformer.") and key != "policy.total"]
    if "transformer.total" in modules:
        names.append("transformer.total")
    names.sort()
    fig = Figure(figsize=(10, max(3.5, 0.42 * len(names) + 2)), layout="constrained")
    FigureCanvasAgg(fig)
    fraction, relative = fig.subplots(1, 2, sharey=True)
    for axis, key, title, scale in (
        (fraction, "changed_fraction", "Changed elements (% of aligned parameters)", 100),
        (relative, "relative_l2", "L2 difference / SFT L2 (unitless)", 1),
    ):
        values = [modules[name][key] for name in names]
        positions = list(range(len(names)))
        axis.barh(positions, [scale * value if value is not None else 0 for value in values],
                  color="#245d79")
        for position, value in zip(positions, values):
            if value is None:
                axis.annotate("undefined", (0, position))
        axis.set_xlabel(title)
        axis.set_yticks(positions, [
            f"{name} ({modules[name]['parameter_count']:,})" for name in names
        ])
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "RLinf OpenVLA-OFT LIBERO-130 checkpoint difference\n"
        f"SFT {identities['sft']['revision'][:10]} · GRPO {identities['grpo']['revision'][:10]}"
    )
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=160)
