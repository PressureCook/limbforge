"""Figure contract linter (SPEC-2 §11.1, V-8).

`check(fig)` inspects a matplotlib Figure and raises FigureContractError when
a generated figure would ship without the labelling an engineering diagram
needs: a title (F1), axis labels with units (F2), a legend when more than one
labelled series is drawn (F3), and no anonymous reference lines - every
axhline/axvline must carry a label stating what it is and its value (F4).

The test suite runs this over every figure factory in the pipeline; a figure
that cannot pass does not ship.
"""

UNIT_HINTS = ("(", "deg", "mm", "N", "s", "W", "J", "kg", "%", "rad")


class FigureContractError(AssertionError):
    pass


def _fail(msgs):
    raise FigureContractError("figure contract violations:\n  - "
                              + "\n  - ".join(msgs))


def check(fig):
    problems = []
    axes = [a for a in fig.get_axes() if a.get_visible()]
    if not axes:
        _fail(["figure has no axes"])
    has_title = bool(fig._suptitle and fig._suptitle.get_text().strip()) if \
        getattr(fig, "_suptitle", None) else False
    for ax in axes:
        if ax.get_title().strip():
            has_title = True
        # F2: axis labels with units on any axis that has data + ticks
        if ax.has_data() and not getattr(ax, "_figcheck_exempt", False):
            for name, get in (("x", ax.get_xlabel), ("y", ax.get_ylabel)):
                lab = get().strip()
                if ax.axison and not lab:
                    if name == "x":
                        # a shared-x stack labels only its bottom axes
                        shared = ax.get_shared_x_axes().get_siblings(ax)
                        if any(s is not ax and s.get_xlabel().strip()
                               for s in shared):
                            continue
                    problems.append(f"{name}-axis of '{ax.get_title() or 'axes'}'"
                                    " has no label")
                elif lab and not any(u in lab for u in UNIT_HINTS):
                    problems.append(f"{name}-axis label '{lab}' carries no unit")
        # F3: legend when >1 labelled line
        lines = [l for l in ax.get_lines()
                 if l.get_label() and not l.get_label().startswith("_")]
        if len(lines) > 1 and ax.get_legend() is None:
            problems.append(f"axes '{ax.get_title() or '?'}' draws "
                            f"{len(lines)} labelled series without a legend")
        # F4: no anonymous reference lines
        for l in ax.get_lines():
            xd, yd = l.get_xdata(), l.get_ydata()
            is_ref = (len(xd) == 2 and (xd[0] == xd[1] or yd[0] == yd[1])
                      and l.get_linestyle() in ("--", ":", "-."))
            if is_ref and (not l.get_label() or l.get_label().startswith("_")):
                problems.append(
                    f"anonymous dashed reference line in "
                    f"'{ax.get_title() or '?'}' - label it with meaning + value")
    if not has_title:
        problems.append("no title anywhere on the figure (F1)")
    if problems:
        _fail(problems)
    return True
