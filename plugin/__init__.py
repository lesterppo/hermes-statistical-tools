"""hermes_statistical_tools — update-surviving statistical analysis tools.

Why this plugin exists
----------------------
`hermes update` resets everything inside the git tree: locally added
tools/*.py (only upstream files survive), toolsets.py and tools_config.py.
Installing these tools by copying them into ~/.hermes/hermes-agent/tools/
therefore loses them on the next update and leaves dangling toolset
references ("Unknown toolsets: medical").

This plugin lives in ~/.hermes/plugins/ — OUTSIDE the git tree — so updates
never touch it.

Two-step registration
---------------------
1. Importing each tool module runs its module-level `registry.register(...)`,
   so the tool exists in the registry with its canonical schema, handler and
   check_fn (single source of truth stays in each tool file).
2. Each tool is then re-registered through `ctx.register_tool(...)`, which
   marks it as plugin-owned — that is what makes `get_plugin_toolsets()`
   see the `medical` toolset and auto-enable it.

No core Hermes files are modified.

Overlap warning
---------------
The sibling `hermes_medical_tools` plugin also registers a `medical`
toolset (including its own `pspp`). Enabling both plugins registers the
same tool names twice — last registration wins. The installer flags this;
disable one set with `hermes plugins disable <name>` if results look off.
"""

from __future__ import annotations

# Tool modules shipped in plugin/tools/. Import runs their registry.register().
_TOOL_MODULES = [
    "pspp_tool",         # pspp         (pure-Python PSPP 2.0.0, 45 test types)
    "statsmodels_tool",  # statsmodels  (mixed models, GEE, RM-ANOVA, MICE)
    "sem_tool",          # sem          (SEM via semopy)
    "irt_tool",          # irt          (IRT via girth)
    "medical_ext_tool",  # medical_ext  (survival, meta-analysis, power)
]

for _m in _TOOL_MODULES:
    try:
        __import__(f"{__name__}.tools.{_m}")
    except Exception as _e:  # noqa: BLE001 — isolate; a missing optional dep
        print(f"[hermes_statistical_tools] WARNING: failed to import {_m}: {_e}")  # must not hide the rest

# Tool name -> toolset, mirroring each tool file's own registry.register().
_TOOLS = [
    ("pspp", "medical"),
    ("statsmodels", "medical"),
    ("sem", "medical"),
    ("irt", "medical"),
    ("medical_ext", "medical"),
]


def register(ctx) -> None:
    """Register every available tool into the medical toolset."""
    from tools.registry import registry

    for name, toolset in _TOOLS:
        entry = registry.get_entry(name)
        if entry is None:
            # Module failed to import or the backend is absent — skip this one
            # rather than crashing the whole plugin.
            continue
        ctx.register_tool(
            name=name,
            toolset=toolset,
            schema=entry.schema,
            handler=entry.handler,
            check_fn=entry.check_fn,
            is_async=entry.is_async,
            emoji=entry.emoji or "",
        )
