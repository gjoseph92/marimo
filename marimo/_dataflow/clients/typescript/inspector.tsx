// inspector.tsx — opt-in "what produced this?" overlay for dataflow apps.
//
// Vendored alongside ``dataflow.tsx`` and zero-cost when not used. Wrap a
// component subtree in ``<Inspectable>`` and the inspector will show, on
// hover, a compact mini-DAG of every variable that contributes data to
// the hovered region (unified ancestor closure when the region reads
// more than one variable), plus a live, kind-dispatched preview of any
// node you point at (or click to pin).
//
// Positioning is anchored to the inspectable region's bounding box, not
// the cursor — this keeps mousemove cheap (no per-pixel React updates)
// and lets the popover stay put while you reach for it.
//
// Interaction model:
//   - Hover an inspectable region → popover shows for that region.
//   - Click an inspectable region → pin it (popover stays even after
//     the cursor leaves).
//   - Click anywhere outside the popover → just *unpin*. After that,
//     hover state takes over: if the cursor is over another
//     inspectable, the popover shows it; if the cursor isn't over
//     anything, the popover fades.
//   - The "unpin" button in the popover header is an explicit
//     dismissal that clears everything.
//
// React doesn't expose which fiber called a hook from inside the hook,
// so truly automatic tracking would need unstable internals. Instead,
// ``<Inspectable>`` opens a ``InspectorTrackerContext`` that the
// existing ``useDataflowValue``/``useDataflowVariable``/``useDataflowInput``
// hooks register with on each render — same UX as zero-config, but
// contained in stable React APIs.

import {
  type CSSProperties,
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import {
  InspectorTrackerContext,
  type Kind,
  getAncestors,
  getDirectDeps,
  useDataflowGraph,
  useDataflowInput,
  useDataflowSchema,
  useDataflowValue,
  useDataflowValuesSnapshot,
  type VariableGraph,
  type VarUpdate,
} from "./dataflow";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface InspectorProviderProps {
  children: ReactNode;
  /**
   * Toggle the overlay on/off without unmounting it. Letting the host app
   * own this state keeps the inspector a pure render-time concern.
   */
  enabled?: boolean;
}

export interface InspectableProps {
  children: ReactNode;
  /**
   * Optional explicit variable list. When omitted, the inspector
   * auto-collects via ``InspectorTrackerContext`` whatever the descendant
   * dataflow hooks read on the most recent render. Provide this when
   * descendants subscribe conditionally and you want a stable surface.
   */
  vars?: readonly string[];
  /** Human-readable label shown in the overlay header. */
  label?: string;
  /**
   * Forwarded to the wrapper ``<div>``. The inspector adds its own
   * ``data-inspect-id`` attribute on top.
   */
  style?: CSSProperties;
  className?: string;
}

interface InspectableRegion {
  id: string;
  label: string | null;
  vars: readonly string[];
  element: HTMLElement;
}

interface InspectorState {
  enabled: boolean;
  registerRegion: (region: InspectableRegion) => void;
  unregisterRegion: (id: string) => void;
}

const InspectorContext = createContext<InspectorState | null>(null);

// ---------------------------------------------------------------------------
// <InspectorProvider>
// ---------------------------------------------------------------------------

export function InspectorProvider({
  children,
  enabled = true,
}: InspectorProviderProps) {
  // ``regionsRef`` is the source of truth (no re-render on change); the
  // version bump only fires when a new region mounts so the overlay's
  // first render reflects it.
  const regionsRef = useRef<Map<string, InspectableRegion>>(new Map());
  const [, setRegionVersion] = useState(0);

  const registerRegion = useCallback((region: InspectableRegion) => {
    regionsRef.current.set(region.id, region);
    setRegionVersion((v) => v + 1);
  }, []);

  const unregisterRegion = useCallback((id: string) => {
    regionsRef.current.delete(id);
  }, []);

  const state = useMemo<InspectorState>(
    () => ({ enabled, registerRegion, unregisterRegion }),
    [enabled, registerRegion, unregisterRegion],
  );

  return (
    <InspectorContext.Provider value={state}>
      {children}
      {enabled && <InspectorOverlay regionsRef={regionsRef} />}
    </InspectorContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// <Inspectable>
// ---------------------------------------------------------------------------

export function Inspectable({
  children,
  vars: explicitVars,
  label,
  style,
  className,
}: InspectableProps) {
  const inspector = useContext(InspectorContext);
  const id = useId();
  const ref = useRef<HTMLDivElement | null>(null);

  // Reset every render so descendants that conditionally drop a hook stop
  // contributing immediately. The tracker stashes into a ref so the
  // post-render effect can read the final snapshot below.
  const collectedRef = useRef<Set<string>>(new Set());
  collectedRef.current = new Set();
  const tracker = useMemo<{ track: (n: string) => void }>(
    () => ({ track: (n) => collectedRef.current.add(n) }),
    [],
  );

  // Re-register on every render — descendant hook order may have shifted
  // since the last commit, so the var set is potentially stale.
  useEffect(() => {
    if (!inspector || !ref.current) return;
    const vars = explicitVars
      ? Array.from(new Set(explicitVars))
      : Array.from(collectedRef.current);
    inspector.registerRegion({
      id,
      label: label ?? null,
      vars,
      element: ref.current,
    });
    return () => inspector.unregisterRegion(id);
    // intentionally no deps: re-run on every render so the var set is fresh
  });

  return (
    <InspectorTrackerContext.Provider value={tracker}>
      <div
        ref={ref}
        data-inspect-id={id}
        className={className}
        style={style}
      >
        {children}
      </div>
    </InspectorTrackerContext.Provider>
  );
}

/** @deprecated Renamed to ``Inspectable`` — kept for backward compatibility. */
export const Inspect = Inspectable;

// ---------------------------------------------------------------------------
// <InspectorOverlay>
// ---------------------------------------------------------------------------

interface OverlayState {
  hoverRegion: InspectableRegion | null;
  pinnedRegion: InspectableRegion | null;
}

const INITIAL_OVERLAY_STATE: OverlayState = {
  hoverRegion: null,
  pinnedRegion: null,
};

function InspectorOverlay({
  regionsRef,
}: {
  regionsRef: React.MutableRefObject<Map<string, InspectableRegion>>;
}) {
  const inspector = useContext(InspectorContext);
  const [state, setState] = useState<OverlayState>(INITIAL_OVERLAY_STATE);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const clearTimerRef = useRef<number | null>(null);

  // ``hoverRegion`` only updates when crossing region boundaries — never
  // per pixel. With region-anchored positioning this means the popover
  // is genuinely cheap to keep open.
  useEffect(() => {
    if (!inspector?.enabled) return;

    const cancelClear = () => {
      if (clearTimerRef.current !== null) {
        window.clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
    };

    const onMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      // Inside the popover: cancel any pending clear so it stays open
      // while the user drives the cursor through DAG nodes / preview.
      if (popoverRef.current?.contains(target)) {
        cancelClear();
        return;
      }
      const inspectEl = target.closest<HTMLElement>("[data-inspect-id]");
      const id = inspectEl?.getAttribute("data-inspect-id") ?? null;
      const region = id ? regionsRef.current.get(id) ?? null : null;
      if (region) {
        cancelClear();
        // Always update hover, even when pinned, so unpinning later
        // immediately reflects whatever the cursor is currently over.
        // The popover display is governed by ``pinned ?? hover``, so
        // pinned still wins for what's shown.
        setState((prev) =>
          prev.hoverRegion?.id === region.id
            ? prev
            : { ...prev, hoverRegion: region },
        );
      }
    };

    const onMouseOut = (e: MouseEvent) => {
      // ``relatedTarget`` is where the cursor went next. If it stayed
      // inside an inspectable region or the popover, the next
      // ``mouseover`` will fire and we don't need to clear; otherwise
      // schedule a tiny grace window so accidental gaps between region
      // and popover don't flicker the UI.
      const next = e.relatedTarget as HTMLElement | null;
      if (next && popoverRef.current?.contains(next)) return;
      if (next?.closest("[data-inspect-id]")) return;
      if (clearTimerRef.current === null) {
        clearTimerRef.current = window.setTimeout(() => {
          clearTimerRef.current = null;
          // Clear hover regardless of pin so the moment the user
          // unpins, the popover correctly hides if the cursor isn't
          // over anything inspectable.
          setState((prev) => ({ ...prev, hoverRegion: null }));
        }, 100);
      }
    };

    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      // Clicks inside the popover are owned by its internal handlers.
      if (popoverRef.current?.contains(target)) return;

      setState((prev) => {
        // While pinned, *any* outside click just unpins. Hover takes
        // over: if the cursor happens to be over another inspectable,
        // the popover transitions to that one; if it isn't, the
        // popover fades. This matches the user's mental model of
        // "click outside = unpin, not dismiss".
        if (prev.pinnedRegion) {
          return { ...prev, pinnedRegion: null };
        }
        // Not pinned: clicking an inspectable pins it. Anything else
        // is a no-op (hover state already governs visibility).
        const inspectEl = target.closest<HTMLElement>("[data-inspect-id]");
        if (!inspectEl) return prev;
        const id = inspectEl.getAttribute("data-inspect-id");
        const region = id ? regionsRef.current.get(id) ?? null : null;
        if (!region) return prev;
        return { hoverRegion: region, pinnedRegion: region };
      });
    };

    document.addEventListener("mouseover", onMouseOver);
    document.addEventListener("mouseout", onMouseOut);
    document.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("mouseover", onMouseOver);
      document.removeEventListener("mouseout", onMouseOut);
      document.removeEventListener("click", onClick);
      cancelClear();
    };
  }, [inspector?.enabled, regionsRef]);

  const region = state.pinnedRegion ?? state.hoverRegion;
  if (!region) return null;

  return createPortal(
    <OverlayPopover
      popoverRef={popoverRef}
      region={region}
      pinned={state.pinnedRegion?.id === region.id}
      onUnpin={() => setState(INITIAL_OVERLAY_STATE)}
    />,
    document.body,
  );
}

// ---------------------------------------------------------------------------
// Popover
// ---------------------------------------------------------------------------

interface OverlayPopoverProps {
  region: InspectableRegion;
  pinned: boolean;
  onUnpin: () => void;
  popoverRef: React.MutableRefObject<HTMLDivElement | null>;
}

const POPOVER_WIDTH = 720;
const POPOVER_HEIGHT = 420;
const ROW_HEIGHT = 22;

function OverlayPopover({
  region,
  pinned,
  onUnpin,
  popoverRef,
}: OverlayPopoverProps) {
  const graph = useDataflowGraph();
  const schema = useDataflowSchema();
  const values = useDataflowValuesSnapshot();
  const [hoveredVar, setHoveredVar] = useState<string | null>(null);
  const [pinnedVar, setPinnedVar] = useState<string | null>(null);

  // Reset the var-level pin whenever the region changes — pinning is
  // scoped to a single popover open.
  useEffect(() => {
    setPinnedVar(null);
    setHoveredVar(null);
  }, [region.id]);

  // Unified ancestor closure: every variable that feeds the region.
  const subgraphNodes = useMemo(() => {
    const nodes = new Set<string>(region.vars);
    for (const v of region.vars) {
      for (const a of getAncestors(graph, v)) nodes.add(a);
    }
    return nodes;
  }, [graph, region.vars]);

  // Topologically ordered list — sources up top, sinks at the bottom.
  const ordered = useMemo(
    () => topologicalOrder(subgraphNodes, graph),
    [subgraphNodes, graph],
  );

  const inputNames = useMemo(
    () => new Set(schema?.inputs.map((i) => i.name) ?? []),
    [schema],
  );

  const popoverStyle = useMemo<CSSProperties>(
    () => positionForRegion(region.element),
    [region.element, region.id],
  );

  const visibleVar = pinnedVar ?? hoveredVar;

  return (
    <div ref={popoverRef} style={popoverStyle} data-dataflow-inspector-popover="">
      <header style={styles.header}>
        <strong>{region.label ?? "inspect"}</strong>
        <span style={styles.headerSub}>
          reads <code>{region.vars.join(", ") || "—"}</code> ·{" "}
          {subgraphNodes.size} variables
        </span>
        {pinned && (
          <button type="button" onClick={onUnpin} style={styles.unpinBtn}>
            unpin
          </button>
        )}
      </header>

      <div style={styles.body}>
        <section style={styles.dagPanel}>
          <MiniDag
            ordered={ordered}
            graph={graph}
            inputNames={inputNames}
            selectedVar={visibleVar}
            onHoverVar={setHoveredVar}
            onPinVar={(name) =>
              setPinnedVar((cur) => (cur === name ? null : name))
            }
          />
        </section>
        <section style={styles.previewPanel}>
          {visibleVar ? (
            <PreviewPanelWithSubscription
              name={visibleVar}
              update={values[visibleVar]}
              isInput={inputNames.has(visibleVar)}
              kind={
                inputNames.has(visibleVar)
                  ? "any"
                  : kindOf(visibleVar, schema)
              }
            />
          ) : (
            <p style={styles.placeholder}>
              Hover any node in the subgraph to preview its value. Click to
              pin.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}

function kindOf(
  name: string,
  schema: ReturnType<typeof useDataflowSchema>,
): Kind {
  return schema?.outputs.find((o) => o.name === name)?.kind ?? "any";
}

// ---------------------------------------------------------------------------
// Positioning
// ---------------------------------------------------------------------------

/**
 * Anchor the popover next to ``region`` — to the right when there's
 * room, otherwise left, then below, then above. Always clamped to the
 * viewport. We compute once per region change (not on mousemove), which
 * is what keeps the inspector cheap during hover.
 */
function positionForRegion(element: HTMLElement): CSSProperties {
  const rect = element.getBoundingClientRect();
  const margin = 12;
  const gap = 8;
  const W = POPOVER_WIDTH;
  const H = POPOVER_HEIGHT;

  const candidates: { left: number; top: number }[] = [
    { left: rect.right + gap, top: rect.top },
    { left: rect.left - W - gap, top: rect.top },
    { left: rect.left, top: rect.bottom + gap },
    { left: rect.left, top: rect.top - H - gap },
  ];

  for (const c of candidates) {
    if (
      c.left >= margin &&
      c.left + W + margin <= window.innerWidth &&
      c.top >= margin &&
      c.top + H + margin <= window.innerHeight
    ) {
      return { ...styles.popover, left: c.left, top: c.top };
    }
  }
  // Nothing fits cleanly: clamp to viewport.
  const left = clamp(rect.right + gap, margin, window.innerWidth - W - margin);
  const top = clamp(rect.top, margin, window.innerHeight - H - margin);
  return { ...styles.popover, left, top };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

// ---------------------------------------------------------------------------
// Mini-DAG — marimo-minimap-style compact SVG visualization.
//
// The layout is a single column of fixed-height rows in topological
// order. Each row has a small circle on the left and the variable name
// to its right. Connection lines render as a single SVG layer behind
// the rows: when a variable is selected we draw an L-shaped path from
// its dot to each ancestor (going up-left) and each descendant (going
// down-left), terminating at the target row. Ancestors of the selection
// shift left by ``WHISKER`` so the path can dock cleanly against them;
// descendants shift right.
// ---------------------------------------------------------------------------

const MARGIN_X = 32; // left padding inside the dag panel
const NAME_X = MARGIN_X + 22; // x at which the variable name starts
const WHISKER = 10;

interface MiniDagProps {
  ordered: readonly string[];
  graph: VariableGraph;
  inputNames: Set<string>;
  selectedVar: string | null;
  onHoverVar: (name: string | null) => void;
  onPinVar: (name: string) => void;
}

function MiniDag({
  ordered,
  graph,
  inputNames,
  selectedVar,
  onHoverVar,
  onPinVar,
}: MiniDagProps) {
  const positions = useMemo(() => {
    const m = new Map<string, number>();
    ordered.forEach((name, i) => m.set(name, i));
    return m;
  }, [ordered]);

  const subgraphSet = useMemo(() => new Set(ordered), [ordered]);
  const totalHeight = ordered.length * ROW_HEIGHT + 8;

  // Pre-compute relations to the selected node, so each row knows
  // whether to jitter and whether to draw bold.
  const relations = useMemo(() => {
    if (!selectedVar)
      return {
        ancestors: new Set<string>(),
        descendants: new Set<string>(),
        directParents: new Set<string>(),
        directChildren: new Set<string>(),
      };
    const ancestors = getAncestors(graph, selectedVar);
    const directParents = new Set(getDirectDeps(graph, selectedVar));
    const directChildren = new Set<string>();
    const descendants = new Set<string>();
    // Children: every node in subgraph that lists selectedVar as a dep.
    for (const name of ordered) {
      const deps = getDirectDeps(graph, name);
      if (deps.includes(selectedVar)) {
        directChildren.add(name);
        descendants.add(name);
      }
    }
    // Transitive descendants (BFS).
    const stack = [...directChildren];
    while (stack.length) {
      const n = stack.pop()!;
      for (const m of ordered) {
        if (descendants.has(m)) continue;
        if (getDirectDeps(graph, m).includes(n)) {
          descendants.add(m);
          stack.push(m);
        }
      }
    }
    return { ancestors, descendants, directParents, directChildren };
  }, [graph, selectedVar, ordered]);

  return (
    <div style={{ position: "relative", height: totalHeight }}>
      <svg
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          overflow: "visible",
        }}
        width="100%"
        height={totalHeight}
      >
        {/* Per-row glyphs: dot + small whiskers for "has parent" / "has child" */}
        {ordered.map((name, i) => {
          const y = i * ROW_HEIGHT + ROW_HEIGHT / 2 + 4;
          const hasParents = getDirectDeps(graph, name).some((d) =>
            subgraphSet.has(d),
          );
          const hasChildren = ordered.some((m) =>
            getDirectDeps(graph, m).includes(name),
          );
          const isSelected = name === selectedVar;
          const inUpstream = relations.ancestors.has(name);
          const inDownstream = relations.descendants.has(name);
          const dx =
            !selectedVar || isSelected
              ? 0
              : inUpstream && !inDownstream
                ? -WHISKER
                : inDownstream && !inUpstream
                  ? WHISKER
                  : 0;
          const cx = MARGIN_X + dx;
          const isInput = inputNames.has(name);
          const fade =
            !!selectedVar &&
            !isSelected &&
            !inUpstream &&
            !inDownstream;
          const color = fade
            ? "#ced4da"
            : isSelected
              ? "#1864ab"
              : "#4361ee";
          return (
            <g key={`glyph-${name}`} opacity={fade ? 0.45 : 1}>
              {hasParents && (
                <path
                  d={`M ${cx - 3} ${y} h -${WHISKER}`}
                  stroke={color}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                  fill="none"
                />
              )}
              {hasChildren && (
                <path
                  d={`M ${cx + 3} ${y} h ${WHISKER}`}
                  stroke={color}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                  fill="none"
                />
              )}
              <circle
                cx={cx}
                cy={y}
                r={isSelected ? 4 : isInput ? 2.5 : 3.5}
                fill={color}
              />
            </g>
          );
        })}

        {/* Connection paths for the selected node */}
        {selectedVar &&
          positions.has(selectedVar) &&
          (() => {
            const sy =
              (positions.get(selectedVar) ?? 0) * ROW_HEIGHT + ROW_HEIGHT / 2 + 4;
            const sx = MARGIN_X;
            const paths: React.ReactNode[] = [];
            for (const parent of relations.directParents) {
              if (!positions.has(parent)) continue;
              const py =
                (positions.get(parent) ?? 0) * ROW_HEIGHT + ROW_HEIGHT / 2 + 4;
              // Parent jittered left by WHISKER → terminate at its right whisker
              const targetX = MARGIN_X - WHISKER + 3;
              paths.push(
                <path
                  key={`up-${parent}`}
                  d={`M ${sx - 3} ${sy} H ${sx - WHISKER - 4} V ${py} H ${targetX}`}
                  stroke="#1864ab"
                  strokeWidth={2.5}
                  fill="none"
                />,
              );
            }
            for (const child of relations.directChildren) {
              if (!positions.has(child)) continue;
              const cy =
                (positions.get(child) ?? 0) * ROW_HEIGHT + ROW_HEIGHT / 2 + 4;
              // Child jittered right by WHISKER → terminate at its left whisker
              const targetX = MARGIN_X + WHISKER - 3;
              paths.push(
                <path
                  key={`down-${child}`}
                  d={`M ${sx + 3} ${sy} H ${sx + WHISKER + 4} V ${cy} H ${targetX}`}
                  stroke="#1864ab"
                  strokeWidth={2.5}
                  fill="none"
                />,
              );
            }
            return paths;
          })()}
      </svg>

      {/* Variable rows (overlay) */}
      {ordered.map((name, i) => {
        const y = i * ROW_HEIGHT;
        const isSelected = name === selectedVar;
        const isInput = inputNames.has(name);
        const fade =
          !!selectedVar &&
          !isSelected &&
          !relations.ancestors.has(name) &&
          !relations.descendants.has(name);
        return (
          <button
            type="button"
            key={name}
            aria-label={`inspect ${name}`}
            style={{
              ...styles.dagNameButton,
              top: y,
              left: NAME_X,
              opacity: fade ? 0.5 : 1,
              fontWeight: isSelected ? 700 : 500,
              color: isSelected ? "#1864ab" : "#212529",
            }}
            onMouseEnter={() => onHoverVar(name)}
            onMouseLeave={() => onHoverVar(null)}
            onClick={(e) => {
              e.stopPropagation();
              onPinVar(name);
            }}
          >
            {name}
            {isInput && <span style={styles.tagInput}>input</span>}
          </button>
        );
      })}
    </div>
  );
}

/**
 * Stable topological order over a node subset. Sources first, then their
 * direct dependents, etc. Cycle-safe.
 */
function topologicalOrder(
  nodes: Set<string>,
  graph: VariableGraph,
): readonly string[] {
  const depth = new Map<string, number>();
  const visiting = new Set<string>();
  function d(name: string): number {
    const cached = depth.get(name);
    if (cached !== undefined) return cached;
    if (visiting.has(name)) return 0;
    visiting.add(name);
    const deps = getDirectDeps(graph, name).filter((p) => nodes.has(p));
    const v = deps.length === 0 ? 0 : 1 + Math.max(...deps.map(d));
    visiting.delete(name);
    depth.set(name, v);
    return v;
  }
  for (const n of nodes) d(n);
  return [...nodes].sort((a, b) => {
    const da = depth.get(a) ?? 0;
    const db = depth.get(b) ?? 0;
    if (da !== db) return da - db;
    return a.localeCompare(b);
  });
}

// ---------------------------------------------------------------------------
// Preview panel
// ---------------------------------------------------------------------------

/**
 * Wrapper that subscribes via ``useDataflowValue`` (or reads ``useDataflowInput``
 * for input variables) so the inspector implicitly streams whatever the
 * user pins. The subscription is dropped as soon as the user picks a
 * different node or unpins.
 */
function PreviewPanelWithSubscription({
  name,
  update,
  isInput,
  kind,
}: {
  name: string;
  update: VarUpdate | undefined;
  isInput: boolean;
  kind: Kind;
}) {
  // Inputs flow through a separate channel — read directly from the
  // input store so the preview shows the current bound value.
  const inputValue = useDataflowInput(isInput ? name : "");
  const liveValue = useDataflowValue(isInput ? "" : name);

  return (
    <div style={styles.previewBody}>
      <h4 style={styles.previewTitle}>
        {name}
        <span style={styles.previewKind}>{isInput ? "input" : kind}</span>
      </h4>
      {isInput ? (
        inputValue === undefined ? (
          <em style={styles.placeholder}>(no value bound)</em>
        ) : (
          <PreviewByKind value={inputValue} kind="any" />
        )
      ) : update ? (
        <PreviewByKind value={update.value} kind={update.kind} />
      ) : liveValue !== undefined ? (
        <PreviewByKind value={liveValue} kind="any" />
      ) : (
        <em style={styles.placeholder}>
          (no value yet — streaming…)
        </em>
      )}
    </div>
  );
}

function PreviewByKind({ value, kind }: { value: unknown; kind: Kind }) {
  if (value == null) return <em style={styles.placeholder}>null</em>;
  if (kind === "table" || isArrayOfRecords(value)) {
    return <TablePreview rows={value as Record<string, unknown>[]} />;
  }
  if (kind === "image" && typeof value === "string") {
    return <img src={value} style={styles.imagePreview} alt="preview" />;
  }
  if (kind === "html" && typeof value === "string") {
    return (
      <div
        style={styles.htmlPreview}
        dangerouslySetInnerHTML={{ __html: value }}
      />
    );
  }
  if (kind === "string") {
    const s = String(value);
    return (
      <pre style={s.length < 200 ? styles.scalarPreview : styles.longStringPreview}>
        {s}
      </pre>
    );
  }
  if (kind === "number" || kind === "integer" || kind === "boolean") {
    return <pre style={styles.scalarPreview}>{String(value)}</pre>;
  }
  return <JsonTree value={value} />;
}

function isArrayOfRecords(v: unknown): v is Record<string, unknown>[] {
  return (
    Array.isArray(v) &&
    v.length > 0 &&
    typeof v[0] === "object" &&
    v[0] !== null &&
    !Array.isArray(v[0])
  );
}

function TablePreview({ rows }: { rows: Record<string, unknown>[] }) {
  const max = 50;
  const display = rows.slice(0, max);
  const cols = useMemo(() => {
    const seen = new Set<string>();
    for (const r of display) for (const k of Object.keys(r)) seen.add(k);
    return Array.from(seen);
  }, [display]);
  return (
    <div style={styles.tableWrap}>
      <table style={styles.table}>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c} style={styles.th}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {display.map((row, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c} style={styles.td}>
                  {String(row[c] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > max && (
        <p style={styles.placeholder}>
          showing first {max} of {rows.length} rows
        </p>
      )}
    </div>
  );
}

function JsonTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value == null) return <span style={styles.jsonNull}>null</span>;
  if (typeof value === "boolean")
    return <span style={styles.jsonBool}>{String(value)}</span>;
  if (typeof value === "number")
    return <span style={styles.jsonNum}>{value}</span>;
  if (typeof value === "string")
    return <span style={styles.jsonStr}>"{value}"</span>;
  if (Array.isArray(value))
    return <ArrayNode items={value} depth={depth} />;
  if (typeof value === "object")
    return <ObjectNode obj={value as Record<string, unknown>} depth={depth} />;
  return <span style={styles.jsonNum}>{JSON.stringify(value)}</span>;
}

function ArrayNode({ items, depth }: { items: unknown[]; depth: number }) {
  const [open, setOpen] = useState(depth < 2);
  if (items.length === 0) return <span style={styles.jsonPunct}>[]</span>;
  return (
    <span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        style={styles.jsonToggle}
        aria-label={open ? "collapse" : "expand"}
      >
        {open ? "▾" : "▸"}
      </button>
      <span style={styles.jsonPunct}>[ {items.length} items ]</span>
      {open && (
        <div style={{ ...styles.jsonNest, marginLeft: 12 }}>
          {items.slice(0, 50).map((it, i) => (
            <div key={i}>
              <span style={styles.jsonKey}>{i}:</span>{" "}
              <JsonTree value={it} depth={depth + 1} />
            </div>
          ))}
          {items.length > 50 && (
            <em style={styles.placeholder}>… {items.length - 50} more</em>
          )}
        </div>
      )}
    </span>
  );
}

function ObjectNode({
  obj,
  depth,
}: {
  obj: Record<string, unknown>;
  depth: number;
}) {
  const keys = Object.keys(obj);
  const [open, setOpen] = useState(depth < 2);
  if (keys.length === 0) return <span style={styles.jsonPunct}>{"{}"}</span>;
  return (
    <span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        style={styles.jsonToggle}
        aria-label={open ? "collapse" : "expand"}
      >
        {open ? "▾" : "▸"}
      </button>
      <span style={styles.jsonPunct}>{`{ ${keys.length} fields }`}</span>
      {open && (
        <div style={{ ...styles.jsonNest, marginLeft: 12 }}>
          {keys.slice(0, 50).map((k) => (
            <div key={k}>
              <span style={styles.jsonKey}>"{k}":</span>{" "}
              <JsonTree value={obj[k]} depth={depth + 1} />
            </div>
          ))}
          {keys.length > 50 && (
            <em style={styles.placeholder}>
              … {keys.length - 50} more keys
            </em>
          )}
        </div>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Styles — inline so the file drops in with no CSS pipeline.
// ---------------------------------------------------------------------------

const styles: Record<string, CSSProperties> = {
  popover: {
    position: "fixed",
    background: "#fff",
    color: "#1a1a2e",
    border: "1px solid #d0d7de",
    borderRadius: 10,
    boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
    fontSize: 13,
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    zIndex: 10000,
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
    width: POPOVER_WIDTH,
    height: POPOVER_HEIGHT,
  },
  header: {
    padding: "8px 12px",
    borderBottom: "1px solid #e5e7eb",
    background: "#f8f9fa",
    display: "flex",
    alignItems: "baseline",
    gap: 8,
    flexShrink: 0,
  },
  headerSub: { color: "#6c757d", fontSize: 11, flex: 1 },
  unpinBtn: {
    border: "1px solid #dee2e6",
    background: "#fff",
    borderRadius: 4,
    padding: "2px 8px",
    fontSize: 11,
    cursor: "pointer",
  },
  body: { display: "flex", flex: 1, minHeight: 0 },
  dagPanel: {
    flex: "0 0 280px",
    overflow: "auto",
    borderRight: "1px solid #e5e7eb",
    padding: "8px 0",
  },
  previewPanel: {
    flex: 1,
    overflow: "auto",
    padding: "10px 12px",
  },
  dagNameButton: {
    position: "absolute",
    height: ROW_HEIGHT,
    paddingLeft: 4,
    paddingRight: 8,
    border: "none",
    background: "transparent",
    cursor: "pointer",
    fontFamily: "ui-monospace, SFMono-Regular, monospace",
    fontSize: 12,
    textAlign: "left",
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    borderRadius: 4,
    transition: "background 80ms ease",
  },
  tagInput: {
    fontSize: 9,
    padding: "1px 5px",
    borderRadius: 999,
    background: "#fff3bf",
    color: "#7a4a00",
    fontWeight: 600,
    textTransform: "uppercase",
  },
  previewBody: {},
  previewTitle: {
    margin: "0 0 8px",
    fontSize: 14,
    display: "flex",
    alignItems: "baseline",
    gap: 8,
  },
  previewKind: {
    fontSize: 10,
    padding: "1px 6px",
    borderRadius: 3,
    background: "#e9ecef",
    color: "#495057",
    textTransform: "uppercase",
    fontWeight: 600,
  },
  placeholder: { color: "#868e96", fontSize: 12 },
  tableWrap: { overflow: "auto", maxHeight: 340 },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 12 },
  th: {
    textAlign: "left",
    padding: "4px 6px",
    borderBottom: "2px solid #dee2e6",
    fontWeight: 600,
    color: "#495057",
    background: "#f8f9fa",
    position: "sticky",
    top: 0,
  },
  td: {
    padding: "3px 6px",
    borderBottom: "1px solid #f1f3f5",
    fontFamily: "ui-monospace, SFMono-Regular, monospace",
    fontSize: 11,
  },
  imagePreview: { maxWidth: "100%", maxHeight: 320, borderRadius: 4 },
  htmlPreview: { fontSize: 12 },
  scalarPreview: {
    background: "#f1f3f5",
    padding: "8px 10px",
    borderRadius: 4,
    fontSize: 16,
    margin: 0,
  },
  longStringPreview: {
    background: "#f1f3f5",
    padding: 8,
    borderRadius: 4,
    fontSize: 11,
    maxHeight: 320,
    overflow: "auto",
    margin: 0,
    whiteSpace: "pre-wrap",
  },
  jsonNull: { color: "#868e96", fontStyle: "italic" },
  jsonBool: { color: "#0f3460" },
  jsonNum: { color: "#0f3460" },
  jsonStr: { color: "#2b8a3e" },
  jsonKey: { color: "#495057", fontWeight: 600 },
  jsonPunct: { color: "#6c757d" },
  jsonNest: { borderLeft: "1px dashed #e9ecef", paddingLeft: 6 },
  jsonToggle: {
    border: "none",
    background: "transparent",
    cursor: "pointer",
    fontSize: 10,
    padding: "0 4px",
    color: "#495057",
  },
};
