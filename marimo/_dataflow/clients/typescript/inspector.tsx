// inspector.tsx — opt-in "what produced this?" overlay for dataflow apps.
//
// Vendored alongside ``dataflow.tsx`` and zero-cost when not used. Wrap a
// component subtree in ``<Inspect>`` and the inspector will show, on hover:
//
//   1. A pruned mini-DAG of every variable that contributes data to the
//      hovered region (unified ancestor closure when the region reads more
//      than one variable).
//   2. A live preview of any node you point at (or click to pin).
//   3. A simple per-variable timing waterfall computed client-side from
//      ``VarUpdate.ts`` arrival times.
//
// React doesn't expose which fiber called a hook from inside the hook, so
// truly automatic tracking would need unstable internals. Instead, an
// ``<Inspect>`` wrapper opens a ``InspectorTrackerContext`` that the
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
  useDataflowSchema,
  useDataflowStatus,
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
  /** Cap how often the overlay re-positions on cursor move (ms). */
  hoverDebounceMs?: number;
}

export interface InspectProps {
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
   * Forwarded to the wrapper ``<div>``. Inspector adds its own
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
  hoverDebounceMs: number;
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
  hoverDebounceMs = 60,
}: InspectorProviderProps) {
  // Region registry. ``regionsRef`` is the source of truth (no re-render on
  // change); ``regionVersion`` only ticks when the overlay needs to recheck.
  const regionsRef = useRef<Map<string, InspectableRegion>>(new Map());
  const [, setRegionVersion] = useState(0);

  const registerRegion = useCallback((region: InspectableRegion) => {
    regionsRef.current.set(region.id, region);
    // The overlay reads from ``regionsRef`` directly on each cursor event;
    // we still bump the version so a freshly mounted region surfaces in the
    // initial render before any cursor movement.
    setRegionVersion((v) => v + 1);
  }, []);

  const unregisterRegion = useCallback((id: string) => {
    regionsRef.current.delete(id);
  }, []);

  const state = useMemo<InspectorState>(
    () => ({
      enabled,
      hoverDebounceMs,
      registerRegion,
      unregisterRegion,
    }),
    [enabled, hoverDebounceMs, registerRegion, unregisterRegion],
  );

  return (
    <InspectorContext.Provider value={state}>
      {children}
      {enabled && <InspectorOverlay regionsRef={regionsRef} />}
    </InspectorContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// <Inspect>
// ---------------------------------------------------------------------------

export function Inspect({
  children,
  vars: explicitVars,
  label,
  style,
  className,
}: InspectProps) {
  const inspector = useContext(InspectorContext);
  const id = useId();
  const ref = useRef<HTMLDivElement | null>(null);

  // Reset every render so descendants that conditionally drop a hook stop
  // contributing immediately. Tracker captures into a ref so the
  // post-render effect can read the final snapshot below.
  const collectedRef = useRef<Set<string>>(new Set());
  collectedRef.current = new Set();
  const tracker = useMemo<{ track: (n: string) => void }>(
    () => ({ track: (n) => collectedRef.current.add(n) }),
    [],
  );

  // Re-register on every render — descendant hook order may have shifted,
  // so the var set is potentially stale across commits.
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

// ---------------------------------------------------------------------------
// <InspectorOverlay>
// ---------------------------------------------------------------------------

interface OverlayState {
  /** Region the cursor is currently over (null when off-region). */
  hoverRegion: InspectableRegion | null;
  /** Region that's been pinned (sticky until the user clicks outside). */
  pinnedRegion: InspectableRegion | null;
  /** Cursor position used to anchor the popover. */
  cursor: { x: number; y: number } | null;
}

const INITIAL_OVERLAY_STATE: OverlayState = {
  hoverRegion: null,
  pinnedRegion: null,
  cursor: null,
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

  const debounceMs = inspector?.hoverDebounceMs ?? 60;

  // Cursor / hover tracking: a single document-level listener walks up the
  // DOM looking for an element with ``data-inspect-id``. Cheaper than
  // registering listeners on every region. A ~200 ms grace period before
  // clearing hover state lets the cursor traverse the gap between the
  // inspectable and the popover without flickering it shut.
  useEffect(() => {
    if (!inspector?.enabled) return;
    let lastMove = 0;

    const cancelClear = () => {
      if (clearTimerRef.current !== null) {
        window.clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
    };

    const onMove = (e: MouseEvent) => {
      const now = performance.now();
      if (now - lastMove < debounceMs) return;
      lastMove = now;

      const target = e.target as HTMLElement | null;
      if (!target) return;

      // While the cursor sits inside the popover we keep whatever the
      // current hover/pinned region is — moving from the inspectable into
      // the popover must not collapse it (the user is reaching for it).
      if (popoverRef.current?.contains(target)) {
        cancelClear();
        return;
      }

      const inspectEl = target.closest<HTMLElement>("[data-inspect-id]");
      const id = inspectEl?.getAttribute("data-inspect-id") ?? null;
      const region = id ? regionsRef.current.get(id) ?? null : null;

      if (region) {
        cancelClear();
        setState((prev) => {
          if (prev.pinnedRegion) return prev;
          if (prev.hoverRegion?.id === region.id && prev.cursor) {
            return { ...prev, cursor: { x: e.clientX, y: e.clientY } };
          }
          return {
            ...prev,
            hoverRegion: region,
            cursor: { x: e.clientX, y: e.clientY },
          };
        });
        return;
      }

      // Off both the inspectable and the popover — schedule a clear.
      if (clearTimerRef.current === null) {
        clearTimerRef.current = window.setTimeout(() => {
          clearTimerRef.current = null;
          setState((prev) =>
            prev.pinnedRegion
              ? prev
              : { ...prev, hoverRegion: null, cursor: null },
          );
        }, 200);
      }
    };

    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      // Click inside the popover: ignore (handled by popover internals).
      if (popoverRef.current?.contains(target)) return;
      const inspectEl = target.closest<HTMLElement>("[data-inspect-id]");
      const id = inspectEl?.getAttribute("data-inspect-id") ?? null;
      if (id) {
        const region = regionsRef.current.get(id) ?? null;
        setState({
          hoverRegion: region,
          pinnedRegion: region,
          cursor: { x: e.clientX, y: e.clientY },
        });
      } else {
        setState(INITIAL_OVERLAY_STATE);
      }
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("click", onClick);
      cancelClear();
    };
  }, [inspector?.enabled, debounceMs, regionsRef]);

  // Visible region precedence: pinned > hover.
  const region = state.pinnedRegion ?? state.hoverRegion;
  if (!region || !state.cursor) return null;

  return createPortal(
    <OverlayPopover
      popoverRef={popoverRef}
      region={region}
      cursor={state.cursor}
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
  cursor: { x: number; y: number };
  pinned: boolean;
  onUnpin: () => void;
  popoverRef: React.MutableRefObject<HTMLDivElement | null>;
}

function OverlayPopover({
  region,
  cursor,
  pinned,
  onUnpin,
  popoverRef,
}: OverlayPopoverProps) {
  const graph = useDataflowGraph();
  const schema = useDataflowSchema();
  const status = useDataflowStatus();
  const values = useDataflowValuesSnapshot();
  const [hoveredVar, setHoveredVar] = useState<string | null>(null);
  const [pinnedVar, setPinnedVar] = useState<string | null>(null);

  // Unified ancestor closure: every variable that feeds the region.
  const subgraphNodes = useMemo(() => {
    const nodes = new Set<string>(region.vars);
    for (const v of region.vars) {
      for (const a of getAncestors(graph, v)) nodes.add(a);
    }
    return nodes;
  }, [graph, region.vars]);

  // Topological layering for the mini-DAG. Sources first; cycle-safe.
  const layers = useMemo(
    () => layeredOrder(subgraphNodes, graph),
    [subgraphNodes, graph],
  );

  // Per-variable arrival time relative to the current run start. ``ts``
  // and ``runStartedAtWall`` share a ``Date.now()`` time basis, so the
  // subtraction is meaningful. If the value is from an earlier run we
  // can't place it on the current run's axis, so it shows as pending.
  const timing = useMemo(() => {
    const t: Record<string, number | null> = {};
    let maxArrived = 0;
    const start = status.runStartedAtWall;
    for (const name of subgraphNodes) {
      const v = values[name];
      const ms =
        v && start !== null && v.runId === status.runId
          ? Math.max(0, Math.round(v.ts - start))
          : null;
      t[name] = ms;
      if (ms !== null && ms > maxArrived) maxArrived = ms;
    }
    return { perVar: t, total: Math.max(maxArrived, status.elapsedMs ?? 0, 1) };
  }, [
    subgraphNodes,
    values,
    status.runId,
    status.elapsedMs,
    status.runStartedAtWall,
  ]);

  const visibleVar = pinnedVar ?? hoveredVar;
  const inputNames = useMemo(
    () => new Set(schema?.inputs.map((i) => i.name) ?? []),
    [schema],
  );

  // Anchor the popover next to the cursor; clamp to viewport. Use a tiny
  // gap so the cursor can move directly onto the popover without a
  // mouse-leave gap.
  const popoverStyle = useMemo<CSSProperties>(() => {
    const W = 720;
    const H = 480;
    const margin = 12;
    const gap = 6;
    let left = cursor.x + gap;
    let top = cursor.y + gap;
    if (left + W + margin > window.innerWidth)
      left = Math.max(margin, cursor.x - W - gap);
    if (top + H + margin > window.innerHeight)
      top = Math.max(margin, window.innerHeight - H - margin);
    return { ...styles.popover, top, left, width: W, maxHeight: H };
  }, [cursor]);

  return (
    <div
      ref={popoverRef}
      style={popoverStyle}
      onClick={(e) => e.stopPropagation()}
      data-dataflow-inspector-popover=""
    >
      <header style={styles.header}>
        <strong>{region.label ?? "inspect"}</strong>
        <span style={styles.headerSub}>
          reads <code>{region.vars.join(", ") || "—"}</code> · subgraph:{" "}
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
            layers={layers}
            graph={graph}
            inputNames={inputNames}
            timing={timing}
            hoveredVar={visibleVar}
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
              kind={
                inputNames.has(visibleVar)
                  ? "input"
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
};

function kindOf(name: string, schema: ReturnType<typeof useDataflowSchema>): Kind | "input" {
  if (!schema) return "any";
  const out = schema.outputs.find((o) => o.name === name);
  if (out) return out.kind;
  return "any";
}

// ---------------------------------------------------------------------------
// Mini-DAG
// ---------------------------------------------------------------------------

interface MiniDagProps {
  layers: readonly (readonly string[])[];
  graph: VariableGraph;
  inputNames: Set<string>;
  timing: { perVar: Record<string, number | null>; total: number };
  hoveredVar: string | null;
  onHoverVar: (name: string | null) => void;
  onPinVar: (name: string) => void;
}

function MiniDag({
  layers,
  graph,
  inputNames,
  timing,
  hoveredVar,
  onHoverVar,
  onPinVar,
}: MiniDagProps) {
  // Layered list: each layer is a row; within layers, alphabetical so the
  // ordering is stable across renders. We render as cards rather than
  // free-floating SVG nodes — easier to scan, and the textual edges are
  // explicit in the "←" line under each card.
  const flat = layers.flatMap((layer) =>
    [...layer].sort().map((name, idx) => ({ name, layerIdx: layers.indexOf(layer), idx })),
  );

  return (
    <div style={styles.dagList}>
      {flat.map(({ name }) => {
        const isInput = inputNames.has(name);
        const isHovered = hoveredVar === name;
        const arrived = timing.perVar[name];
        const deps = getDirectDeps(graph, name);
        return (
          <button
            type="button"
            key={name}
            aria-label={`inspect ${name}`}
            style={{
              ...styles.dagRow,
              ...(isHovered ? styles.dagRowHovered : {}),
            }}
            onMouseEnter={() => onHoverVar(name)}
            onMouseLeave={() => onHoverVar(null)}
            onClick={(e) => {
              e.stopPropagation();
              onPinVar(name);
            }}
          >
            <div style={styles.dagRowMain}>
              <span style={styles.dagName}>{name}</span>
              {isInput ? (
                <span style={styles.tagInput}>input</span>
              ) : null}
              {deps.length > 0 && (
                <span style={styles.dagDeps}>
                  ← {deps.join(", ")}
                </span>
              )}
            </div>
            <MiniWaterfall
              arrivedMs={arrived}
              totalMs={timing.total}
              parentArrivedMs={Math.max(
                0,
                ...deps.map((d) => timing.perVar[d] ?? 0),
              )}
            />
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MiniWaterfall — per-variable timing bar
// ---------------------------------------------------------------------------

interface MiniWaterfallProps {
  arrivedMs: number | null;
  totalMs: number;
  /** Latest parent arrival; used as the bar's left edge for "wait" timing. */
  parentArrivedMs: number;
}

function MiniWaterfall({
  arrivedMs,
  totalMs,
  parentArrivedMs,
}: MiniWaterfallProps) {
  if (arrivedMs === null) {
    return (
      <div style={styles.waterfallRow}>
        <div style={styles.waterfallTrack} />
        <span style={styles.waterfallPending}>—</span>
      </div>
    );
  }
  const start = Math.max(0, Math.min(parentArrivedMs, arrivedMs));
  const leftPct = (start / totalMs) * 100;
  const widthPct = Math.max(2, ((arrivedMs - start) / totalMs) * 100);
  return (
    <div style={styles.waterfallRow}>
      <div
        style={styles.waterfallTrack}
        title={`arrived at +${arrivedMs}ms (since parent: ~${arrivedMs - start}ms)`}
      >
        <div
          style={{
            ...styles.waterfallBar,
            left: `${leftPct}%`,
            width: `${widthPct}%`,
          }}
        />
      </div>
      <span style={styles.waterfallLabel}>{arrivedMs}ms</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ValuePreview — kind-dispatched rich rendering
// ---------------------------------------------------------------------------

/**
 * Wrapper that calls ``useDataflowValue`` for the previewed variable so
 * the inspector implicitly subscribes to whatever the user pins. The
 * subscription is dropped as soon as the user picks a different node or
 * unpins (the wrapper unmounts and the refcount goes back to zero), so
 * the inspector never adds permanent overhead to ``/run`` requests.
 */
function PreviewPanelWithSubscription({
  name,
  update,
  kind,
}: {
  name: string;
  update: VarUpdate | undefined;
  kind: Kind | "input";
}) {
  // Inputs don't flow through ``var`` events, so don't auto-subscribe.
  const isInput = kind === "input";
  const live = useDataflowValue(isInput ? "" : name);
  // Prefer the freshly-subscribed value if the snapshot is stale.
  const effective: VarUpdate | undefined =
    update && update.value !== undefined
      ? update
      : live !== undefined
        ? ({ name, value: live, kind: "any", encoding: "json", runId: "", ts: 0 } as VarUpdate)
        : undefined;
  return <ValuePreview name={name} update={effective} kind={kind} />;
}

function ValuePreview({
  name,
  update,
  kind,
}: {
  name: string;
  update: VarUpdate | undefined;
  kind: Kind | "input";
}) {
  if (!update) {
    return (
      <div style={styles.previewBody}>
        <h4 style={styles.previewTitle}>{name}</h4>
        <p style={styles.placeholder}>
          (no value yet — subscribe to it or trigger a run)
        </p>
      </div>
    );
  }
  const value = update.value;
  return (
    <div style={styles.previewBody}>
      <h4 style={styles.previewTitle}>
        {name}
        <span style={styles.previewKind}>{kind}</span>
      </h4>
      <PreviewByKind value={value} kind={update.kind} />
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
        // Trusted source: comes from the dataflow API the user wrote.
        dangerouslySetInnerHTML={{ __html: value }}
      />
    );
  }
  if (kind === "string") {
    const s = String(value);
    if (s.length < 200)
      return <pre style={styles.scalarPreview}>{s}</pre>;
    return <pre style={styles.longStringPreview}>{s}</pre>;
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
  // Lightweight collapsible JSON renderer; no deps. Keeps to ~3 levels by
  // default; click to expand deeper. Loops on circular refs are guarded
  // against by limiting recursion depth and by ``JSON.stringify`` later.
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
// Internals
// ---------------------------------------------------------------------------

/**
 * Layered topological order over a node subset. Each layer is a list of
 * names whose direct deps live in earlier layers. Cycle-safe (orphan nodes
 * land in their own trailing layer).
 */
function layeredOrder(
  nodes: Set<string>,
  graph: VariableGraph,
): readonly (readonly string[])[] {
  const layer = new Map<string, number>();
  const visiting = new Set<string>();
  function depth(name: string): number {
    const cached = layer.get(name);
    if (cached !== undefined) return cached;
    if (visiting.has(name)) return 0; // cycle guard
    visiting.add(name);
    const deps = getDirectDeps(graph, name).filter((d) => nodes.has(d));
    const d = deps.length === 0 ? 0 : 1 + Math.max(...deps.map(depth));
    visiting.delete(name);
    layer.set(name, d);
    return d;
  }
  for (const n of nodes) depth(n);
  const buckets = new Map<number, string[]>();
  for (const [name, d] of layer) {
    const b = buckets.get(d) ?? [];
    b.push(name);
    buckets.set(d, b);
  }
  return Array.from(buckets.keys())
    .sort((a, b) => a - b)
    .map((d) => buckets.get(d)!);
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
  },
  header: {
    padding: "8px 12px",
    borderBottom: "1px solid #e5e7eb",
    background: "#f8f9fa",
    display: "flex",
    alignItems: "baseline",
    gap: 8,
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
    flex: "0 0 360px",
    overflow: "auto",
    borderRight: "1px solid #e5e7eb",
    padding: "8px",
  },
  previewPanel: {
    flex: 1,
    overflow: "auto",
    padding: "10px 12px",
  },
  dagList: { display: "flex", flexDirection: "column", gap: 4 },
  dagRow: {
    padding: "6px 8px",
    borderRadius: 6,
    cursor: "pointer",
    border: "1px solid transparent",
    background: "#fff",
    display: "block",
    width: "100%",
    textAlign: "left",
    font: "inherit",
    color: "inherit",
  },
  dagRowHovered: {
    background: "#e7f5ff",
    borderColor: "#74c0fc",
  },
  dagRowMain: {
    display: "flex",
    alignItems: "baseline",
    gap: 6,
    fontFamily: "ui-monospace, SFMono-Regular, monospace",
    fontSize: 12,
  },
  dagName: { fontWeight: 600, color: "#1864ab" },
  dagDeps: { color: "#6c757d", fontSize: 11 },
  tagInput: {
    fontSize: 9,
    padding: "1px 5px",
    borderRadius: 999,
    background: "#fff3bf",
    color: "#7a4a00",
    fontWeight: 600,
    textTransform: "uppercase",
  },
  waterfallRow: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginTop: 4,
  },
  waterfallTrack: {
    position: "relative",
    height: 6,
    flex: 1,
    background: "#f1f3f5",
    borderRadius: 3,
    minWidth: 40,
  },
  waterfallBar: {
    position: "absolute",
    top: 0,
    bottom: 0,
    background: "linear-gradient(90deg, #4361ee, #3a0ca3)",
    borderRadius: 3,
  },
  waterfallLabel: {
    fontSize: 10,
    color: "#6c757d",
    minWidth: 36,
    textAlign: "right",
    fontVariantNumeric: "tabular-nums",
  },
  waterfallPending: {
    color: "#adb5bd",
    fontSize: 10,
    minWidth: 36,
    textAlign: "right",
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
  tableWrap: { overflow: "auto", maxHeight: 360 },
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
  imagePreview: { maxWidth: "100%", maxHeight: 360, borderRadius: 4 },
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

