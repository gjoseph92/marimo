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
//   - Pin via either the "pin" button in the popover header or by
//     clicking the inspectable region itself. Once pinned, the
//     popover stays put and the inspector stops reacting to hovers
//     and outside clicks until you explicitly unpin. This lets you
//     interact with the surrounding UI (move sliders, click buttons,
//     edit inputs) while keeping the inspector locked to the same
//     region.
//   - Unpin via the same toggle button in the popover header (its
//     label flips to "unpin" while pinned). After unpinning the
//     popover continues showing the current region in hover mode,
//     and fades when the cursor leaves both the region and the
//     popover.
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
        setState((prev) =>
          // While pinned, ignore hover changes — the user is
          // interacting with the surrounding UI and the inspector
          // should stay locked to the pinned region.
          prev.pinnedRegion || prev.hoverRegion?.id === region.id
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
          setState((prev) =>
            // Don't disturb the pinned region; pinned wins for
            // display, and we want to leave hover untouched so
            // unpinning later doesn't flash a stale region.
            prev.pinnedRegion ? prev : { ...prev, hoverRegion: null },
          );
        }, 100);
      }
    };

    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      // Clicks inside the popover are owned by its internal handlers
      // (e.g. the Unpin button, DAG node selection).
      if (popoverRef.current?.contains(target)) return;

      setState((prev) => {
        // While pinned, the inspector ignores outside clicks so the
        // user can freely interact with surrounding UI (sliders,
        // buttons, dropdowns) without losing context. Unpin via the
        // toggle button in the popover header.
        if (prev.pinnedRegion) return prev;
        // Not pinned: clicking an inspectable pins it. Clicks on
        // non-inspectable elements are a no-op.
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
      onTogglePin={() =>
        setState((prev) => ({
          // Toggle: when pinned, unpin (popover keeps showing the
          // current region in hover mode and can fade as usual);
          // when unpinned, pin to the currently-displayed region so
          // the user can interact with surrounding UI without
          // losing it. Provides an explicit affordance for the
          // common case where the popover is covering the
          // inspectable element and the click-to-pin shortcut is
          // unreachable.
          ...prev,
          pinnedRegion: prev.pinnedRegion ? null : prev.hoverRegion,
        }))
      }
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
  onTogglePin: () => void;
  popoverRef: React.MutableRefObject<HTMLDivElement | null>;
}

const POPOVER_WIDTH = 760;
const POPOVER_HEIGHT = 440;

function OverlayPopover({
  region,
  pinned,
  onTogglePin,
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

  // Stable node list (membership only — layout is computed inside
  // ``MiniDag`` from the graph itself).
  const nodes = useMemo(() => [...subgraphNodes], [subgraphNodes]);

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
        <button
          type="button"
          onClick={onTogglePin}
          style={styles.pinBtn}
          aria-pressed={pinned}
          title={
            pinned
              ? "Unpin (popover follows your cursor again)"
              : "Pin in place (interact with surrounding UI without dismissing)"
          }
        >
          {pinned ? "unpin" : "pin"}
        </button>
      </header>

      <div style={styles.body}>
        <section style={styles.dagPanel}>
          <MiniDag
            nodes={nodes}
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
              Hover any node in the graph to preview its value. Click to
              pin the preview.
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
// Mini-DAG — column-compressed, top-to-bottom DAG drawn all at once.
//
// Goal: stay narrow even on graphs with wide fan-in or long lineages.
//
// Column assignment ("compressed depth"): a node shares its parent's
// column iff the parent has only one child *and* the child has only
// one parent (i.e., a 1-to-1 chain edge). Otherwise the node lives
// one column past its deepest parent. The effect is that long single
// lineages collapse into one vertical column instead of marching
// rightward.
//
// Row assignment: chain runs stay contiguous within a column. Within
// a column we walk each chain head — a node whose chain predecessor
// (the same-column parent) is absent — and lay its chain successors
// out directly below it.
//
// Edge routing depends on the relationship of the endpoints:
//   - Same column ⇒ "chain" edge: a single vertical segment from the
//     parent's bottom to the child's top.
//   - Different column, child has multiple parents ⇒ "fan-in": exit
//     the parent's right side, run horizontally a few px past the
//     child's left edge, drop straight down into the child's top.
//     All parents of the same child reuse this x, so they visually
//     converge into one drop into the child's top.
//   - Different column, child has a single parent (so this *is* a
//     fan-out from the parent) ⇒ "fan-out": exit the parent's right
//     side, run to a merge column just left of the child, drop to the
//     child's vertical center, then enter the child's left side.
//
// Layout is static — selection only changes color emphasis.
// ---------------------------------------------------------------------------

const NODE_HEIGHT = 24;
// Sibling vs. level gap: tighter spacing for nodes at the same depth
// (visually a sibling group), wider for nodes at different depths
// (visually a downstream step). Without the difference, chained
// nodes look like siblings.
const ROW_GAP = 4;
const LEVEL_GAP = 18;
const COL_GAP = 5;
const MERGE_GAP = 3;
// Fan-in edges drop into the child's top this many pixels past its
// left edge, so the merge column hugs the parents' right side instead
// of running all the way to the child's center.
const FANIN_INSET = 4;
const PADDING_X = 12;
const PADDING_Y = 12;
const MIN_NODE_WIDTH = 64;
const MAX_NODE_WIDTH = 140;
const FONT_PX = 12;

interface NodePos {
  x: number; // left edge
  y: number; // top edge
  width: number;
  column: number; // visual column index
}

type EdgeStyle = "chain" | "fanin" | "fanout";

interface Edge {
  from: string;
  to: string;
  style: EdgeStyle;
}

interface DagLayout {
  positions: Map<string, NodePos>;
  edges: ReadonlyArray<Edge>;
  width: number;
  height: number;
}

function layoutDag(nodes: readonly string[], graph: VariableGraph): DagLayout {
  const set = new Set(nodes);

  // Adjacency in the subgraph.
  const childrenOf = new Map<string, string[]>();
  const parentsOf = new Map<string, string[]>();
  for (const n of nodes) {
    childrenOf.set(n, []);
    parentsOf.set(n, getDirectDeps(graph, n).filter((p) => set.has(p)));
  }
  for (const n of nodes) {
    for (const p of parentsOf.get(n)!) childrenOf.get(p)!.push(n);
  }

  // Compressed depth — chain edges don't increment the column.
  const column = new Map<string, number>();
  const visiting = new Set<string>();
  function computeColumn(name: string): number {
    const cached = column.get(name);
    if (cached !== undefined) return cached;
    if (visiting.has(name)) return 0;
    visiting.add(name);
    const parents = parentsOf.get(name)!;
    let v: number;
    if (parents.length === 0) {
      v = 0;
    } else if (
      parents.length === 1 &&
      childrenOf.get(parents[0])!.length === 1
    ) {
      // 1-to-1 chain edge — share the parent's column.
      v = computeColumn(parents[0]);
    } else {
      v = Math.max(...parents.map(computeColumn)) + 1;
    }
    visiting.delete(name);
    column.set(name, v);
    return v;
  }
  for (const n of nodes) computeColumn(n);

  // True depth (longest path from a source) — chain edges *do*
  // increment this. Used purely to widen the vertical gap between
  // adjacent rows whose depths differ, so chained children visually
  // sit below their parent rather than alongside siblings.
  const depth = new Map<string, number>();
  const visitingD = new Set<string>();
  function computeDepth(name: string): number {
    const cached = depth.get(name);
    if (cached !== undefined) return cached;
    if (visitingD.has(name)) return 0;
    visitingD.add(name);
    const parents = parentsOf.get(name)!;
    const v =
      parents.length === 0 ? 0 : 1 + Math.max(...parents.map(computeDepth));
    visitingD.delete(name);
    depth.set(name, v);
    return v;
  }
  for (const n of nodes) computeDepth(n);

  const colNodes = new Map<number, string[]>();
  for (const n of nodes) {
    const c = column.get(n)!;
    if (!colNodes.has(c)) colNodes.set(c, []);
    colNodes.get(c)!.push(n);
  }

  // Topological order (Kahn), stable on ties.
  const inDeg = new Map<string, number>();
  for (const n of nodes) inDeg.set(n, parentsOf.get(n)!.length);
  const queue: string[] = nodes
    .filter((n) => inDeg.get(n) === 0)
    .sort((a, b) => a.localeCompare(b));
  const topo: string[] = [];
  while (queue.length) {
    const n = queue.shift()!;
    topo.push(n);
    for (const c of childrenOf.get(n)!) {
      const d = inDeg.get(c)! - 1;
      inDeg.set(c, d);
      if (d === 0) {
        let i = queue.length;
        while (i > 0 && queue[i - 1].localeCompare(c) > 0) i--;
        queue.splice(i, 0, c);
      }
    }
  }
  const topoIdx = new Map<string, number>();
  topo.forEach((n, i) => topoIdx.set(n, i));

  // Row assignment: process columns in order. Within a column, walk
  // each chain head down its sole same-column successor so chained
  // nodes stay contiguous (and disjoint chain runs in the same column
  // never get visually interleaved).
  const sortedCols = [...colNodes.keys()].sort((a, b) => a - b);
  const row = new Map<string, number>();
  let nextRow = 0;
  for (const c of sortedCols) {
    const ns = colNodes.get(c)!;
    const inCol = new Set(ns);
    const chainPrev = new Map<string, string | null>();
    for (const n of ns) {
      chainPrev.set(n, parentsOf.get(n)!.find((p) => inCol.has(p)) ?? null);
    }
    const heads = ns
      .filter((n) => chainPrev.get(n) === null)
      .sort((a, b) => (topoIdx.get(a) ?? 0) - (topoIdx.get(b) ?? 0));
    for (const head of heads) {
      let curr: string | null = head;
      while (curr !== null) {
        row.set(curr, nextRow++);
        // The chain rule guarantees at most one same-column child.
        curr = childrenOf.get(curr)!.find((c) => inCol.has(c)) ?? null;
      }
    }
  }

  // Pixel layout — nodes centered within their column.
  const widthFor = (name: string) =>
    Math.max(
      MIN_NODE_WIDTH,
      Math.min(MAX_NODE_WIDTH, name.length * (FONT_PX * 0.62) + 22),
    );
  const colWidth = sortedCols.map((c) =>
    Math.max(MIN_NODE_WIDTH, ...colNodes.get(c)!.map(widthFor)),
  );
  const colIdx = new Map<number, number>();
  sortedCols.forEach((c, i) => colIdx.set(c, i));
  const colX: number[] = [];
  let cx = PADDING_X;
  for (let i = 0; i < sortedCols.length; i++) {
    colX.push(cx);
    cx += colWidth[i] + COL_GAP;
  }

  // Depth-aware row Y positions: small gap when adjacent rows are at
  // the same depth (sibling group), wider gap when they differ
  // (downstream step). Built once for all rows and indexed by row.
  const nodeAtRow: string[] = new Array(nextRow);
  for (const [n, r] of row) nodeAtRow[r] = n;
  const rowY: number[] = [];
  let cy = PADDING_Y;
  for (let r = 0; r < nextRow; r++) {
    if (r > 0) {
      const sameLevel =
        depth.get(nodeAtRow[r]) === depth.get(nodeAtRow[r - 1]);
      cy += NODE_HEIGHT + (sameLevel ? ROW_GAP : LEVEL_GAP);
    }
    rowY.push(cy);
  }

  const positions = new Map<string, NodePos>();
  for (const n of nodes) {
    const c = column.get(n)!;
    const ci = colIdx.get(c)!;
    const r = row.get(n)!;
    const w = widthFor(n);
    const center = colX[ci] + colWidth[ci] / 2;
    positions.set(n, {
      x: center - w / 2,
      y: rowY[r],
      width: w,
      column: ci,
    });
  }

  const edges: Edge[] = [];
  for (const n of nodes) {
    for (const p of parentsOf.get(n)!) {
      let style: EdgeStyle;
      if (column.get(p) === column.get(n)) {
        style = "chain";
      } else if (parentsOf.get(n)!.length > 1) {
        style = "fanin";
      } else {
        style = "fanout";
      }
      edges.push({ from: p, to: n, style });
    }
  }

  const totalWidth =
    sortedCols.length === 0
      ? PADDING_X * 2
      : colX[sortedCols.length - 1] +
        colWidth[sortedCols.length - 1] +
        PADDING_X;
  const totalHeight =
    nextRow === 0
      ? PADDING_Y * 2
      : rowY[nextRow - 1] + NODE_HEIGHT + PADDING_Y;

  return { positions, edges, width: totalWidth, height: totalHeight };
}

interface MiniDagProps {
  nodes: readonly string[];
  graph: VariableGraph;
  inputNames: Set<string>;
  selectedVar: string | null;
  onHoverVar: (name: string | null) => void;
  onPinVar: (name: string) => void;
}

function MiniDag({
  nodes,
  graph,
  inputNames,
  selectedVar,
  onHoverVar,
  onPinVar,
}: MiniDagProps) {
  const layout = useMemo(() => layoutDag(nodes, graph), [nodes, graph]);

  // Edges incident to selection get emphasized; everything else stays
  // visible at full structure — only color changes.
  const incidentEdges = useMemo(() => {
    if (!selectedVar) return new Set<string>();
    const s = new Set<string>();
    for (const { from, to } of layout.edges) {
      if (from === selectedVar || to === selectedVar) s.add(`${from}→${to}`);
    }
    return s;
  }, [selectedVar, layout.edges]);

  return (
    <div style={{ width: layout.width, minHeight: layout.height }}>
      <svg
        width={layout.width}
        height={layout.height}
        style={{ display: "block", overflow: "visible" }}
      >
        {/* Edges painted first so nodes overlap them cleanly. */}
        {layout.edges.map(({ from, to, style }) => {
          const a = layout.positions.get(from);
          const b = layout.positions.get(to);
          if (!a || !b) return null;
          const aRight = a.x + a.width;
          const aCenterX = a.x + a.width / 2;
          const aBottom = a.y + NODE_HEIGHT;
          const aCenterY = a.y + NODE_HEIGHT / 2;
          const bLeft = b.x;
          const bCenterX = b.x + b.width / 2;
          const bTop = b.y;
          const bCenterY = b.y + NODE_HEIGHT / 2;
          let d: string;
          if (style === "chain") {
            d = `M ${aCenterX} ${aBottom} V ${bTop}`;
          } else if (style === "fanin") {
            // Multi-parent merge into the child's top, close to its
            // left edge instead of its center. Keeps the horizontal
            // run short (a few px past the widest parent) while still
            // entering through the top. All parents of the same child
            // share this x, so they visually converge into one drop.
            const mx = bLeft + Math.min(b.width / 2, FANIN_INSET);
            d = `M ${aRight} ${aCenterY} H ${mx} V ${bTop}`;
          } else {
            // Fan-out: one parent, multiple children. Enter the
            // child's left side via a merge column just to its left.
            const mx = Math.max(aRight + 2, bLeft - MERGE_GAP);
            d = `M ${aRight} ${aCenterY} H ${mx} V ${bCenterY} H ${bLeft}`;
          }
          const incident = incidentEdges.has(`${from}→${to}`);
          return (
            <path
              key={`${from}→${to}`}
              d={d}
              fill="none"
              stroke={incident ? "#1864ab" : "#adb5bd"}
              strokeWidth={incident ? 2 : 1.25}
              opacity={selectedVar && !incident ? 0.55 : 1}
            />
          );
        })}

        {/* Nodes — labeled rounded pills. Click + hover are handled
            on the foreignObject so they get full HTML button semantics
            (focus ring, ARIA, keyboard activation). */}
        {nodes.map((name) => {
          const p = layout.positions.get(name);
          if (!p) return null;
          const isSelected = name === selectedVar;
          const isInput = inputNames.has(name);
          const stroke = isSelected
            ? "#1864ab"
            : isInput
              ? "#ffc078"
              : "#a5b3c1";
          const fill = isSelected
            ? "#e7f5ff"
            : isInput
              ? "#fff9db"
              : "#f8f9fa";
          return (
            <g key={`node-${name}`}>
              <rect
                x={p.x}
                y={p.y}
                width={p.width}
                height={NODE_HEIGHT}
                rx={6}
                fill={fill}
                stroke={stroke}
                strokeWidth={isSelected ? 2 : 1}
              />
              <foreignObject
                x={p.x}
                y={p.y}
                width={p.width}
                height={NODE_HEIGHT}
              >
                <button
                  type="button"
                  aria-label={`inspect ${name}`}
                  onMouseEnter={() => onHoverVar(name)}
                  onMouseLeave={() => onHoverVar(null)}
                  onClick={(e) => {
                    e.stopPropagation();
                    onPinVar(name);
                  }}
                  style={{
                    ...styles.dagNodeButton,
                    color: isSelected ? "#1864ab" : "#212529",
                    fontWeight: isSelected ? 700 : 500,
                  }}
                >
                  {name}
                </button>
              </foreignObject>
            </g>
          );
        })}
      </svg>
    </div>
  );
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
  pinBtn: {
    border: "1px solid #dee2e6",
    background: "#fff",
    borderRadius: 4,
    padding: "2px 8px",
    fontSize: 11,
    cursor: "pointer",
  },
  body: { display: "flex", flex: 1, minHeight: 0 },
  dagPanel: {
    flex: "0 0 320px",
    overflow: "auto",
    borderRight: "1px solid #e5e7eb",
    padding: 0,
  },
  previewPanel: {
    flex: 1,
    overflow: "auto",
    padding: "10px 12px",
  },
  dagNodeButton: {
    width: "100%",
    height: "100%",
    border: "none",
    background: "transparent",
    cursor: "pointer",
    fontFamily: "ui-monospace, SFMono-Regular, monospace",
    fontSize: 12,
    padding: 0,
    margin: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
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
