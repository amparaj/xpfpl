// Small shared pieces: club chips, stat tiles, legends, the chart wrapper and the sortable table.

import * as Plot from "@observablehq/plot";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useSite } from "../site";

// ---------------------------------------------------------------- club chip

export function Club({ id, code }: { id?: number; code?: number }) {
  const site = useSite();
  const team = id !== undefined ? site.team.get(id) : code !== undefined ? site.teamByCode.get(code) : undefined;
  if (!team) {
    // A club from an earlier season (not in this season's list): name it from the Polymarket table.
    const name = Object.entries(site.meta.polymarket_teams).find(([, c]) => c === code)?.[0];
    return <span className="club" title={name}>{name ? clubShort(name) : "?"}</span>;
  }
  const [bg, fg] = site.meta.club_colours[team.short] ?? ["var(--chip)", "var(--ink)"];
  return (
    <span className="club" style={{ background: bg, color: fg }} title={team.name}>
      {team.short}
    </span>
  );
}

/** "Leicester City FC" -> "LEI": a short name for a club that isn't in this season's list. */
export const clubShort = (name: string) => name.replace(/^AFC /, "").slice(0, 3).toUpperCase();

/** A club's short name by persistent code, this season's or an earlier one's. */
export function useClubName() {
  const site = useSite();
  return (code: number) => {
    const t = site.teamByCode.get(code);
    if (t) return t.short;
    const name = Object.entries(site.meta.polymarket_teams).find(([, c]) => c === code)?.[0];
    return name ? clubShort(name) : String(code);
  };
}

// ---------------------------------------------------------------- stat tiles

export interface TileProps { label: string; value: ReactNode; note?: ReactNode }

export function Tiles({ tiles }: { tiles: TileProps[] }) {
  return (
    <div className="tiles">
      {tiles.map((t) => (
        <div className="tile" key={t.label}>
          <div className="tile-label">{t.label}</div>
          <div className="tile-value">{t.value}</div>
          {t.note !== undefined && <div className="tile-note">{t.note}</div>}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- legend

export function Legend({ items }: { items: { label: string; color: string; kind?: "rect" | "line" | "dot" }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label} className="legend-item">
          <span className={`key key-${i.kind ?? "rect"}`} style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- chart

/** Renders an Observable Plot chart at the container's width, redrawn when it resizes. */
export function Chart({ make, height, ariaLabel }: {
  make: (width: number) => (SVGSVGElement | HTMLElement);
  height?: number;
  ariaLabel: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (!el || width === 0) return;
    const chart = make(width);
    el.replaceChildren(chart);
    return () => chart.remove();
  }, [make, width]);
  return <div ref={ref} className="chart" style={{ minHeight: height }} role="img" aria-label={ariaLabel} />;
}

/** Chart defaults: recessive axes and grid in the text tokens, the site's font. */
export function plotDefaults(width: number): Plot.PlotOptions {
  return {
    width,
    style: { fontFamily: "inherit", fontSize: "12px", background: "transparent", color: "var(--muted)", overflow: "visible" },
    marginLeft: 44,
  };
}

/** Band padding that keeps each bar at most 24px wide, whatever the chart width and bar count. */
export function barPadding(width: number, bars: number, margins = 64): number {
  const band = (width - margins) / Math.max(bars, 1);
  return Math.min(0.9, Math.max(0.35, 1 - 24 / band));
}

// ---------------------------------------------------------------- table

export interface Column<T> {
  key: string;
  label: ReactNode;
  value: (r: T) => number | string | null | undefined;
  render?: (r: T) => ReactNode;
  numeric?: boolean;
  title?: string;
}

export function Table<T>({ columns, data, sort: initialSort, desc: initialDesc = true, limit, rowKey, onRow, selected }: {
  columns: Column<T>[];
  data: T[];
  sort?: string;
  desc?: boolean;
  limit?: number;
  rowKey: (r: T) => string | number;
  onRow?: (r: T) => void;
  selected?: string | number | null;
}) {
  const [sort, setSort] = useState(initialSort ?? null);
  const [desc, setDesc] = useState(initialDesc);
  const [all, setAll] = useState(false);
  const sorted = useMemo(() => {
    const col = columns.find((c) => c.key === sort);
    if (!col) return data;
    const nulls = (v: unknown) => v === null || v === undefined || Number.isNaN(v);
    return [...data].sort((a, b) => {
      const x = col.value(a), y = col.value(b);
      if (nulls(x)) return nulls(y) ? 0 : 1;
      if (nulls(y)) return -1;
      const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
      return desc ? -c : c;
    });
  }, [columns, data, sort, desc]);
  const shown = limit && !all ? sorted.slice(0, limit) : sorted;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.numeric ? "num" : undefined} title={c.title}
                  aria-sort={sort === c.key ? (desc ? "descending" : "ascending") : undefined}>
                <button onClick={() => {
                  if (sort === c.key) setDesc(!desc);
                  else { setSort(c.key); setDesc(!!c.numeric); }
                }}>
                  {c.label}
                  <span className="sort-mark">{sort === c.key ? (desc ? "▼" : "▲") : ""}</span>
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => {
            const key = rowKey(r);
            return (
              <tr key={key} onClick={onRow ? () => onRow(r) : undefined}
                  className={`${onRow ? "clickable" : ""} ${selected === key ? "selected" : ""}`}>
                {columns.map((c) => (
                  <td key={c.key} className={c.numeric ? "num" : undefined}>
                    {c.render ? c.render(r) : String(c.value(r) ?? "–")}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      {limit && sorted.length > limit && (
        <button className="link" onClick={() => setAll(!all)}>
          {all ? "Show fewer" : `Show all ${sorted.length}`}
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- misc

export function Segmented<T extends string | number>({ options, value, onChange, label }: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} role="radio" aria-checked={o.value === value}
                className={o.value === value ? "on" : undefined} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Loading() {
  return <p className="muted">Loading…</p>;
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="note">{children}</p>;
}
