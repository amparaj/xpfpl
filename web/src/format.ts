// Number and date formats, shared so every page shows the same kind of number the same way
// (the dashboard's rules: points and xP to two decimals, chances as percentages).

export const POSITIONS: Record<number, string> = { 1: "GKP", 2: "DEF", 3: "MID", 4: "FWD" };

const missing = (v: unknown): v is null | undefined => v === null || v === undefined || Number.isNaN(v as number);

export const pts = (v: number | null | undefined) => (missing(v) ? "–" : v.toFixed(2));
export const int = (v: number | null | undefined) => (missing(v) ? "–" : Math.round(v).toLocaleString());
export const signed = (v: number | null | undefined, digits = 2) =>
  missing(v) ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`;
export const pct = (v: number | null | undefined, digits = 0) => (missing(v) ? "–" : `${(v * 100).toFixed(digits)}%`);
export const money = (v: number | null | undefined) => (missing(v) ? "–" : `£${v.toFixed(1)}m`);
export const dec = (v: number | null | undefined, digits = 2) => (missing(v) ? "–" : v.toFixed(digits));
export const compact = (v: number | null | undefined) =>
  missing(v) ? "–" : new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(v);

export const when = (iso: string | null | undefined) =>
  iso
    ? new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })
    : "–";
export const day = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "–";

export const isMissing = missing;
