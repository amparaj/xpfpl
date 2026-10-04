// Number and date formats, shared so every page shows the same kind of number the same way
// (the dashboard's rules: points and xP to two decimals, chances as percentages).

export const POSITIONS: Record<number, string> = { 1: "GKP", 2: "DEF", 3: "MID", 4: "FWD" };
/** FPL's player status codes, in words ("a", available, has none). */
export const STATUS: Record<string, string> = { d: "Doubtful", i: "Injured", s: "Suspended", u: "Unavailable", n: "Not in squad" };

/** FPL's flag in words: "Doubtful 75%", but just "Injured" (or Suspended, ...): FPL gives those 0%, which says nothing more. */
export function flagLabel(status: string, chance: number | null | undefined): string {
  const word = STATUS[status] ?? "Available";
  return status === "d" && chance != null ? `${word} ${chance}%` : word;
}

const missing = (v: unknown): v is null | undefined => v === null || v === undefined || Number.isNaN(v as number);

export const pts = (v: number | null | undefined) => (missing(v) ? "–" : v.toFixed(2));
export const int = (v: number | null | undefined) => (missing(v) ? "–" : Math.round(v).toLocaleString());
export const signed = (v: number | null | undefined, digits = 2) => {
  if (missing(v)) return "–";
  const r = Number(v.toFixed(digits));          // so -0.04 at one decimal is "0.0", not "−0.0"
  return `${r > 0 ? "+" : r < 0 ? "−" : ""}${Math.abs(r).toFixed(digits)}`;
};
export const pct = (v: number | null | undefined, digits = 0) => (missing(v) ? "–" : `${(v * 100).toFixed(digits)}%`);
export const money = (v: number | null | undefined) => (missing(v) ? "–" : `£${v.toFixed(1)}m`);
export const dec = (v: number | null | undefined, digits = 2) => (missing(v) ? "–" : v.toFixed(digits));
export const compact = (v: number | null | undefined) =>
  missing(v) ? "–" : new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(v);

export const when = (iso: string | null | undefined) =>
  iso
    ? new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
    : "–";
export const day = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "–";

export const isMissing = missing;

/** Risk from the chance of 2 points or fewer in the gameweek: how often the pick lets you down.
 * Among players on 3+ xP about one in ten is Low (45% or less), the rest split Medium / High at 60%. */
export const risk = (pBlank: number | null | undefined): "Low" | "Medium" | "High" | null =>
  missing(pBlank) ? null : pBlank <= 0.45 ? "Low" : pBlank <= 0.6 ? "Medium" : "High";
